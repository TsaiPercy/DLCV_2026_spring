"""Training script for DETR digit detection (HW2)."""

import argparse
import json
import os
import sys
import time
from datetime import datetime

import numpy as np
import torch
import torch.nn.functional as F
import yaml
from scipy.optimize import linear_sum_assignment
from torch.amp import GradScaler, autocast
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from tqdm import tqdm

from dataset import build_dataloaders
from model import build_model


# ---------------------------------------------------------------------------
# Bounding-box utilities
# ---------------------------------------------------------------------------

def box_cxcywh_to_xyxy(boxes):
    """[cx, cy, w, h] → [x1, y1, x2, y2]."""
    cx, cy, w, h = boxes.unbind(-1)
    return torch.stack(
        [cx - w / 2, cy - h / 2, cx + w / 2, cy + h / 2], dim=-1
    )


def giou_pairwise(b1, b2):
    """Element-wise GIoU for two [N, 4] sets in xyxy format."""
    ix1 = torch.max(b1[:, 0], b2[:, 0])
    iy1 = torch.max(b1[:, 1], b2[:, 1])
    ix2 = torch.min(b1[:, 2], b2[:, 2])
    iy2 = torch.min(b1[:, 3], b2[:, 3])
    inter = (ix2 - ix1).clamp(0) * (iy2 - iy1).clamp(0)

    a1 = (b1[:, 2] - b1[:, 0]).clamp(0) * (b1[:, 3] - b1[:, 1]).clamp(0)
    a2 = (b2[:, 2] - b2[:, 0]).clamp(0) * (b2[:, 3] - b2[:, 1]).clamp(0)
    union = a1 + a2 - inter
    iou = inter / (union + 1e-6)

    ex1 = torch.min(b1[:, 0], b2[:, 0])
    ey1 = torch.min(b1[:, 1], b2[:, 1])
    ex2 = torch.max(b1[:, 2], b2[:, 2])
    ey2 = torch.max(b1[:, 3], b2[:, 3])
    enc = (ex2 - ex1).clamp(0) * (ey2 - ey1).clamp(0)

    return iou - (enc - union) / (enc + 1e-6)


def giou_cost_matrix(pred_boxes, gt_boxes):
    """Compute an [N, M] GIoU cost matrix (pred vs. GT) in cxcywh format."""
    N = pred_boxes.shape[0]
    M = gt_boxes.shape[0]
    pred_xy = box_cxcywh_to_xyxy(pred_boxes)   # [N, 4]
    gt_xy = box_cxcywh_to_xyxy(gt_boxes)       # [M, 4]

    # Expand to [N, M, 4] for pairwise computation
    p = pred_xy.unsqueeze(1).expand(N, M, 4).reshape(N * M, 4)
    g = gt_xy.unsqueeze(0).expand(N, M, 4).reshape(N * M, 4)
    giou = giou_pairwise(p, g).view(N, M)
    return -giou  # cost = negative GIoU


# ---------------------------------------------------------------------------
# Hungarian matching
# ---------------------------------------------------------------------------

@torch.no_grad()
def hungarian_match(logits, boxes, target):
    """
    Match N predictions to M ground-truth objects.

    Args:
        logits: [N, num_classes+1] – raw class logits
        boxes:  [N, 4]             – predicted boxes (cxcywh, normalised)
        target: dict with 'labels' [M] (1-indexed) and 'boxes' [M, 4]

    Returns:
        (pred_idx, gt_idx) LongTensors of matched indices
    """
    M = target["labels"].shape[0]
    if M == 0:
        empty = torch.zeros(0, dtype=torch.long)
        return empty, empty

    probs = logits.softmax(-1)  # [N, C+1]
    # GT labels are 1-indexed; map to 0-indexed for column selection
    gt_cls = target["labels"] - 1  # [M]

    # Classification cost: -prob[gt_class]  → [N, M]
    cls_cost = -probs[:, gt_cls]

    # L1 bbox cost: [N, M]
    gt_boxes = target["boxes"].to(boxes.device)
    l1_cost = torch.cdist(boxes, gt_boxes, p=1)

    # GIoU cost: [N, M]
    g_cost = giou_cost_matrix(boxes, gt_boxes)

    cost = cls_cost + 5.0 * l1_cost + 2.0 * g_cost
    if torch.isnan(cost).any():
        print(f"[NaN debug] logits NaN={torch.isnan(logits).any().item()}")
        print(f"[NaN debug] boxes NaN={torch.isnan(boxes).any().item()}")
        print(f"[NaN debug] gt_boxes NaN={torch.isnan(gt_boxes).any().item()}")
        print(f"[NaN debug] cls_cost NaN={torch.isnan(cls_cost).any().item()}")
        print(f"[NaN debug] l1_cost NaN={torch.isnan(l1_cost).any().item()}")
        print(f"[NaN debug] g_cost NaN={torch.isnan(g_cost).any().item()}")
        raise ValueError("matrix contains invalid numeric entries")
    pred_idx, gt_idx = linear_sum_assignment(cost.cpu().numpy())
    return (
        torch.tensor(pred_idx, dtype=torch.long),
        torch.tensor(gt_idx, dtype=torch.long),
    )


# ---------------------------------------------------------------------------
# Loss
# ---------------------------------------------------------------------------

class DETRLoss(torch.nn.Module):
    """Bipartite matching loss: CE + L1 + GIoU."""

    def __init__(
        self,
        num_classes,
        no_object_weight=0.1,
        lambda_cls=1.0,
        lambda_bbox=5.0,
        lambda_giou=2.0,
    ):
        super().__init__()
        self.num_classes = num_classes
        self.lambda_cls = lambda_cls
        self.lambda_bbox = lambda_bbox
        self.lambda_giou = lambda_giou

        # Down-weight the "no-object" class (index = num_classes)
        weight = torch.ones(num_classes + 1)
        weight[num_classes] = no_object_weight
        self.register_buffer("cls_weight", weight)

    def _compute_single_loss(self, pred_logits, pred_boxes, targets_list):
        """Compute loss for a single set of predictions."""
        B, N, _ = pred_logits.shape
        no_obj = self.num_classes

        cls_losses, bbox_losses, giou_losses = [], [], []
        total_matched = 0

        for b in range(B):
            logits_b = pred_logits[b]
            boxes_b = pred_boxes[b]
            tgt = targets_list[b]
            M = tgt["labels"].shape[0]

            pred_idx, gt_idx = hungarian_match(logits_b, boxes_b, tgt)

            tgt_cls = torch.full(
                (N,), no_obj, dtype=torch.long, device=logits_b.device
            )
            if M > 0 and len(pred_idx) > 0:
                tgt_cls[pred_idx] = tgt["labels"][gt_idx].to(logits_b.device) - 1

            cls_loss = F.cross_entropy(
                logits_b, tgt_cls, weight=self.cls_weight.to(logits_b.device)
            )
            cls_losses.append(cls_loss)

            if M > 0 and len(pred_idx) > 0:
                matched_pred = boxes_b[pred_idx]
                matched_gt = tgt["boxes"][gt_idx].to(boxes_b.device)
                bbox_losses.append(
                    F.l1_loss(matched_pred, matched_gt, reduction="sum")
                )
                giou = giou_pairwise(
                    box_cxcywh_to_xyxy(matched_pred),
                    box_cxcywh_to_xyxy(matched_gt),
                )
                giou_losses.append((1 - giou).sum())
                total_matched += len(pred_idx)

        num_boxes = max(total_matched, 1)
        cls_loss_avg = torch.stack(cls_losses).mean()
        bbox_loss_avg = (
            torch.stack(bbox_losses).sum() / num_boxes
            if bbox_losses
            else pred_logits.new_tensor(0.0)
        )
        giou_loss_avg = (
            torch.stack(giou_losses).sum() / num_boxes
            if giou_losses
            else pred_logits.new_tensor(0.0)
        )

        total = (
            self.lambda_cls * cls_loss_avg
            + self.lambda_bbox * bbox_loss_avg
            + self.lambda_giou * giou_loss_avg
        )
        return total, cls_loss_avg, bbox_loss_avg, giou_loss_avg

    def forward(self, outputs, targets_list):
        # Main loss from final decoder layer
        main_loss, cls_loss, bbox_loss, giou_loss = self._compute_single_loss(
            outputs["pred_logits"], outputs["pred_boxes"], targets_list
        )

        # Auxiliary losses from intermediate decoder layers
        aux_loss = outputs["pred_logits"].new_tensor(0.0)
        if "aux_outputs" in outputs:
            for aux_out in outputs["aux_outputs"]:
                a_loss, _, _, _ = self._compute_single_loss(
                    aux_out["pred_logits"], aux_out["pred_boxes"], targets_list
                )
                aux_loss = aux_loss + a_loss

        total = main_loss + aux_loss

        return {
            "loss": total,
            "cls_loss": cls_loss.detach(),
            "bbox_loss": bbox_loss.detach(),
            "giou_loss": giou_loss.detach(),
        }


# ---------------------------------------------------------------------------
# COCO mAP evaluation (lightweight, no pycocotools dependency)
# ---------------------------------------------------------------------------

def compute_ap(recalls, precisions):
    """Compute area under PR curve using 101-point interpolation."""
    recalls = np.concatenate([[0.0], recalls, [1.0]])
    precisions = np.concatenate([[0.0], precisions, [0.0]])
    for i in range(len(precisions) - 2, -1, -1):
        precisions[i] = max(precisions[i], precisions[i + 1])
    idx = np.where(recalls[1:] != recalls[:-1])[0]
    return np.sum((recalls[idx + 1] - recalls[idx]) * precisions[idx + 1])


@torch.no_grad()
def evaluate_map(model, val_loader, device, iou_thresholds=None, score_thr=0.01):
    """Compute per-class AP and mAP over the validation set."""
    if iou_thresholds is None:
        iou_thresholds = np.arange(0.5, 1.0, 0.05)

    model.eval()
    all_preds = []   # list of (image_id, cls, score, [x1,y1,x2,y2] norm)
    all_gts = {}     # image_id → list of (cls, [x1,y1,x2,y2] norm)

    for images, masks, targets_list in val_loader:
        images = images.to(device)
        masks = masks.to(device)
        outputs = model(images, masks)
        pred_logits = outputs["pred_logits"]
        pred_boxes = outputs["pred_boxes"]

        for b, tgt in enumerate(targets_list):
            img_id = int(tgt["image_id"])
            logits = pred_logits[b]           # [N, C+1]
            boxes = pred_boxes[b]             # [N, 4] cxcywh normalised
            probs = logits.softmax(-1)

            scores, classes = probs[:, :-1].max(dim=-1)
            boxes_xyxy = box_cxcywh_to_xyxy(boxes)

            for n in range(logits.shape[0]):
                s = scores[n].item()
                if s < score_thr:
                    continue
                c = int(classes[n].item()) + 1  # 1-indexed
                b4 = boxes_xyxy[n].tolist()
                all_preds.append((img_id, c, s, b4))

            # Ground truth
            gt_boxes_xyxy = box_cxcywh_to_xyxy(tgt["boxes"])
            all_gts[img_id] = [
                (int(lbl.item()), gt_boxes_xyxy[i].tolist())
                for i, lbl in enumerate(tgt["labels"])
            ]

    # COCO-style mAP: average AP over all IoU thresholds and classes
    num_classes = 10
    all_aps = []  # one entry per (cls, iou_thr)

    for iou_thr in iou_thresholds:
        for cls in range(1, num_classes + 1):
            cls_preds = sorted(
                [(s, img_id, b4) for img_id, c, s, b4 in all_preds if c == cls],
                key=lambda x: -x[0],
            )
            num_gt = sum(
                sum(1 for lbl, _ in gts if lbl == cls)
                for gts in all_gts.values()
            )
            if num_gt == 0:
                continue

            tp = np.zeros(len(cls_preds))
            fp = np.zeros(len(cls_preds))
            matched = {img_id: set() for img_id in all_gts}

            for k, (_, img_id, pred_box) in enumerate(cls_preds):
                gts_for_img = [
                    (i, b4)
                    for i, (lbl, b4) in enumerate(all_gts.get(img_id, []))
                    if lbl == cls
                ]
                best_iou, best_j = 0.0, -1
                for j, gt_box in gts_for_img:
                    iou = _box_iou_single(pred_box, gt_box)
                    if iou > best_iou:
                        best_iou, best_j = iou, j

                if best_iou >= iou_thr and best_j not in matched[img_id]:
                    tp[k] = 1
                    matched[img_id].add(best_j)
                else:
                    fp[k] = 1

            tp_cum = np.cumsum(tp)
            fp_cum = np.cumsum(fp)
            rec = tp_cum / num_gt
            prec = tp_cum / (tp_cum + fp_cum + 1e-9)
            all_aps.append(compute_ap(rec, prec))

    return float(np.mean(all_aps)) if all_aps else 0.0


def _box_iou_single(b1, b2):
    ix1, iy1 = max(b1[0], b2[0]), max(b1[1], b2[1])
    ix2, iy2 = min(b1[2], b2[2]), min(b1[3], b2[3])
    inter = max(0, ix2 - ix1) * max(0, iy2 - iy1)
    a1 = (b1[2] - b1[0]) * (b1[3] - b1[1])
    a2 = (b2[2] - b2[0]) * (b2[3] - b2[1])
    union = a1 + a2 - inter
    return inter / (union + 1e-6)


# ---------------------------------------------------------------------------
# Train / val loops
# ---------------------------------------------------------------------------

def train_one_epoch(
    model, loader, criterion, optimizer, scaler, device, epoch, grad_accum=1, max_norm=0.1
):
    model.train()
    total_loss = 0.0
    total_cls_loss = 0.0
    total_bbox_loss = 0.0
    total_giou_loss = 0.0
    n_batches = len(loader)
    optimizer.zero_grad()

    pbar = tqdm(loader, desc=f"Train E{epoch}", unit="batch", dynamic_ncols=True)
    for i, (images, masks, targets_list) in enumerate(pbar):
        images = images.to(device)
        masks = masks.to(device)

        with autocast(device_type="cuda", dtype=torch.bfloat16, enabled=device.type == "cuda"):
            outputs = model(images, masks)
            # Skip batch if model outputs contain NaN/Inf (AMP overflow)
            if (torch.isnan(outputs["pred_logits"]).any() or
                    torch.isnan(outputs["pred_boxes"]).any()):
                print(f"\n[warn] NaN in model output at batch {i}, skipping")
                optimizer.zero_grad()
                continue
            loss_dict = criterion(outputs, targets_list)
            # Scale loss so gradients are averaged across accumulation steps
            loss = loss_dict["loss"] / grad_accum

        scaler.scale(loss).backward()

        if (i + 1) % grad_accum == 0 or (i + 1) == n_batches:
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=max_norm)
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad()

        total_loss += loss_dict["loss"].item()
        total_cls_loss += loss_dict["cls_loss"].item()
        total_bbox_loss += loss_dict["bbox_loss"].item()
        total_giou_loss += loss_dict["giou_loss"].item()
        avg_loss = total_loss / (i + 1)
        pbar.set_postfix(
            loss=f"{avg_loss:.4f}",
            cls=f"{loss_dict['cls_loss'].item():.4f}",
            bbox=f"{loss_dict['bbox_loss'].item():.4f}",
            giou=f"{loss_dict['giou_loss'].item():.4f}",
        )

    return {
        "loss": total_loss / n_batches,
        "cls_loss": total_cls_loss / n_batches,
        "bbox_loss": total_bbox_loss / n_batches,
        "giou_loss": total_giou_loss / n_batches,
    }


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

TIME_BUDGET_SECONDS = float("inf")


def main(args):
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(line_buffering=True)

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    train_loader, val_loader, _ = build_dataloaders(
        data_dir=args.data_dir,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        max_side=args.max_side,
        min_side=args.min_side,
    )
    print(f"Train samples: {len(train_loader.dataset)}")
    print(f"Val samples:   {len(val_loader.dataset)}")

    model = build_model(
        num_classes=args.num_classes,
        num_queries=args.num_queries,
        d_model=args.d_model,
        nhead=args.nhead,
        num_encoder_layers=args.num_encoder_layers,
        num_decoder_layers=args.num_decoder_layers,
        dim_feedforward=args.dim_feedforward,
        dropout=args.dropout,
        pretrained_backbone=True,
    )
    model = model.to(device)

    criterion = DETRLoss(
        num_classes=args.num_classes,
        no_object_weight=args.no_object_weight,
        lambda_cls=args.lambda_cls,
        lambda_bbox=args.lambda_bbox,
        lambda_giou=args.lambda_giou,
    )

    # Separate LRs: lower for pretrained backbone, higher for transformer
    backbone_params = list(model.backbone.parameters())
    other_params = [
        p for n, p in model.named_parameters()
        if not n.startswith("backbone.")
    ]
    optimizer = AdamW(
        [
            {"params": backbone_params, "lr": args.backbone_lr},
            {"params": other_params, "lr": args.lr},
        ],
        weight_decay=args.weight_decay,
    )
    scheduler = CosineAnnealingLR(optimizer, T_max=args.epochs, eta_min=1e-7)
    scaler = GradScaler(enabled=device.type == "cuda")

    os.makedirs(args.save_dir, exist_ok=True)
    os.makedirs(args.log_dir, exist_ok=True)

    # Optionally resume from a checkpoint (model weights only)
    if args.resume:
        ckpt = torch.load(args.resume, map_location=device)
        model.load_state_dict(ckpt["model_state_dict"])
        print(f"Resumed from {args.resume} (epoch={ckpt.get('epoch')}, mAP={ckpt.get('val_map', 0):.4f})")

    best_map = 0.0
    best_ckpt_path = None
    run_ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_path = os.path.join(args.log_dir, f"train_{run_ts}.json")
    training_log = {
        "config": vars(args),
        "start_time": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "device": str(device),
        "epochs": [],
    }

    wall_start = time.time()
    training_seconds = 0.0

    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()

    for epoch in range(1, args.epochs + 1):
        t0 = time.time()
        print(f"\n=== Epoch {epoch}/{args.epochs} ===")

        train_metrics = train_one_epoch(
            model, train_loader, criterion, optimizer, scaler, device, epoch,
            grad_accum=args.grad_accum,
            max_norm=args.max_norm,
        )
        train_loss = train_metrics["loss"]
        t_val = time.time()
        val_map = evaluate_map(model, val_loader, device, score_thr=args.val_score_thr)
        val_time = time.time() - t_val

        scheduler.step()

        epoch_time = time.time() - t0
        training_seconds += epoch_time
        lr_trans = optimizer.param_groups[1]["lr"]
        lr_bb = optimizer.param_groups[0]["lr"]

        print(
            f"Epoch [{epoch}/{args.epochs}] "
            f"LR: {lr_trans:.2e} (trans) / {lr_bb:.2e} (backbone) | "
            f"Train Loss: {train_loss:.4f} | Val mAP: {val_map:.4f} | "
            f"Time: {epoch_time:.1f}s (val: {val_time:.1f}s)"
        )

        is_best = val_map > best_map
        if is_best:
            best_map = val_map
            best_ckpt_path = os.path.join(
                args.save_dir, f"best_model_{run_ts}.pth"
            )
            torch.save(
                {
                    "epoch": epoch,
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "val_map": val_map,
                },
                best_ckpt_path,
            )
            print(f"  -> Best model saved (mAP={val_map:.4f})")

        training_log["epochs"].append(
            {
                "epoch": epoch,
                "train_loss": round(train_loss, 6),
                "train_cls_loss": round(train_metrics["cls_loss"], 6),
                "train_bbox_loss": round(train_metrics["bbox_loss"], 6),
                "train_giou_loss": round(train_metrics["giou_loss"], 6),
                "val_map": round(val_map, 6),
                "elapsed_sec": round(epoch_time, 2),
                "val_time_sec": round(val_time, 2),
                "is_best": is_best,
            }
        )
        with open(log_path, "w") as f:
            json.dump(training_log, f, indent=2)

        if training_seconds >= TIME_BUDGET_SECONDS:
            print(f"Time budget reached after epoch {epoch}. Stopping.")
            break

    total_sec = time.time() - wall_start
    peak_vram = 0.0
    if torch.cuda.is_available():
        peak_vram = torch.cuda.max_memory_allocated() / 1024 / 1024
    num_params_m = sum(p.numel() for p in model.parameters()) / 1e6

    training_log["end_time"] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    training_log["best_val_map"] = round(best_map, 6)
    training_log["training_seconds"] = round(training_seconds, 1)
    training_log["total_seconds"] = round(total_sec, 1)
    training_log["peak_vram_mb"] = round(peak_vram, 1)
    training_log["num_params_M"] = round(num_params_m, 1)
    training_log["best_checkpoint"] = best_ckpt_path
    with open(log_path, "w") as f:
        json.dump(training_log, f, indent=2)

    print(f"\nTraining complete. Best Val mAP: {best_map:.4f}")
    print(f"Training log: {log_path}")
    if best_ckpt_path:
        print(f"Best checkpoint: {best_ckpt_path}")
    print(f"peak_vram_mb:  {peak_vram:.1f}")
    print(f"num_params_M:  {num_params_m:.1f}")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def load_config(path):
    with open(path) as f:
        return yaml.safe_load(f)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train DETR for digit detection")
    parser.add_argument("--config", default="configs/default.yaml")
    # Data
    parser.add_argument("--data_dir", default=None)
    parser.add_argument("--save_dir", default=None)
    parser.add_argument("--log_dir", default=None)
    # Training
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--batch_size", type=int, default=None)
    parser.add_argument("--lr", type=float, default=None)
    parser.add_argument("--backbone_lr", type=float, default=None)
    parser.add_argument("--weight_decay", type=float, default=None)
    parser.add_argument("--dropout", type=float, default=None)
    parser.add_argument("--num_workers", type=int, default=None)
    parser.add_argument("--max_side", type=int, default=None)
    parser.add_argument("--min_side", type=int, default=None)
    # Model
    parser.add_argument("--num_classes", type=int, default=None)
    parser.add_argument("--num_queries", type=int, default=None)
    parser.add_argument("--d_model", type=int, default=None)
    parser.add_argument("--nhead", type=int, default=None)
    parser.add_argument("--num_encoder_layers", type=int, default=None)
    parser.add_argument("--num_decoder_layers", type=int, default=None)
    parser.add_argument("--dim_feedforward", type=int, default=None)
    # Loss
    parser.add_argument("--no_object_weight", type=float, default=None)
    parser.add_argument("--lambda_cls", type=float, default=None)
    parser.add_argument("--lambda_bbox", type=float, default=None)
    parser.add_argument("--lambda_giou", type=float, default=None)
    parser.add_argument("--grad_accum", type=int, default=None)
    parser.add_argument("--max_norm", type=float, default=None)
    parser.add_argument("--device", type=str, default=None)
    parser.add_argument("--val_score_thr", type=float, default=None)
    parser.add_argument("--resume", type=str, default=None,
                        help="Path to checkpoint to resume model weights from")

    args = parser.parse_args()
    cfg = load_config(args.config)

    # Merge: CLI overrides config; config overrides hardcoded defaults
    for key, val in vars(args).items():
        if key == "config":
            continue
        if val is None and key in cfg:
            setattr(args, key, cfg[key])

    # Hardcoded fallbacks
    defaults = dict(
        data_dir="nycu-hw2-data",
        save_dir="checkpoints",
        log_dir="log",
        epochs=50,
        batch_size=4,
        lr=1e-4,
        backbone_lr=1e-5,
        weight_decay=1e-4,
        dropout=0.1,
        num_workers=4,
        max_side=640,
        min_side=480,
        num_classes=10,
        num_queries=100,
        d_model=256,
        nhead=8,
        num_encoder_layers=6,
        num_decoder_layers=6,
        dim_feedforward=2048,
        no_object_weight=0.1,
        lambda_cls=1.0,
        lambda_bbox=5.0,
        lambda_giou=2.0,
        grad_accum=1,
        max_norm=0.1,
        device="cuda",
        val_score_thr=0.01,
    )
    for key, val in defaults.items():
        if getattr(args, key, None) is None:
            setattr(args, key, val)

    main(args)
