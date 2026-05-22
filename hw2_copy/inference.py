"""Inference script for DETR digit detection (HW2).

Generates pred.json in COCO detection format for submission.
"""

import argparse
import json
import os
import zipfile

import torch
import yaml

from dataset import TestDataset, collate_fn, get_val_transforms, build_dataloaders
from model import build_model
from torch.utils.data import DataLoader
from train import box_cxcywh_to_xyxy, evaluate_map


def run_inference(model, test_loader, device, score_threshold=0.5):
    """Run model on test set and return COCO-format predictions."""
    model.eval()
    results = []

    with torch.no_grad():
        for images, masks, targets_list in test_loader:
            images = images.to(device)
            masks = masks.to(device)

            outputs = model(images, masks)
            pred_logits = outputs["pred_logits"]  # [B, N, C+1]
            pred_boxes = outputs["pred_boxes"]    # [B, N, 4]

            for b, tgt in enumerate(targets_list):
                image_id = int(tgt["image_id"])
                orig_h, orig_w = tgt["orig_size"].tolist()

                logits = pred_logits[b]   # [N, C+1]
                boxes = pred_boxes[b]     # [N, 4] cxcywh normalised

                probs = logits.softmax(-1)
                # Exclude "no-object" class (last index)
                scores, classes = probs[:, :-1].max(dim=-1)

                for n in range(logits.shape[0]):
                    score = scores[n].item()
                    if score < score_threshold:
                        continue

                    category_id = int(classes[n].item()) + 1  # 1-indexed

                    cx, cy, w, h = boxes[n].tolist()
                    # Convert to pixel-space [x_min, y_min, w, h]
                    x_min = (cx - w / 2) * orig_w
                    y_min = (cy - h / 2) * orig_h
                    bbox_w = w * orig_w
                    bbox_h = h * orig_h

                    results.append(
                        {
                            "image_id": image_id,
                            "bbox": [x_min, y_min, bbox_w, bbox_h],
                            "score": score,
                            "category_id": category_id,
                        }
                    )

    return results


def main(args):
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Build model and load checkpoint
    model = build_model(
        num_classes=args.num_classes,
        num_queries=args.num_queries,
        d_model=args.d_model,
        nhead=args.nhead,
        num_encoder_layers=args.num_encoder_layers,
        num_decoder_layers=args.num_decoder_layers,
        dim_feedforward=args.dim_feedforward,
        dropout=0.0,  # no dropout at inference
        pretrained_backbone=False,
    )

    ckpt = torch.load(args.checkpoint, map_location=device)
    state = ckpt.get("model_state_dict", ckpt)
    model.load_state_dict(state)
    model = model.to(device)
    print(f"Loaded checkpoint: {args.checkpoint}")

    test_dataset = TestDataset(
        img_dir=os.path.join(args.data_dir, "test"),
        transforms=get_val_transforms(),
        max_side=args.max_side,
        min_side=args.min_side,
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        collate_fn=collate_fn,
        pin_memory=True,
    )
    print(f"Test samples: {len(test_dataset)}")

    if args.run_val:
        _, val_loader, _ = build_dataloaders(
            data_dir=args.data_dir,
            batch_size=args.batch_size,
            num_workers=args.num_workers,
            max_side=args.max_side,
            min_side=args.min_side,
        )
        val_map = evaluate_map(model, val_loader, device)
        print(f"Val mAP: {val_map:.4f}")

    results = run_inference(model, test_loader, device, args.score_threshold)
    print(f"Total predictions: {len(results)}")

    # Save pred.json
    os.makedirs(args.output_dir, exist_ok=True)
    pred_json_path = os.path.join(args.output_dir, "pred.json")
    with open(pred_json_path, "w") as f:
        json.dump(results, f, indent=4)
    print(f"Saved: {pred_json_path}")

    # Package for submission
    zip_path = os.path.join(args.output_dir, "submission.zip")
    with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_DEFLATED) as zf:
        zf.write(pred_json_path, "pred.json")
    print(f"Saved: {zip_path}")


def load_config(path):
    with open(path) as f:
        return yaml.safe_load(f)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="DETR inference for digit detection")
    parser.add_argument("--config", default="configs/default.yaml")
    parser.add_argument("--checkpoint", required=True, help="Path to .pth checkpoint")
    parser.add_argument("--data_dir", default=None)
    parser.add_argument("--output_dir", default="predictions")
    parser.add_argument("--batch_size", type=int, default=None)
    parser.add_argument("--num_workers", type=int, default=None)
    parser.add_argument("--max_side", type=int, default=None)
    parser.add_argument("--min_side", type=int, default=None)
    parser.add_argument("--score_threshold", type=float, default=None)
    # Model architecture (must match training)
    parser.add_argument("--num_classes", type=int, default=None)
    parser.add_argument("--num_queries", type=int, default=None)
    parser.add_argument("--d_model", type=int, default=None)
    parser.add_argument("--nhead", type=int, default=None)
    parser.add_argument("--num_encoder_layers", type=int, default=None)
    parser.add_argument("--num_decoder_layers", type=int, default=None)
    parser.add_argument("--dim_feedforward", type=int, default=None)
    parser.add_argument("--dropout", type=float, default=None)
    parser.add_argument("--device", type=str, default=None)
    parser.add_argument("--run_val", action="store_true", help="Run validation mAP before inference")

    args = parser.parse_args()
    cfg = load_config(args.config)

    for key, val in vars(args).items():
        if key == "config":
            continue
        if val is None and key in cfg:
            setattr(args, key, cfg[key])

    defaults = dict(
        data_dir="nycu-hw2-data",
        batch_size=4,
        num_workers=4,
        max_side=640,
        min_side=480,
        score_threshold=0.5,
        num_classes=10,
        num_queries=100,
        d_model=256,
        nhead=8,
        num_encoder_layers=6,
        num_decoder_layers=6,
        dim_feedforward=2048,
        dropout=0.0,
        device="cuda",
    )
    for key, val in defaults.items():
        if getattr(args, key, None) is None:
            setattr(args, key, val)

    main(args)
