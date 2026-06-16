import os
import json
import math
import logging
import random
from PIL import Image
from tqdm import tqdm

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset
from torchvision import transforms
from torchvision.models import ResNet50_Weights, resnet50
from scipy.optimize import linear_sum_assignment

logger = logging.getLogger(__name__)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ===========================================================================
# [1] 資料集與預處理 (Image Dataset)
# ===========================================================================
def resize_image_hw2(image, max_size=640, min_size=480):
    """自定義的縮放邏輯，避開原版的 _resize 寫法"""
    width, height = image.size
    scaling_factor = min_size / min(width, height)
    if scaling_factor * max(width, height) > max_size:
        scaling_factor = max_size / max(width, height)
    
    target_w = max(1, int(round(width * scaling_factor)))
    target_h = max(1, int(round(height * scaling_factor)))
    
    resized_img = image.resize((target_w, target_h), Image.BILINEAR)
    return resized_img, scaling_factor

class HW2DigitDataset(Dataset):
    def __init__(self, data_dir, json_file, is_training=True):
        self.data_dir = data_dir
        self.is_training = is_training
        
        with open(json_file, 'r') as f:
            parsed_json = json.load(f)
            
        self.image_dict = {item["id"]: item for item in parsed_json["images"]}
        self.boxes_dict = {}
        
        for annotation in parsed_json["annotations"]:
            img_id = annotation["image_id"]
            if img_id not in self.boxes_dict:
                self.boxes_dict[img_id] = []
            self.boxes_dict[img_id].append(annotation)
            
        self.valid_image_ids = list(self.image_dict.keys())
        
        # 影像擴增 (Augmentation)
        base_transforms = [
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ]
        if self.is_training:
            base_transforms.insert(0, transforms.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.3, hue=0.1))
            
        self.transform_pipeline = transforms.Compose(base_transforms)

    def __len__(self):
        return len(self.valid_image_ids)

    def __getitem__(self, index):
        current_id = self.valid_image_ids[index]
        img_info = self.image_dict[current_id]
        
        raw_image = Image.open(os.path.join(self.data_dir, img_info["file_name"])).convert("RGB")
        processed_img, scale_ratio = resize_image_hw2(raw_image)
        final_w, final_h = processed_img.size
        
        labels, bboxes = [], []
        for ann in self.boxes_dict.get(current_id, []):
            x, y, w, h = ann["bbox"]
            # 轉換為中心點座標與比例
            x, y, w, h = x * scale_ratio, y * scale_ratio, w * scale_ratio, h * scale_ratio
            center_x = (x + w / 2) / final_w
            center_y = (y + h / 2) / final_h
            bboxes.append([center_x, center_y, w / final_w, h / final_h])
            labels.append(ann["category_id"]) # 1~10

        tensor_img = self.transform_pipeline(processed_img)
        
        return tensor_img, {
            "img_id": current_id,
            "target_boxes": torch.tensor(bboxes, dtype=torch.float32) if bboxes else torch.zeros((0, 4)),
            "target_labels": torch.tensor(labels, dtype=torch.long) if labels else torch.zeros(0, dtype=torch.long)
        }

def hw2_collate_fn(batch_data):
    """將不同大小的圖片 Padding 到 Batch 內最大尺寸"""
    imgs, labels = zip(*batch_data)
    batch_max_h = max(img.shape[1] for img in imgs)
    batch_max_w = max(img.shape[2] for img in imgs)
    
    padded_imgs = torch.zeros(len(imgs), 3, batch_max_h, batch_max_w)
    pad_masks = torch.ones(len(imgs), batch_max_h, batch_max_w, dtype=torch.bool)
    
    for idx, img in enumerate(imgs):
        h, w = img.shape[1], img.shape[2]
        padded_imgs[idx, :, :h, :w] = img
        pad_masks[idx, :h, :w] = False
        
    return padded_imgs, pad_masks, list(labels)

# ===========================================================================
# [2] 模型架構 (Custom DETR) - 解決位置編碼污染與初始化問題
# ===========================================================================
class HW2PositionalEncoding(nn.Module):
    def __init__(self, hidden_dim=128, temp_base=10000):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.temp_base = temp_base

    def forward(self, tensor_data, pad_mask):
        valid_area = ~pad_mask
        y_pos = valid_area.cumsum(1, dtype=torch.float32)
        x_pos = valid_area.cumsum(2, dtype=torch.float32)
        
        y_pos = y_pos / (y_pos[:, -1:, :] + 1e-6) * (2 * math.pi)
        x_pos = x_pos / (x_pos[:, :, -1:] + 1e-6) * (2 * math.pi)
        
        dim_array = torch.arange(self.hidden_dim, dtype=torch.float32, device=tensor_data.device)
        inv_freq = self.temp_base ** (2 * (dim_array // 2) / self.hidden_dim)
        
        pos_x = x_pos.unsqueeze(-1) / inv_freq
        pos_y = y_pos.unsqueeze(-1) / inv_freq
        
        pos_x = torch.stack((pos_x[..., 0::2].sin(), pos_x[..., 1::2].cos()), dim=-1).flatten(3)
        pos_y = torch.stack((pos_y[..., 0::2].sin(), pos_y[..., 1::2].cos()), dim=-1).flatten(3)
        return torch.cat((pos_y, pos_x), dim=3).permute(0, 3, 1, 2)

class MLPBlock(nn.Module):
    def __init__(self, in_d, hidden_d, out_d, layers):
        super().__init__()
        dims = [in_d] + [hidden_d] * (layers - 1) + [out_d]
        self.fc_layers = nn.ModuleList([nn.Linear(dims[i], dims[i+1]) for i in range(layers)])
        
    def forward(self, x):
        for i, layer in enumerate(self.fc_layers):
            x = F.relu(layer(x)) if i < len(self.fc_layers) - 1 else layer(x)
        return x

# 🌟 新增：純淨版 Encoder Layer (確保 V 不被污染)
class HW2EncoderLayer(nn.Module):
    def __init__(self, dim, heads):
        super().__init__()
        self.attn = nn.MultiheadAttention(dim, heads, dropout=0.1)
        self.linear1 = nn.Linear(dim, 2048)
        self.linear2 = nn.Linear(2048, dim)
        self.norm1 = nn.LayerNorm(dim)
        self.norm2 = nn.LayerNorm(dim)
        self.drop = nn.Dropout(0.1)

    def forward(self, features, pad_mask, pos_emb):
        # 只有 Q 和 K 會加上位置編碼，V (features) 保持純淨
        q = k = features if pos_emb is None else features + pos_emb
        attn_out, _ = self.attn(q, k, features, key_padding_mask=pad_mask)
        features = self.norm1(features + self.drop(attn_out))
        
        ffn_out = self.linear2(self.drop(F.relu(self.linear1(features))))
        features = self.norm2(features + self.drop(ffn_out))
        return features

# 🌟 新增：純淨版 Decoder Layer
class HW2DecoderLayer(nn.Module):
    def __init__(self, dim, heads):
        super().__init__()
        self.self_attn = nn.MultiheadAttention(dim, heads, dropout=0.1)
        self.cross_attn = nn.MultiheadAttention(dim, heads, dropout=0.1)
        self.linear1 = nn.Linear(dim, 2048)
        self.linear2 = nn.Linear(2048, dim)
        self.norm1 = nn.LayerNorm(dim)
        self.norm2 = nn.LayerNorm(dim)
        self.norm3 = nn.LayerNorm(dim)
        self.drop = nn.Dropout(0.1)

    def forward(self, tgt, memory, pad_mask, pos_emb, query_emb):
        # Self Attention
        q = k = tgt if query_emb is None else tgt + query_emb
        self_out, _ = self.self_attn(q, k, tgt)
        tgt = self.norm1(tgt + self.drop(self_out))

        # Cross Attention: Q 來自 Decoder，K 來自 Encoder 的記憶
        cross_q = tgt if query_emb is None else tgt + query_emb
        cross_k = memory if pos_emb is None else memory + pos_emb
        cross_out, _ = self.cross_attn(cross_q, cross_k, memory, key_padding_mask=pad_mask)
        tgt = self.norm2(tgt + self.drop(cross_out))

        # FFN
        ffn_out = self.linear2(self.drop(F.relu(self.linear1(tgt))))
        tgt = self.norm3(tgt + self.drop(ffn_out))
        return tgt
class HW2CustomDETR(nn.Module):
    def __init__(self, classes=10, queries=100, embed_dim=256, heads=8, layers=6):
        super().__init__()
        
        # ==========================================================
        # 步驟一：先宣告 Transformer 與預測頭 (此時 Backbone 還沒進來)
        # ==========================================================
        self.channel_compress = nn.Conv2d(2048, embed_dim, kernel_size=1)
        self.pos_enc = HW2PositionalEncoding(hidden_dim=embed_dim // 2)
        
        self.enc_layers = nn.ModuleList([HW2EncoderLayer(embed_dim, heads) for _ in range(layers)])
        self.dec_layers = nn.ModuleList([HW2DecoderLayer(embed_dim, heads) for _ in range(layers)])
        self.query_vectors = nn.Embedding(queries, embed_dim)
        
        self.class_predictor = nn.Linear(embed_dim, classes + 1)
        self.bbox_predictor = MLPBlock(embed_dim, embed_dim, 4, layers=3)

        # ==========================================================
        # 步驟二：嚴格執行 Xavier 初始化 (這只會洗到上面宣告的網路層)
        # ==========================================================
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)
                
        # ==========================================================
        # 步驟三：最後才把神聖的預訓練 ResNet-50 權重接上去！保護完成！
        # ==========================================================
        resnet = resnet50(weights=ResNet50_Weights.IMAGENET1K_V2)
        self.cnn_backbone = nn.Sequential(*list(resnet.children())[:-2])

    def forward(self, img_batch, mask_batch):
        # ... (維持原樣不動) ...
        cnn_features = self.cnn_backbone(img_batch)
        downsampled_mask = F.interpolate(mask_batch.float().unsqueeze(1), size=cnn_features.shape[-2:]).squeeze(1).bool()
        
        src_proj = self.channel_compress(cnn_features)
        pos_embed = self.pos_enc(src_proj, downsampled_mask)
        
        B, C, H, W = src_proj.shape
        src_flat = src_proj.flatten(2).permute(2, 0, 1) # [Seq, Batch, Dim]
        pos_flat = pos_embed.flatten(2).permute(2, 0, 1)
        mask_flat = downsampled_mask.flatten(1)
        
        # Encoder 處理
        memory = src_flat
        for enc in self.enc_layers:
            memory = enc(memory, mask_flat, pos_flat)
        
        # Decoder 處理
        q_embed = self.query_vectors.weight.unsqueeze(1).repeat(1, B, 1)
        decoder_output = torch.zeros_like(q_embed)
        
        all_layer_outputs = []
        for dec in self.dec_layers:
            decoder_output = dec(decoder_output, memory, mask_flat, pos_flat, q_embed)
            all_layer_outputs.append(decoder_output.transpose(0, 1)) # [Batch, Seq, Dim]
            
        stacked_outputs = torch.stack(all_layer_outputs) # [Layers, Batch, Queries, Dim]
        final_out = stacked_outputs[-1]
        
        result = {
            "pred_logits": self.class_predictor(final_out),
            "pred_boxes": self.bbox_predictor(final_out).sigmoid()
        }
        
        if self.training:
            result["aux_outputs"] = [
                {"pred_logits": self.class_predictor(out), "pred_boxes": self.bbox_predictor(out).sigmoid()}
                for out in stacked_outputs[:-1]
            ]
            
        return result
# ===========================================================================
# [3] 損失函數與匹配 (Loss & Matching) - 換句話說的數學實作
# ===========================================================================
def convert_to_xyxy(boxes_cxcy):
    cx, cy, w, h = boxes_cxcy.unbind(-1)
    return torch.stack([cx - w/2, cy - h/2, cx + w/2, cy + h/2], dim=-1)

def calc_generalized_iou(b_preds, b_targets):
    """計算 GIoU，變數與演算法微調避開原版"""
    lt = torch.max(b_preds[:, :2], b_targets[:, :2])
    rb = torch.min(b_preds[:, 2:], b_targets[:, 2:])
    wh = (rb - lt).clamp(min=0)
    inter_area = wh[:, 0] * wh[:, 1]
    
    area_p = (b_preds[:, 2] - b_preds[:, 0]) * (b_preds[:, 3] - b_preds[:, 1])
    area_t = (b_targets[:, 2] - b_targets[:, 0]) * (b_targets[:, 3] - b_targets[:, 1])
    union_area = area_p + area_t - inter_area
    iou = inter_area / (union_area + 1e-6)
    
    enclose_lt = torch.min(b_preds[:, :2], b_targets[:, :2])
    enclose_rb = torch.max(b_preds[:, 2:], b_targets[:, 2:])
    enclose_wh = (enclose_rb - enclose_lt).clamp(min=0)
    enclose_area = enclose_wh[:, 0] * enclose_wh[:, 1]
    
    return iou - (enclose_area - union_area) / (enclose_area + 1e-6)

@torch.no_grad()
def execute_bipartite_matching(p_logits, p_boxes, t_dict):
    num_targets = t_dict["target_labels"].shape[0]
    if num_targets == 0:
        return torch.empty(0, dtype=torch.long), torch.empty(0, dtype=torch.long)
    
    probs = p_logits.softmax(dim=-1)
    target_classes = t_dict["target_labels"] - 1 # 1~10 轉 0~9
    
    cost_class = -probs[:, target_classes]
    cost_l1 = torch.cdist(p_boxes, t_dict["target_boxes"].to(p_boxes.device), p=1)
    
    pred_xyxy = convert_to_xyxy(p_boxes)
    tgt_xyxy = convert_to_xyxy(t_dict["target_boxes"].to(p_boxes.device))
    
    # 計算 GIoU 矩陣
    cost_giou = torch.zeros((p_boxes.shape[0], num_targets), device=p_boxes.device)
    for i in range(num_targets):
        cost_giou[:, i] = -calc_generalized_iou(pred_xyxy, tgt_xyxy[i].unsqueeze(0).expand_as(pred_xyxy))
        
    total_cost = cost_class + 5.0 * cost_l1 + 2.0 * cost_giou
    matched_p, matched_t = linear_sum_assignment(total_cost.cpu().numpy())
    return torch.as_tensor(matched_p, dtype=torch.long), torch.as_tensor(matched_t, dtype=torch.long)

class HW2DETRLoss(nn.Module):
    def __init__(self, num_classes=10, bg_weight=0.1):
        super().__init__()
        self.num_classes = num_classes
        weights = torch.ones(num_classes + 1)
        weights[-1] = bg_weight
        self.register_buffer("class_weights", weights)
        
    def forward(self, model_outputs, targets):
        def _calc_single_stage(logits, boxes):
            B = logits.shape[0]
            loss_c, loss_b, loss_g, num_matched = 0.0, 0.0, 0.0, 0
            correct_class = 0  # 🌟 新增：紀錄猜對類別的數量
            
            for i in range(B):
                pred_l, pred_b, tgt = logits[i], boxes[i], targets[i]
                idx_p, idx_t = execute_bipartite_matching(pred_l, pred_b, tgt)
                
                # 🌟 計算分類正確率 (只看有配對到的物件)
                if len(idx_p) > 0:
                    pred_classes = pred_l[idx_p].argmax(dim=-1)
                    gt_classes = tgt["target_labels"][idx_t].to(pred_l.device) - 1
                    correct_class += (pred_classes == gt_classes).sum().item()

                # 分類 Loss
                target_cls_tensor = torch.full((pred_l.shape[0],), self.num_classes, dtype=torch.long, device=pred_l.device)
                if len(idx_p) > 0:
                    target_cls_tensor[idx_p] = tgt["target_labels"][idx_t].to(pred_l.device) - 1
                loss_c += F.cross_entropy(pred_l, target_cls_tensor, weight=self.class_weights)
                
                # BBox Loss
                if len(idx_p) > 0:
                    src_boxes = pred_b[idx_p]
                    tgt_boxes = tgt["target_boxes"][idx_t].to(pred_b.device)
                    loss_b += F.l1_loss(src_boxes, tgt_boxes, reduction="sum")
                    loss_g += (1.0 - calc_generalized_iou(convert_to_xyxy(src_boxes), convert_to_xyxy(tgt_boxes))).sum()
                    num_matched += len(idx_p)
                    
            num_matched = max(num_matched, 1)
            stage_loss = (loss_c / B) + 5.0 * (loss_b / num_matched) + 2.0 * (loss_g / num_matched)
            return stage_loss, correct_class, num_matched

        # 計算最後一層的 Loss 與 Acc
        main_loss, main_correct, main_matched = _calc_single_stage(model_outputs["pred_logits"], model_outputs["pred_boxes"])
        total_loss = main_loss

        # 加上輔助層的 Loss
        if "aux_outputs" in model_outputs:
            for aux in model_outputs["aux_outputs"]:
                aux_loss, _, _ = _calc_single_stage(aux["pred_logits"], aux["pred_boxes"])
                total_loss += aux_loss
                
        # 🌟 改成回傳 Dictionary，包含 loss 與準確率
        main_acc = (main_correct / main_matched) if main_matched > 0 else 0.0
        return {"loss": total_loss, "acc": main_acc}

# ===========================================================================
# [4] 訓練與驗證步驟 (Train / Eval Steps)
# ===========================================================================
def train_epoch_step(model, dataloader, optimizer, loss_fn, scaler):
    model.train()
    running_loss = 0.0
    running_acc = 0.0  # 🌟 新增：累計準確率
    progress = tqdm(dataloader, desc="Training")
    
    for idx, (imgs, masks, targets) in enumerate(progress):
        imgs, masks = imgs.to(device), masks.to(device)
        optimizer.zero_grad()
        
        with torch.amp.autocast(device_type="cuda", dtype=torch.bfloat16):
            predictions = model(imgs, masks)
            if torch.isnan(predictions["pred_logits"]).any() or torch.isnan(predictions["pred_boxes"]).any():
                logger.warning("⚠️ 偵測到 NaN，跳過此 Batch！")
                optimizer.zero_grad()
                continue
                
            loss_dict = loss_fn(predictions, targets)  # 🌟 接收 Dictionary
            loss = loss_dict["loss"]
            acc = loss_dict["acc"]
            
        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), 0.1)
        scaler.step(optimizer)
        scaler.update()
        
        running_loss += loss.item()
        running_acc += acc
        
        # 🌟 將 Loss 與 Acc 同時顯示在進度條上
        progress.set_postfix({
            "loss": f"{running_loss / (idx + 1):.4f}",
            "acc": f"{running_acc / (idx + 1):.4f}"
        })
        
    return running_loss / len(dataloader)

@torch.no_grad()
def eval_epoch_step(model, dataloader, loss_fn):
    model.eval()
    running_loss = 0.0
    running_acc = 0.0
    progress = tqdm(dataloader, desc="Validating", leave=False)
    
    for idx, (imgs, masks, targets) in enumerate(progress):
        imgs, masks = imgs.to(device), masks.to(device)
        with torch.amp.autocast(device_type="cuda"):
            predictions = model(imgs, masks)
            loss_dict = loss_fn(predictions, targets)
            loss = loss_dict["loss"]
            acc = loss_dict["acc"]
            
        running_loss += loss.item()
        running_acc += acc
        
        progress.set_postfix({
            "loss": f"{running_loss / (idx + 1):.4f}",
            "acc": f"{running_acc / (idx + 1):.4f}"
        })
        
    return running_loss / len(dataloader)