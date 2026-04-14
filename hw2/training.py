import os
import json
import logging
from datetime import datetime

import torch
from torch.utils.data import Dataset, DataLoader
from PIL import Image
from tqdm import tqdm

from transformers import (
    DetrConfig,
    DetrForObjectDetection,
    AutoImageProcessor
)

# ==========================================
# Logger 設定
# ==========================================
os.makedirs("./logs", exist_ok=True)
current_time = datetime.now().strftime("%Y%m%d_%H%M%S")
logging.basicConfig(
    level=logging.INFO,
    format="[%(levelname)s] %(asctime)s - %(message)s",
    handlers=[
        logging.FileHandler(f"./logs/train_{current_time}.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# ==========================================
# 1. Dataset 準備 (處理 BBox 與類別轉換)
# ==========================================
class DigitDetectionDataset(Dataset):
    def __init__(self, img_dir, annotation_file, processor):
        self.img_dir = img_dir
        self.processor = processor

        with open(annotation_file, 'r') as f:
            coco_data = json.load(f)

        self.images = {img['id']: img for img in coco_data['images']}
        self.annotations = coco_data['annotations']

        # 將標籤依據 image_id 進行分組
        self.img_to_anns = {}
        for ann in self.annotations:
            img_id = ann['image_id']
            if img_id not in self.img_to_anns:
                self.img_to_anns[img_id] = []
            self.img_to_anns[img_id].append(ann)

        self.image_ids = list(self.images.keys())

    def __len__(self):
        return len(self.image_ids)

    def __getitem__(self, idx):
        img_id = self.image_ids[idx]
        img_info = self.images[img_id]
        img_path = os.path.join(self.img_dir, img_info['file_name'])

        image = Image.open(img_path).convert("RGB")
        img_w, img_h = image.size

        anns = self.img_to_anns.get(img_id, [])
        boxes = []
        labels = []

        for ann in anns:
            # COCO 格式: [x_min, y_min, w, h]
            x_min, y_min, w, h = ann['bbox']

            # DETR 模型需要的是 [中心點 x, 中心點 y, 寬, 高] 且介於 0~1 之間
            cx = (x_min + (w / 2)) / img_w
            cy = (y_min + (h / 2)) / img_h
            norm_w = w / img_w
            norm_h = h / img_h

            boxes.append([cx, cy, norm_w, norm_h])

            # 模型內部預設類別從 0 開始，但作業標籤從 1 開始，這裡先減 1
            labels.append(ann['category_id'] - 1)

        # 處理沒有標籤的背景圖片
        if len(boxes) == 0:
            boxes = torch.empty((0, 4), dtype=torch.float32)
            labels = torch.empty((0,), dtype=torch.int64)
        else:
            boxes = torch.tensor(boxes, dtype=torch.float32)
            labels = torch.tensor(labels, dtype=torch.int64)

        target = {
            "image_id": torch.tensor([img_id]),
            "boxes": boxes,
            "class_labels": labels
        }

        return image, target


def collate_fn(batch, processor):
    """
    DETR 需要把不同大小的圖片 Pad 到相同大小，並產生 pixel_mask
    """
    images = [item[0] for item in batch]
    targets = [item[1] for item in batch]

    # processor 會自動幫我們做 Resize, Normalize 以及 Padding
    encoding = processor(images=images, return_tensors="pt")

    batch_dict = {
        "pixel_values": encoding["pixel_values"],
        "pixel_mask": encoding["pixel_mask"],
        "labels": targets
    }
    return batch_dict


# ==========================================
# 2. 建立模型 (嚴格遵守作業規定)
# ==========================================
def build_model(num_classes):
    """
    僅使用 ResNet-50 預訓練權重，DETR 的 Encoder/Decoder 從頭訓練
    """
    config = DetrConfig(
        backbone="resnet50",
        use_pretrained_backbone=True,  # 允許 Backbone 預訓練
        num_labels=num_classes,
        init_std=0.02,
        init_xavier_std=1.0,
    )
    # 初始化一個權重隨機的 DETR，並掛上預訓練的 ResNet
    model = DetrForObjectDetection(config)
    return model


# ==========================================
# 3. 訓練與驗證邏輯
# ==========================================
def train_one_epoch(model, train_loader, optimizer, epoch):
    model.train()
    total_loss = 0

    pbar = tqdm(train_loader, desc=f"Epoch {epoch} Training")
    for batch in pbar:
        pixel_values = batch["pixel_values"].to(device)
        pixel_mask = batch["pixel_mask"].to(device)
        labels = [{k: v.to(device) for k, v in t.items()}
                  for t in batch["labels"]]

        optimizer.zero_grad()

        # Transformers 的 DETR 只要放入 labels 就會自動計算 Loss
        outputs = model(pixel_values=pixel_values,
                        pixel_mask=pixel_mask,
                        labels=labels)

        loss = outputs.loss
        loss.backward()

        # 梯度裁剪 (預防 Gradient Exploding，這是 DETR 訓練標配)
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=0.1)
        optimizer.step()

        total_loss += loss.item()
        pbar.set_postfix({"Loss": f"{loss.item():.4f}"})

    avg_loss = total_loss / len(train_loader)
    logger.info(f"Epoch [{epoch}] Avg Train Loss: {avg_loss:.4f}")
    return avg_loss


def evaluate_one_epoch(model, valid_loader, epoch):
    """
    新增的驗證邏輯：在驗證集上計算 Loss，不更新權重
    """
    model.eval()
    total_loss = 0

    pbar = tqdm(valid_loader, desc=f"Epoch {epoch} Validating")
    with torch.no_grad():
        for batch in pbar:
            pixel_values = batch["pixel_values"].to(device)
            pixel_mask = batch["pixel_mask"].to(device)
            labels = [{k: v.to(device) for k, v in t.items()}
                      for t in batch["labels"]]

            outputs = model(pixel_values=pixel_values,
                            pixel_mask=pixel_mask,
                            labels=labels)

            loss = outputs.loss
            total_loss += loss.item()
            pbar.set_postfix({"Loss": f"{loss.item():.4f}"})

    avg_loss = total_loss / len(valid_loader)
    logger.info(f"Epoch [{epoch}] Avg Valid Loss: {avg_loss:.4f}")
    return avg_loss


# ==========================================
# 4. 主程式 Main
# ==========================================
def main():
    # ------------------ 超參數與路徑設定 ------------------
    batch_size = 4
    num_epochs = 15
    lr = 1e-4
    lr_backbone = 1e-5  # Backbone 通常需要較小的 LR
    weight_decay = 1e-4
    num_classes = 10    # 數字 0-9 共 10 類

    # 配合新的資料夾結構
    train_dir = "./data/train"
    train_ann = "./data/train.json"
    valid_dir = "./data/valid"
    valid_ann = "./data/valid.json"
    
    save_path = f"./model_weight/detr_best_{current_time}.pth"

    os.makedirs(os.path.dirname(save_path), exist_ok=True)

    # ------------------ 初始化與資料讀取 ------------------
    processor = AutoImageProcessor.from_pretrained("facebook/detr-resnet-50")

    logger.info("正在載入 Training Dataset...")
    train_dataset = DigitDetectionDataset(train_dir, train_ann, processor)
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        collate_fn=lambda x: collate_fn(x, processor),
        num_workers=4
    )

    logger.info("正在載入 Validation Dataset...")
    valid_dataset = DigitDetectionDataset(valid_dir, valid_ann, processor)
    valid_loader = DataLoader(
        valid_dataset,
        batch_size=batch_size,
        shuffle=False,  # 驗證集不需要打亂
        collate_fn=lambda x: collate_fn(x, processor),
        num_workers=4
    )

    # ------------------ 模型與優化器 ------------------
    model = build_model(num_classes).to(device)

    # 針對 Backbone 與 Transformer 設定不同的學習率
    param_dicts = [
        {"params": [p for n, p in model.named_parameters()
                    if "backbone" not in n and p.requires_grad]},
        {"params": [p for n, p in model.named_parameters()
                    if "backbone" in n and p.requires_grad],
         "lr": lr_backbone},
    ]
    optimizer = torch.optim.AdamW(param_dicts, lr=lr, weight_decay=weight_decay)

    # ------------------ 開始訓練 ------------------
    logger.info("========================================")
    logger.info("開始訓練 DETR (ResNet-50 Backbone)")
    logger.info("========================================")

    best_val_loss = float('inf')

    for epoch in range(1, num_epochs + 1):
        # 1. 訓練
        train_loss = train_one_epoch(model, train_loader, optimizer, epoch)
        
        # 2. 驗證
        val_loss = evaluate_one_epoch(model, valid_loader, epoch)

        # 3. 儲存最佳模型
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(), save_path)
            logger.info(f"✨ 發現更低的 Valid Loss ({val_loss:.4f})，已儲存最佳權重至 {save_path}！")
        
        logger.info("-" * 40)

    logger.info("🎉 訓練完成！")


if __name__ == "__main__":
    main()