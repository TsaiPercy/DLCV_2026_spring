import os
import json
import logging
from tqdm import tqdm
from PIL import Image

import torch
from torch.utils.data import Dataset, DataLoader
from transformers import (
    DetrConfig,
    DetrForObjectDetection,
    AutoImageProcessor
)

# ==========================================
# Logger 設定
# ==========================================
logging.basicConfig(
    level=logging.INFO,
    format="[%(levelname)s] %(asctime)s - %(message)s",
)
logger = logging.getLogger(__name__)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# ==========================================
# 1. 測試集 Dataset 準備 (無標籤版本)
# ==========================================
class DigitTestDataset(Dataset):
    def __init__(self, img_dir):
        self.img_dir = img_dir
        # 取得所有圖片檔案，並假設檔名即為 image_id (例如: "123.jpg" -> 123)
        self.image_files = [
            f for f in os.listdir(img_dir)
            if f.lower().endswith(('.png', '.jpg', '.jpeg'))
        ]
        
    def __len__(self):
        return len(self.image_files)

    def __getitem__(self, idx):
        file_name = self.image_files[idx]
        img_path = os.path.join(self.img_dir, file_name)
        
        # 萃取 image_id (移除副檔名)
        image_id = int(os.path.splitext(file_name)[0])
        
        image = Image.open(img_path).convert("RGB")
        original_size = image.size  # (width, height)
        
        return image, image_id, original_size


def test_collate_fn(batch, processor):
    """
    為測試集準備的 collate_fn，不需要處理 labels
    """
    images = [item[0] for item in batch]
    image_ids = [item[1] for item in batch]
    original_sizes = [item[2] for item in batch]

    encoding = processor(images=images, return_tensors="pt")

    return {
        "pixel_values": encoding["pixel_values"],
        "pixel_mask": encoding["pixel_mask"],
        "image_ids": image_ids,
        "original_sizes": original_sizes
    }


# ==========================================
# 2. 建立與載入模型
# ==========================================
def load_trained_model(weight_path, num_classes):
    """
    建立與訓練時一模一樣的架構，並載入訓練好的權重
    """
    logger.info("正在建立 DETR 模型架構...")
    config = DetrConfig(
        backbone="resnet50",
        use_pretrained_backbone=False, # 推論時不需要重新下載 Backbone 預訓練
        num_labels=num_classes,
    )
    model = DetrForObjectDetection(config)
    
    logger.info(f"正在載入權重: {weight_path}")
    state_dict = torch.load(weight_path, map_location=device)
    model.load_state_dict(state_dict)
    model.to(device)
    model.eval()
    
    return model


# ==========================================
# 3. 執行推論與生成 pred.json
# ==========================================
def inference_and_save(model, test_loader, processor, output_path, threshold=0.5):
    results = []

    logger.info(f"開始推論，信心閥值 (Threshold) 設為: {threshold}")
    
    with torch.no_grad():
        for batch in tqdm(test_loader, desc="Inference"):
            pixel_values = batch["pixel_values"].to(device)
            pixel_mask = batch["pixel_mask"].to(device)
            image_ids = batch["image_ids"]
            original_sizes = batch["original_sizes"]

            # 模型推論
            outputs = model(pixel_values=pixel_values, pixel_mask=pixel_mask)

            # 將 (width, height) 轉換為 DETR 需要的 [[height, width], ...]
            target_sizes = torch.tensor(
                [[h, w] for w, h in original_sizes]
            ).to(device)

            # 後處理：過濾掉信心分數低於 threshold 的預測，並轉回實際圖片像素座標
            processed_results = processor.post_process_object_detection(
                outputs, target_sizes=target_sizes, threshold=threshold
            )

            # 將結果轉為作業要求的 COCO 格式
            for i, result in enumerate(processed_results):
                img_id = image_ids[i]
                scores = result["scores"].cpu().tolist()
                labels = result["labels"].cpu().tolist()
                boxes = result["boxes"].cpu().tolist()

                for score, label, box in zip(scores, labels, boxes):
                    # post_process 輸出的是 [x_min, y_min, x_max, y_max]
                    x_min, y_min, x_max, y_max = box
                    w = x_max - x_min
                    h = y_max - y_min

                    results.append({
                        "image_id": int(img_id),
                        "bbox": [x_min, y_min, w, h],
                        "score": score,
                        "category_id": int(label) + 1  # 記得類別要轉回 1 開始
                    })

    # 儲存為 pred.json (這是 CodaBench 規定的檔名)
    with open(output_path, "w") as f:
        json.dump(results, f, indent=4)
        
    logger.info(f"✅ 推論完成！共生成 {len(results)} 個 Bounding Boxes。")
    logger.info(f"✅ 結果已儲存至 {output_path}")


# ==========================================
# Main 執行區塊
# ==========================================
def main():
    # ------------------ 設定區 ------------------
    test_dir = "./data/test"                   # 測試集圖片資料夾
    weight_path = "./model_weight/detr_best.pth" # 你訓練好的模型權重
    output_json = "pred.json"                  # 必須命名為 pred.json [cite: 172]
    
    batch_size = 8
    num_classes = 10
    confidence_threshold = 0.5  # 可以根據結果微調，例如降到 0.4 抓出更多數字
    
    # --------------------------------------------
    
    # 1. 準備 Processor 與 DataLoader
    processor = AutoImageProcessor.from_pretrained("facebook/detr-resnet-50")
    test_dataset = DigitTestDataset(img_dir=test_dir)
    test_loader = DataLoader(
        test_dataset, 
        batch_size=batch_size, 
        shuffle=False, 
        collate_fn=lambda x: test_collate_fn(x, processor),
        num_workers=4
    )
    
    # 2. 載入模型
    model = load_trained_model(weight_path, num_classes)
    
    # 3. 開始推論
    inference_and_save(
        model=model, 
        test_loader=test_loader, 
        processor=processor, 
        output_path=output_json,
        threshold=confidence_threshold
    )


if __name__ == "__main__":
    main()