import os
import json
import logging
import torch
from torch.utils.data import DataLoader
from transformers import AutoImageProcessor
from tqdm import tqdm

from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval

# 引入你寫好的 func.py
import func

# ==========================================
# 基礎設定
# ==========================================
logging.basicConfig(level=logging.INFO, format="[%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def main():
    logger.info("=" * 50)
    logger.info("DLCV HW2 - COCO Metrics Evaluation")
    logger.info("=" * 50)

    # --------------------------------------------------------------
    # 1. 路徑與參數設定 (請確認這裡符合你的檔案結構)
    # --------------------------------------------------------------
    # 替換成你最新訓練出來的最佳權重檔名
    weight_path = "./model_weight/detr_best_20260415_144031.pth" 
    
    # 驗證集的圖片資料夾與官方標籤
    valid_dir = "./data/valid"
    valid_ann_path = "./data/valid.json"
    
    # 暫存模型預測結果的路徑
    temp_pred_path = "./submission/temp_val_pred.json"
    os.makedirs("./submission", exist_ok=True)

    batch_size = 4
    num_classes = 10
    dropout_rate = 0.2
    
    # 💡 評估時，閥值要調到極低 (例如 0.01 或 0.05)
    # 因為 mAP 需要畫 Precision-Recall 曲線，必須給它足夠多的候選框去算面積
    confidence_threshold = 0.05  

    # --------------------------------------------------------------
    # 2. 準備 DataLoader 與 Model
    # --------------------------------------------------------------
    processor = AutoImageProcessor.from_pretrained("facebook/detr-resnet-50")
    
    # 這裡很巧妙地借用 TestDataset，因為我們只想要圖片和原始尺寸，不需要它讀取標籤
    val_dataset = func.DigitTestDataset(img_dir=valid_dir)
    val_loader = DataLoader(
        val_dataset, 
        batch_size=batch_size, 
        shuffle=False, 
        collate_fn=lambda x: func.test_collate_fn(x, processor),
        num_workers=4
    )

    model = func.build_model(num_classes, dropout_rate, if_train=False, weight_path=weight_path)

    # --------------------------------------------------------------
    # 3. 執行推論，產生預測結果
    # --------------------------------------------------------------
    results = []
    logger.info("開始對驗證集進行推論...")
    
    with torch.no_grad():
        for batch in tqdm(val_loader, desc="Evaluating"):
            pixel_values = batch["pixel_values"].to(device)
            pixel_mask = batch["pixel_mask"].to(device)
            image_ids = batch["image_ids"]
            original_sizes = batch["original_sizes"]

            outputs = model(pixel_values=pixel_values, pixel_mask=pixel_mask)

            # 座標轉換還原
            target_sizes = torch.tensor([[h, w] for w, h in original_sizes]).to(device)
            processed_results = processor.post_process_object_detection(
                outputs, target_sizes=target_sizes, threshold=confidence_threshold
            )

            # 轉換為 COCO 格式
            for i, result in enumerate(processed_results):
                img_id = image_ids[i]
                scores = result["scores"].cpu().tolist()
                labels = result["labels"].cpu().tolist()
                boxes = result["boxes"].cpu().tolist()

                for score, label, box in zip(scores, labels, boxes):
                    x_min, y_min, x_max, y_max = box
                    w = x_max - x_min
                    h = y_max - y_min

                    results.append({
                        "image_id": int(img_id),
                        "bbox": [x_min, y_min, w, h],
                        "score": score,
                        "category_id": int(label) + 1  # 類別加 1 轉回官方格式
                    })

    # 將預測結果存成暫存 JSON
    with open(temp_pred_path, "w") as f:
        json.dump(results, f)
    logger.info(f"預測結果已暫存至 {temp_pred_path}")

    # --------------------------------------------------------------
    # 4. 召喚 pycocotools 進行嚴格評分
    # --------------------------------------------------------------
    logger.info("\n" + "=" * 50)
    logger.info("開始計算 COCO Metrics (mAP)")
    logger.info("=" * 50)

    # 載入官方正確解答 (Ground Truth)
    cocoGt = COCO(valid_ann_path)
    # 載入模型預測結果 (Detection)
    cocoDt = cocoGt.loadRes(temp_pred_path)

    # 啟動評估引擎
    cocoEval = COCOeval(cocoGt, cocoDt, 'bbox')
    cocoEval.evaluate()
    cocoEval.accumulate()
    cocoEval.summarize()  # 這行會印出那張漂亮的終極成績單

if __name__ == "__main__":
    main()