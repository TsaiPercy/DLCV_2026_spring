import os
import json
import logging
from datetime import datetime
from PIL import Image
from tqdm import tqdm
import torch

import func

os.makedirs("./logs", exist_ok=True)
current_time = datetime.now().strftime("%Y%m%d_%H%M%S")
logging.basicConfig(level=logging.INFO, format="[%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def main():
    logger.info("--- Starting Inference ---")
    
    # 參數設定
    test_folder = "./data/test"
    # 記得替換成你最新訓練好的權重名稱！
    weight_file = "./model_weight/detr_custom_YOUR_TIMESTAMP.pth" 
    out_json = f"./submission/pred_{current_time}.json"
    os.makedirs("./submission", exist_ok=True)
    
    threshold = 0.5
    
    # 建立模型
    model = func.HW2CustomDETR(queries=100).to(device)
    model.load_state_dict(torch.load(weight_file, map_location=device))
    model.eval()
    
    # 圖片預處理 Pipeline
    from torchvision import transforms
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    image_files = [f for f in os.listdir(test_folder) if f.endswith(('.jpg', '.png'))]
    results_list = []
    
    logger.info(f"Found {len(image_files)} test images.")
    
    with torch.no_grad():
        for file_name in tqdm(image_files, desc="Inferencing"):
            img_id = int(os.path.splitext(file_name)[0])
            img_path = os.path.join(test_folder, file_name)
            
            raw_img = Image.open(img_path).convert("RGB")
            orig_w, orig_h = raw_img.size
            
            resized_img, scale_factor = func.resize_image_hw2(raw_img)
            new_w, new_h = resized_img.size
            
            img_tensor = transform(resized_img).unsqueeze(0).to(device)
            mask_tensor = torch.zeros((1, new_h, new_w), dtype=torch.bool).to(device)
            
            # 推論
            outputs = model(img_tensor, mask_tensor)
            probs = outputs["pred_logits"][0].softmax(-1)
            boxes = outputs["pred_boxes"][0] # cxcywh (0~1)
            
            scores, class_preds = probs[:, :-1].max(dim=-1)
            
            for score, cls_idx, box in zip(scores, class_preds, boxes):
                if score.item() > threshold:
                    # 還原座標到原圖大小
                    cx, cy, w, h = box.tolist()
                    cx, cy = cx * new_w, cy * new_h
                    w, h = w * new_w, h * new_h
                    
                    x_min = (cx - w / 2) / scale_factor
                    y_min = (cy - h / 2) / scale_factor
                    w_orig = w / scale_factor
                    h_orig = h / scale_factor
                    
                    results_list.append({
                        "image_id": img_id,
                        "bbox": [x_min, y_min, w_orig, h_orig],
                        "score": score.item(),
                        "category_id": int(cls_idx.item()) + 1 # 轉回 1~10
                    })
                    
    # 儲存 JSON
    with open(out_json, "w") as f:
        json.dump(results_list, f)
        
    logger.info(f"Inference Complete! Saved to {out_json}")

if __name__ == "__main__":
    main()