import os
import json
import logging
from tqdm import tqdm
from PIL import Image
from datetime import datetime

import torch
from torch.utils.data import Dataset, DataLoader
from transformers import (
    DetrConfig,
    DetrForObjectDetection,
    AutoImageProcessor
)


# my append py
import func




# ==========================================
# Logger setting
# ==========================================
os.makedirs("./logs", exist_ok=True)
current_time = datetime.now().strftime("%Y%m%d_%H%M%S")
logPath = f"./logs/inference_{current_time}.log"


logging.basicConfig(
    level=logging.INFO,
    format="[%(levelname)s] %(asctime)s - %(message)s",
    datefmt="%H:%M:%S",
    handlers=[
        logging.FileHandler(logPath, mode='w', encoding='utf-8'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# ==========================================
# device
# ==========================================
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")



# ==========================================
# inference_and_save
# ==========================================
def inference_and_save(
    model, 
    test_loader, 
    processor, 
    output_path, 
    threshold=0.5
):
    results = []

    logger.info(f"[Inference] Threshold= {threshold}")
    
    with torch.no_grad():
        for batch in tqdm(test_loader, desc="Inference"):

            pixel_values = batch["pixel_values"].to(device)
            pixel_mask = batch["pixel_mask"].to(device)

            image_ids = batch["image_ids"]
            original_sizes = batch["original_sizes"]

            # inference
            outputs = model(pixel_values=pixel_values, pixel_mask=pixel_mask)

            # from 0~1 proportion (width, height) back to ori DETR (height, width)
            target_sizes = torch.tensor(
                [[h, w] for w, h in original_sizes]
            ).to(device)

            # post process: filter confidence lower than threshold
            processed_results = processor.post_process_object_detection(
                outputs, target_sizes=target_sizes, threshold=threshold
            )

            # COCO
            for i, result in enumerate(processed_results):
                img_id = image_ids[i]
                scores = result["scores"].cpu().tolist()
                labels = result["labels"].cpu().tolist()
                boxes = result["boxes"].cpu().tolist()


                # Pascal VOC to COCO
                for score, label, box in zip(scores, labels, boxes):
                    # Pascal VOC ==> [x_min, y_min, x_max, y_max]
                    x_min, y_min, x_max, y_max = box
                    w = x_max - x_min
                    h = y_max - y_min

                    results.append({
                        "image_id": int(img_id),
                        "bbox": [x_min, y_min, w, h],
                        "score": score,

                        # class transform back to start from 1
                        "category_id": int(label) + 1  
                    })

    # save pred.json
    with open(output_path, "w") as f:
        json.dump(results, f, indent=4)
    

    logger.info(f"[Inference] Inference finish")
    logger.info(f"Total {len(results)} Bounding Boxes")
    logger.info(f"Save to {output_path}")


# ==========================================
# main
# ==========================================
def main():

    logger.info("=" * 50)
    logger.info("DLCV HW2 - Inference")
    logger.info(f"Using device: {device}")
    logger.info("=" * 50)


    # --------------------------------------------------------------
    # path setting
    # --------------------------------------------------------------
    train_model_name = "20260414_212539"

    test_dir = "./data/test"
    weight_path = f"./model_weight/detr_best_{train_model_name}.pth"
    output_json = f"./submission/pred_{current_time}_{train_model_name}.json"
    os.makedirs("submission", exist_ok=True)





    # ==============================================================
    # hyper setting
    # ==============================================================
    batch_size = 1
    num_classes = 10
    dropout_rate = 0.2
    confidence_threshold = 0.05  # 可以根據結果微調，例如降到 0.4 抓出更多數字
    
    # --------------------------------------------
    
    # 1. prepare Processor and DataLoader
    processor = AutoImageProcessor.from_pretrained("facebook/detr-resnet-50")
    test_dataset = func.DigitTestDataset(img_dir=test_dir)
    test_loader = DataLoader(
        test_dataset, 
        batch_size=batch_size, 
        shuffle=False, 
        collate_fn=lambda x: func.test_collate_fn(x, processor),
        num_workers=4,
    )
    
    # 2. load_model
    model = func.build_model(
        num_classes, 
        dropout_rate, 
        False, 
        weight_path,
    )
    
    # 3. start inference
    inference_and_save(
        model=model, 
        test_loader=test_loader, 
        processor=processor, 
        output_path=output_json,
        threshold=confidence_threshold
    )


if __name__ == "__main__":
    main()