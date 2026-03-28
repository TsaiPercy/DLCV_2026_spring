import os
import csv
import logging
from datetime import datetime

import torch
from torchvision import datasets, models
from torch.utils.data import DataLoader
from tqdm import tqdm

# my append py
import func


# ==========================================
# Logger setting
# ==========================================
os.makedirs("./logs", exist_ok=True)
currentTime = datetime.now().strftime("%Y%m%d_%H%M%S")
logPath = f"./logs/inference_{currentTime}.log"

logging.basicConfig(
    level=logging.INFO,
    format="[%(levelname)s] %(asctime)s - %(message)s",
    datefmt="%H:%M:%S",
    handlers=[
        # write to log file
        logging.FileHandler(logPath, mode='w', encoding='utf-8'),
        # and print on terminal
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

device = "cuda" if torch.cuda.is_available() else "cpu"


# ==========================================
# inference main
# ==========================================
def inference(testDir,
              model,
              weightPath,
              imgResize,
              batchSize=16,
              use_StarHead=True
              ):

    # --------------------------------------------------------------
    # model load
    # --------------------------------------------------------------
    if use_StarHead:
        model = func.build_model_StarHead(model)
    else:
        model = func.build_model(model)

    model = func.load_weight(model, weightPath)
    model.eval()

    # ==============================================================
    # load data
    # ==============================================================
    _, testTransform = func.set_transform(imgResize=imgResize)

    testDataset = func.ImageDataset(testDir, testTransform)
    testLoader = DataLoader(testDataset, batch_size=batchSize, shuffle=False)

    logger.info(f"[Inference] load {len(testDataset)} test data")

    # adjust label
    trainDataset = datasets.ImageFolder("./data/train")
    idx_to_class = {v: k for k, v in trainDataset.class_to_idx.items()}

    # ==============================================================
    # inference start（argmax）
    # ==============================================================
    result = []

    pbar = tqdm(testLoader, desc="Inference")

    with torch.no_grad():
        for imgs, fileNames in pbar:
            imgs = imgs.to(device)
            outputs = model(imgs)

            preds = outputs.argmax(dim=1)

            for fileName, pred in zip(fileNames, preds.cpu().tolist()):
                imageId = os.path.splitext(fileName)[0]

                real_label = idx_to_class[pred]
                result.append([imageId, real_label])

    logger.info(f"[Inference] Inference, total {len(result)} data")
    return result


# ==========================================
# write CSV
# ==========================================
def write_csv(result,
              outputCsv="./submission/prediction.csv"):

    os.makedirs("submission", exist_ok=True)

    with open(outputCsv, "w", newline="") as csvFile:
        writer = csv.writer(csvFile)
        writer.writerow(["image_name", "pred_label"])
        writer.writerows(result)

    logger.info(f"Saved: {outputCsv} ({len(result)} rows)")


# ==========================================
# main
# ==========================================
def main():
    logger.info("=" * 50)
    logger.info("DLCV HW1 - Inference (Single Model)")
    logger.info(f"Using device: {device}")
    logger.info("=" * 50)

    train_model_name = "20260328_004701"
    testDir = "./data/test"
    weightPath = f"./model_weight/resnet50_best_{train_model_name}.pth"

    outputCsv = f"./submission/prediction_{currentTime}_{train_model_name}.csv"
    os.makedirs(os.path.dirname(outputCsv), exist_ok=True)

    imgResize = (448, 448)  # (224, 224) , (448, 448)
    batchSize = 64
    model = models.resnet50(weights="IMAGENET1K_V2")
    use_StarHead = True

    logger.info(f"[Config] weight_path={weightPath}")
    logger.info(f"[Config] img_resize={imgResize}, batch_size={batchSize}")

    result = inference(testDir=testDir,
                       model=model,
                       weightPath=weightPath,
                       imgResize=imgResize,
                       batchSize=batchSize,
                       use_StarHead=use_StarHead)

    write_csv(
        result=result,
        outputCsv=outputCsv,
    )


if __name__ == "__main__":
    main()
