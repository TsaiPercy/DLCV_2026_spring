import os
import logging
from datetime import datetime
import random

import numpy as np
import wandb
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, models

# my append py
import func


# ==========================================
# Logger setting
# ==========================================
os.makedirs("./logs", exist_ok=True)
currentTime = datetime.now().strftime("%Y%m%d_%H%M%S")
logPath = f"./logs/training_{currentTime}.log"

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
# setting seed
# ==========================================
def set_seed(seed=60):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    logger.info(f"[Seed] Set random seed = {seed}")


def seed_worker(workerId):
    workerSeed = torch.initial_seed() % 2**32
    np.random.seed(workerSeed)
    random.seed(workerSeed)


# ==========================================
# training main
# ==========================================
def training(trainDir, validDir, savePath, batchSize, numWorkers,
             numEpochs, learningRate, weightDecay, numClasses, dropoutRate,
             model, imgResize, seed, generator, use_StarHead):

    # ==============================================================
    # model setting
    # ==============================================================
    if use_StarHead:
        model = func.build_model_StarHead(
            model=model, numClasses=numClasses, dropoutRate=dropoutRate
        ).to(device)
    else:
        model = func.build_model(
            model=model, numClasses=numClasses, dropoutRate=dropoutRate
        ).to(device)

    logger.info(f"[Training] Model moved to {device}")

    # label smoothing
    label_smoothing = 0.1
    criterion = nn.CrossEntropyLoss(label_smoothing=label_smoothing)

    # Transfer Learning: backbone (smaller lr) vs head (higher lr)
    backboneParams = []
    headParams = []
    for name, param in model.named_parameters():
        if "fc" in name:
            headParams.append(param)
        else:
            backboneParams.append(param)

    optimizer = optim.AdamW([
        {"params": backboneParams, "lr": learningRate * 0.1},
        {"params": headParams, "lr": learningRate},
    ], weight_decay=weightDecay)

    scheduler = optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=numEpochs
    )

    logger.info(f"[Training] Backbone lr={learningRate * 0.1}, "
                f"Head lr={learningRate}")
    logger.info(f"[Training] Optimizer=AdamW, weight_decay={weightDecay}")
    logger.info(f"[Training] Scheduler=CosineAnnealingLR, T_max={numEpochs}")

    # ==============================================================
    # load data
    # ==============================================================
    trainDataset = datasets.ImageFolder(trainDir)
    logger.info(f"[Data] load {len(trainDataset)} train data, "
                f"total {len(trainDataset.classes)} classes")

    valDataset = datasets.ImageFolder(validDir)
    logger.info(f"[Data] load {len(valDataset)} valid data, "
                f"total {len(valDataset.classes)} classes")

    if trainDataset.class_to_idx != valDataset.class_to_idx:
        logger.error("Train and Val class different")
        print("Train:", trainDataset.class_to_idx)
        print("Val:", valDataset.class_to_idx)

    trainTransform, valTransform = func.set_transform(imgResize=imgResize)

    trainDataset.transform = trainTransform
    valDataset.transform = valTransform

    trainLoader = DataLoader(
        trainDataset, batch_size=batchSize, shuffle=True,
        num_workers=numWorkers, worker_init_fn=seed_worker,
        generator=generator
    )
    valLoader = DataLoader(
        valDataset, batch_size=batchSize, shuffle=False,
        num_workers=numWorkers, worker_init_fn=seed_worker,
        generator=generator
    )

    # ==============================================================
    # W&B setting
    # ==============================================================
    wandb.init(
        project="NYCU_DLCV_HW1",
        name=f"ResNet50_lr{learningRate}_bs{batchSize}",
        config={
            "currentTime": currentTime,
            "learning_rate": learningRate,
            "numWorkers": numWorkers,
            "epochs": numEpochs,
            "batch_size": batchSize,
            "weight_decay": weightDecay,
            "dropoutRate": dropoutRate,
            "img_resize": imgResize,
            "optimizer": "AdamW, backbone use 1/10 lr",
            "scheduler": "CosineAnnealingLR(optimizer, T_max=numEpochs)",
            "seed": seed,
            "Loss": "CrossEntropyLoss",
            "label_smoothing": label_smoothing,
            "use_StarHead": use_StarHead,
        }
    )

    # ==============================================================
    # train start
    # ==============================================================
    bestValLoss = 10000.0
    bestValAcc = -1

    for epoch in range(numEpochs):
        trainLoss, trainAcc = func.train_one_epoch(
            model, trainLoader, optimizer, criterion
        )
        valLoss, valAcc = func.eval_one_epoch(model, valLoader, criterion)
        scheduler.step()

        currentLr = optimizer.param_groups[0]["lr"]
        logger.info(f"Epoch [{epoch+1}/{numEpochs}] lr={currentLr:.6f}")

        logger.info(f"   Train Loss: {trainLoss:.4f} | "
                    f"Train Acc: {trainAcc:.5f}")

        logger.info(f"   Val   Loss: {valLoss:.4f} | "
                    f"Val   Acc: {valAcc:.5f}")

        logger.info("-" * 51)

        if valLoss < bestValLoss:
            bestValLoss = valLoss
            bestValAcc = valAcc
            torch.save(model.state_dict(), savePath)
            logger.info(f"Model saved to: {savePath} "
                        f"(Val Loss: {valLoss:.5f}) "
                        f"(Val Acc: {valAcc:.5f})")

        wandb.log({
            "Epoch": epoch+1,
            "Train/Loss": trainLoss,
            "Train/Accuracy": trainAcc,
            "Val/Loss": valLoss,
            "Val/Accuracy": valAcc,
            "Learning_Rate": currentLr
        })

    logger.info(f"[Training] Train finish, Best Val Acc: {bestValAcc:.5f} "
                f"Best Val Loss: {bestValLoss:.5f}")
    wandb.finish()


# ==========================================
# main
# ==========================================
def main():
    seed = 60
    set_seed(seed)
    generator = torch.Generator()
    generator.manual_seed(seed)

    logger.info("=" * 50)
    logger.info("DLCV HW1 - 100 img classification (ResNet)")
    logger.info(f"Using device: {device}")
    logger.info("=" * 50)

    # --------------------------------------------------------------
    # path setting
    # --------------------------------------------------------------
    trainDir = "./data/train"
    validDir = "./data/val"
    savePath = f"./model_weight/resnet50_best_{currentTime}.pth"
    os.makedirs(os.path.dirname(savePath), exist_ok=True)

    # ==============================================================
    # hyper setting
    # ==============================================================
    batchSize = 32
    numWorkers = 8
    numEpochs = 100

    learningRate = 1e-4
    weightDecay = 5e-4
    numClasses = 100
    dropoutRate = 0.5

    # ResNet50 (pretrained on ImageNet, ~25.6M params < 100M)
    model = models.resnet50(weights="IMAGENET1K_V2")
    use_StarHead = True

    # Resize img for input
    imgResize = (448, 448)  # (224, 224) , (448, 448)

    logger.info(f"[Config] batch_size={batchSize}, epochs={numEpochs}")
    logger.info(f"[Config] lr={learningRate}, weight_decay={weightDecay}")
    logger.info(f"[Config] num_classes={numClasses}, img_resize={imgResize}")
    logger.info("[Config] model=ResNet50")

    training(
        trainDir=trainDir,
        validDir=validDir,
        savePath=savePath,
        batchSize=batchSize,
        numWorkers=numWorkers,
        numEpochs=numEpochs,
        learningRate=learningRate,
        weightDecay=weightDecay,
        numClasses=numClasses,
        dropoutRate=dropoutRate,
        model=model,
        imgResize=imgResize,
        seed=seed,
        generator=generator,
        use_StarHead=use_StarHead,
    )


if __name__ == "__main__":
    main()
