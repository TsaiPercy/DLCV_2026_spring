import os
import json
import logging
import random
import numpy as np
from datetime import datetime

import torch
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from PIL import Image

from transformers import (
    DetrConfig,
    DetrForObjectDetection,
    AutoImageProcessor
)


import wandb
from tqdm import tqdm

# my append py
import func

# ==========================================
# Logger setting
# ==========================================
os.makedirs("./logs", exist_ok=True)
current_time = datetime.now().strftime("%Y%m%d_%H%M%S")
logPath = f"./logs/training_{current_time}.log"


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



def training_pipeline(
    train_dir,
    train_ann,
    valid_dir,
    valid_ann,
    save_path,

    batch_size,
    num_epochs,
    lr,
    lr_backbone,
    dropout_rate,

    weight_decay,
    num_classes,

    seed,
    num_workers,
    generator,

):

    # ==============================================================
    # load data
    # ==============================================================
    processor = AutoImageProcessor.from_pretrained("facebook/detr-resnet-50")

    logger.info(f"[Data] load train data")
    train_dataset = func.DigitDetectionDataset(train_dir, train_ann, processor)
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        collate_fn=lambda x: func.collate_fn(x, processor),
        num_workers=4,
        worker_init_fn=seed_worker,
        generator=generator
    )

    logger.info(f"[Data] load valid data")
    valid_dataset = func.DigitDetectionDataset(valid_dir, valid_ann, processor)
    valid_loader = DataLoader(
        valid_dataset,
        batch_size=batch_size,
        shuffle=False,
        collate_fn=lambda x: func.collate_fn(x, processor),
        num_workers=4,
        worker_init_fn=seed_worker,
        generator=generator
    )


    


    # ==============================================================
    # model setting
    # ==============================================================
    model = func.build_model(num_classes, dropout_rate, True).to(device)
    logger.info(f"[Training] Model moved to {device}")

    param_dicts = [
        {"params": [p for n, p in model.named_parameters()
                    if "backbone" not in n and p.requires_grad],
        "lr": lr},
        {"params": [p for n, p in model.named_parameters()
                    if "backbone" in n and p.requires_grad],
         "lr": lr_backbone},
    ]
    optimizer = optim.AdamW(param_dicts, weight_decay=weight_decay)

    scheduler = optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=num_epochs
    )


    # ==============================================================
    # train start
    # ==============================================================
    logger.info("========================================")
    logger.info("Start training DETR (ResNet-50 Backbone)")
    logger.info("========================================")

    best_val_loss = float('inf')

    for epoch in range(num_epochs):

        # 1. train
        train_loss = func.train_one_epoch(
            model, train_loader, optimizer
        )
        
        # 2. eval
        val_loss = func.eval_one_epoch(model, valid_loader)

        # 3. step scheduler
        scheduler.step()

        # save best
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(), save_path)
            logger.info(f"Model saved to: {save_path} "
                        f"(Val Loss: {val_loss:.5f}) ")
        
        

        # print info
        current_Lr = optimizer.param_groups[0]["lr"]
        logger.info(f"Epoch [{epoch+1}/{num_epochs}] lr={current_Lr:.6f}")

        logger.info(f" Train Loss: {train_loss:.4f}")

        logger.info(f" Val Loss: {val_loss:.4f}")

        logger.info("-" * 51)

        wandb.log({
            "Epoch": epoch+1,
            "Train/Loss": train_loss,
            "Val/Loss": val_loss,
            "Learning_Rate": current_Lr
        })



    logger.info(f"[Training] Train finish")
    logger.info(f"Best Val Loss: {best_val_loss:.5f}")
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
    logger.info("DLCV HW2 - Digit Detection (DETR - ResNet-50)")
    logger.info(f"Using device: {device}")
    logger.info("=" * 50)


    # --------------------------------------------------------------
    # path setting
    # --------------------------------------------------------------
    train_dir = "./data/train"
    train_ann = "./data/train.json"
    valid_dir = "./data/valid"
    valid_ann = "./data/valid.json"
    save_path = f"./model_weight/detr_best_{current_time}.pth"
    os.makedirs(os.path.dirname(save_path), exist_ok=True)



    # ==============================================================
    # hyper setting
    # ==============================================================
    batch_size = 4
    num_epochs = 15
    lr = 1e-4
    lr_backbone = 1e-5
    dropout_rate = 0.2

    weight_decay = 1e-4
    num_classes = 10    # 0-9 => 10 classes 

    num_workers = 8




    logger.info(f"[Config] batch_size={batch_size}, epochs={num_epochs}")
    logger.info(f"[Config] lr={lr}, lr_backbone={lr_backbone}")
    logger.info(f"[Config] dropout_rate={dropout_rate}")
    logger.info(f"[Config] weight_decay={weight_decay}, num_classes={num_classes}")
    logger.info(f"[Config] model=ResNet50")
    

    # ==============================================================
    # W&B setting
    # ==============================================================
    wandb.init(
        project="NYCU_DLCV_HW2",
        name=f"ResNet50_lr{lr}_bs{batch_size}",
        config={
            "current_time": current_time,
            "batch_size": batch_size,
            "num_epochs": num_epochs,

            "lr": lr,
            "lr_backbone": lr_backbone,
            "dropout_rate": dropout_rate,
            
            "weight_decay": weight_decay,
            "num_workers": num_workers,
            
            "optimizer": "AdamW",
            "scheduler": "CosineAnnealingLR(optimizer, T_max=num_epochs)",
            "seed": seed,
            "num_workers": num_workers,
        }
    )



    training_pipeline(
        train_dir=train_dir,
        train_ann=train_ann,
        valid_dir=valid_dir,
        valid_ann=valid_ann,
        save_path=save_path,

        batch_size=batch_size,
        num_epochs=num_epochs,
        lr=lr,
        lr_backbone=lr_backbone,
        dropout_rate=dropout_rate,

        weight_decay=weight_decay,
        num_classes=num_classes,

        seed=seed,
        num_workers=num_workers,
        generator=generator,
    )







if __name__ == "__main__":
    main()