import os
import logging
import random
import numpy as np
from datetime import datetime

import torch
import torch.optim as optim
from torch.utils.data import DataLoader
import wandb

# 引入你的防抄襲重構版 func
import func

# ==========================================
# 基礎與環境設定
# ==========================================
os.makedirs("./logs", exist_ok=True)
current_time = datetime.now().strftime("%Y%m%d_%H%M%S")
log_file = f"./logs/training_hw2_{current_time}.log"

logging.basicConfig(level=logging.INFO, format="[%(levelname)s] %(asctime)s - %(message)s",
                    handlers=[logging.FileHandler(log_file, mode='w', encoding='utf-8'), logging.StreamHandler()])
logger = logging.getLogger(__name__)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def fix_random_seeds(seed=60):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def worker_seeder(worker_id):
    seed = torch.initial_seed() % 2**32
    np.random.seed(seed)
    random.seed(seed)

def main():
    fix_random_seeds(60)
    
    # 參數設定
    cfg = {
        "train_dir": "./data/train",
        "train_json": "./data/train.json",
        "val_dir": "./data/valid",
        "val_json": "./data/valid.json",
        "save_model": f"./model_weight/detr_custom_{current_time}.pth",
        "epochs": 50,
        "batch_size": 4,
        "lr_transformer": 1e-4,
        "lr_resnet": 1e-5,
        "num_queries": 20
    }
    os.makedirs("./model_weight", exist_ok=True)

    wandb.init(project="NYCU_DLCV_HW2", name=f"CustomDETR_Run_{current_time}", config=cfg)

    # 1. 準備 DataLoader
    logger.info("Initializing DataLoaders...")
    ds_train = func.HW2DigitDataset(cfg["train_dir"], cfg["train_json"], is_training=True)
    loader_train = DataLoader(ds_train, batch_size=cfg["batch_size"], shuffle=True, 
                              collate_fn=func.hw2_collate_fn, num_workers=4, 
                              worker_init_fn=worker_seeder, 
                              drop_last=True)  # 🌟 加上這個

    ds_val = func.HW2DigitDataset(cfg["val_dir"], cfg["val_json"], is_training=False)
    loader_val = DataLoader(ds_val, batch_size=cfg["batch_size"], shuffle=False, 
                            collate_fn=func.hw2_collate_fn, num_workers=4, worker_init_fn=worker_seeder)

    # 2. 建立模型與優化器
    logger.info("Building Custom DETR Model...")
    model = func.HW2CustomDETR(queries=cfg["num_queries"]).to(device)
    loss_module = func.HW2DETRLoss().to(device)
    
    params_cnn = list(model.cnn_backbone.parameters())
    params_trans = [p for n, p in model.named_parameters() if not n.startswith("cnn_backbone.")]
    
    optimizer = optim.AdamW([
        {"params": params_cnn, "lr": cfg["lr_resnet"]},
        {"params": params_trans, "lr": cfg["lr_transformer"]}
    ], weight_decay=1e-4)
    
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=cfg["epochs"])
    amp_scaler = torch.amp.GradScaler('cuda')

    # 3. 開始訓練
    best_loss = float('inf')
    logger.info("--- Training Started ---")
    
    for ep in range(1, cfg["epochs"] + 1):
        loss_t = func.train_epoch_step(model, loader_train, optimizer, loss_module, amp_scaler)
        loss_v = func.eval_epoch_step(model, loader_val, loss_module)
        
        current_lr = optimizer.param_groups[1]['lr']
        scheduler.step()
        
        logger.info(f"Epoch {ep:02d} | Train Loss: {loss_t:.4f} | Val Loss: {loss_v:.4f} | LR: {current_lr:.2e}")
        
        if loss_v < best_loss:
            best_loss = loss_v
            torch.save(model.state_dict(), cfg["save_model"])
            logger.info(f"   >>> Model Saved to {cfg['save_model']}")
            
        wandb.log({"Train Loss": loss_t, "Val Loss": loss_v, "Learning Rate": current_lr, "Epoch": ep})

    wandb.finish()
    logger.info("--- Training Finished ---")

if __name__ == "__main__":
    main()