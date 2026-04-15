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



# ==========================================
# Logger setting (get name from main func)
# ==========================================
logger = logging.getLogger(__name__)


# ==========================================
# device
# ==========================================
device = "cuda" if torch.cuda.is_available() else "cpu"


# ==========================================
# Dataset (Training use) [process BBox and class change]
# ==========================================
class DigitDetectionDataset(Dataset):
    def __init__(self, img_dir, annotation_file, processor):
        self.img_dir = img_dir
        self.processor = processor

        self.coco_data = self._get_coco_json_data(annotation_file)
        

        self.images = {img['id']: img for img in self.coco_data['images']}
        self.annotations = self.coco_data['annotations']

        # put multi anns into image_id 
        self.img_to_anns = {}

        for ann in self.annotations:
            img_id = ann['image_id']
            if img_id not in self.img_to_anns:
                self.img_to_anns[img_id] = []
            self.img_to_anns[img_id].append(ann)

        self.image_ids = list(self.images.keys())

    def __len__(self):
        return len(self.image_ids)

    def _get_coco_json_data(self, annotation_file):
        with open(annotation_file, 'r') as f:
            coco_data = json.load(f)
        return coco_data

    def __getitem__(self, idx):
        img_id = self.image_ids[idx]
        img_info = self.images[img_id]
        img_path = os.path.join(self.img_dir, img_info['file_name'])

        image = Image.open(img_path).convert("RGB")
        anns = self.img_to_anns.get(img_id, [])

        target = {
            "image_id": img_id, 
            "annotations": anns
        }

        return image, target


def collate_fn(batch, processor):
    """
    Dynamic Padding: 
    process imgs to same as (max size img in this batch)
    """
    images = [item[0] for item in batch]
    targets = [item[1] for item in batch]

    encoding = processor(
        images=images, 
        annotations=targets,
        return_tensors="pt"
    )

    labels = encoding["labels"]
    for label_dict in labels:
        label_dict["class_labels"] = label_dict["class_labels"] - 1


    # pixel_mask is the padding black part
    batch_dict = {
        "pixel_values": encoding["pixel_values"],
        "pixel_mask": encoding["pixel_mask"],
        "labels": labels
    }
    return batch_dict





# ==========================================
# Dataset (Testing use) [no label]
# ==========================================
class DigitTestDataset(Dataset):
    def __init__(self, img_dir):
        self.img_dir = img_dir
        
        self.image_files = [
            f for f in os.listdir(img_dir)
            if f.lower().endswith(('.png', '.jpg', '.jpeg'))
        ]
        
    def __len__(self):
        return len(self.image_files)

    def __getitem__(self, idx):
        file_name = self.image_files[idx]
        img_path = os.path.join(self.img_dir, file_name)
        
        # file name = image_id ("123.jpg" -> 123)
        image_id = int(os.path.splitext(file_name)[0])
        
        image = Image.open(img_path).convert("RGB")
        original_size = image.size  # (width, height)
        
        return image, image_id, original_size


def test_collate_fn(batch, processor):
    """
    test use collate_fn, no labels
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
# build model
# ==========================================
def build_model(
    num_classes, 
    num_query, 
    dropout_rate, 
    if_train, 
    weight_path=""
):
    """
    pretrained_backbone ResNet-50, init Encoder/Decoder of DETR
    """
    config = DetrConfig(
        backbone="resnet50",
        use_pretrained_backbone=if_train,
        num_labels=num_classes,
        num_query=num_query,                       # default 100
        init_std=0.02,
        init_xavier_std=1.0,

        
        
        # append Dropout
        # dropout=dropout_rate,            # default 0.1
        # attention_dropout=0.1,           # Attention Dropout
        # activation_dropout=0.1,          # FFN Dropout
        
        
    
        auxiliary_loss=True,
    )
    # use pretrain ResNet to init DETR 
    model = DetrForObjectDetection(config)


    if not if_train:
        logger.info(f"load weight: {weight_path}")
        state_dict = torch.load(weight_path, map_location=device)
        model.load_state_dict(state_dict)
        model.to(device)
        model.eval()



    return model









# ==========================================
# train_one_epoch
# ==========================================
def train_one_epoch(model, train_loader, optimizer):
    model.train()
    total_loss = 0

    pbar = tqdm(train_loader, desc="Training")

    for batch in pbar:
        pixel_values = batch["pixel_values"].to(device)
        pixel_mask = batch["pixel_mask"].to(device)

        labels = [{k: v.to(device) for k, v in t.items()}
                  for t in batch["labels"]]

        optimizer.zero_grad()

        outputs = model(pixel_values=pixel_values,
                        pixel_mask=pixel_mask,
                        labels=labels)

        loss = outputs.loss
        loss.backward()

        # Gradient clip
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=0.1)
        optimizer.step()

        total_loss += loss.item()
        pbar.set_postfix({"Loss": f"{loss.item():.4f}"})

    avg_loss = total_loss / len(train_loader)
    logger.info(f"Avg Train Loss: {avg_loss:.4f}")
    return avg_loss




# ==========================================
# eval_one_epoch
# ==========================================
def eval_one_epoch(model, valid_loader):
    model.eval()
    total_loss = 0

    pbar = tqdm(valid_loader, desc=f"Evaluating")
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
    logger.info(f"Avg Eval Loss: {avg_loss:.4f}")
    return avg_loss

