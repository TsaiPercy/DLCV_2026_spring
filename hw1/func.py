import os
import logging
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms, datasets, models

from PIL import Image
from sklearn.model_selection import StratifiedShuffleSplit

import random
import numpy as np
from tqdm import tqdm


# ==========================================
# Logger setting (get name from main func)
# ==========================================
logger = logging.getLogger(__name__)

device = "cuda" if torch.cuda.is_available() else "cpu"


# ==========================================
# ImageDataset use for load inference dataset
# ==========================================
class ImageDataset(Dataset):
    def __init__(self, imgDir, transform):
        self.imgDir = imgDir
        self.fileNames = sorted(os.listdir(imgDir))
        self.transform = transform
        logger.debug(f"[ImageDataset] load {len(self.fileNames)} imgs from {imgDir}")

    def __len__(self):
        return len(self.fileNames)

    def __getitem__(self, idx):
        fileName = self.fileNames[idx]
        img = Image.open(os.path.join(self.imgDir, fileName)).convert("RGB")
        return self.transform(img), fileName




# ==========================================
# check_model_size
# ==========================================
def check_model_size(model):
    totalParams = sum(p.numel() for p in model.parameters())
    totalParamsInMillion = totalParams / 1e6

    logger.info(f"========================================")
    logger.info(f"Total Params (M): {totalParamsInMillion:.2f} M")
    logger.info(f"========================================")

    if totalParamsInMillion >= 100:
        logger.warning(f"model_size over 100M: now ({totalParamsInMillion:.2f}M)")
    else:
        logger.info(f"model_size correctly < 100M)")

    return totalParamsInMillion


# ==========================================
# build ResNet
# ==========================================
def build_model(model, numClasses=100, dropoutRate=0.5):
    """
    change ResNet model fc to 100 class
    """
    inFeatures = model.fc.in_features

    model.fc = nn.Sequential(
        nn.Dropout(p=dropoutRate),
        nn.Linear(inFeatures, numClasses)
    )

    logger.debug(f"[build_model] fc layer change:"
                    f"in_features={inFeatures}, Dropout(p={dropoutRate}), num_classes={numClasses}")
    check_model_size(model)

    return model


# ==========================================
# load_weight
# ==========================================
def load_weight(model, weightPath):
    logger.debug(f"[load_weight] load_weight: {weightPath}")
    stateDict = torch.load(weightPath, map_location=device)
    model.load_state_dict(stateDict)
    model.to(device)
    logger.info(f"[load_weight] load_weight finish")
    return model


# ==========================================
# train_one_epoch
# ==========================================
def train_one_epoch(model, dataloader, optimizer, criterion):
    model.train()
    totalLoss = 0
    correctCount = 0
    totalCount = 0

    pbar = tqdm(dataloader, desc="Training")

    for imgs, labels in pbar:
        imgs, labels = imgs.to(device), labels.to(device)

        optimizer.zero_grad()

        outputs = model(imgs)
        loss = criterion(outputs, labels)

        loss.backward()
        optimizer.step()

        totalLoss += loss.item() * imgs.size(0)

        preds = outputs.argmax(dim=1)
        correctCount += (preds == labels).sum().item()
        totalCount += imgs.size(0)

        pbar.set_postfix({
            "loss": f"{totalLoss/totalCount:.5f}",
            "acc": f"{correctCount/totalCount:.5f}"
        })

    epochLoss = totalLoss / totalCount
    epochAcc = correctCount / totalCount
    logger.debug(f"[train_one_epoch] loss={epochLoss:.5f}, acc={epochAcc:.5f}")
    return epochLoss, epochAcc


# ==========================================
# eval_one_epoch
# ==========================================
def eval_one_epoch(model, dataloader, criterion):
    model.eval()
    totalLoss = 0
    correctCount = 0
    totalCount = 0

    pbar = tqdm(dataloader, desc="Evaluating")

    with torch.no_grad():
        for imgs, labels in pbar:
            imgs, labels = imgs.to(device), labels.to(device)

            outputs = model(imgs)
            loss = criterion(outputs, labels)

            totalLoss += loss.item() * imgs.size(0)

            preds = outputs.argmax(dim=1)
            correctCount += (preds == labels).sum().item()
            totalCount += imgs.size(0)

            pbar.set_postfix({
                "loss": f"{totalLoss/totalCount:.4f}",
                "acc": f"{correctCount/totalCount:.4f}"
            })

    epochLoss = totalLoss / totalCount
    epochAcc = correctCount / totalCount
    logger.debug(f"[eval_one_epoch] loss={epochLoss:.5f}, acc={epochAcc:.5f}")
    return epochLoss, epochAcc


# ==========================================
# Data Augmentation
# ==========================================
def set_transform(imgResize):
    MEAN = [0.485, 0.456, 0.406]
    STD = [0.229, 0.224, 0.225]

    trainTransform = transforms.Compose([
        transforms.RandomResizedCrop(imgResize[0], scale=(0.7, 1.0)),
        transforms.RandomHorizontalFlip(),
        transforms.TrivialAugmentWide(),
        transforms.ToTensor(),
        transforms.Normalize(mean=MEAN, std=STD),
        transforms.RandomErasing(p=0.2),
    ])

    valTransform = transforms.Compose([
        transforms.Resize(int(imgResize[0] * 1.15)),
        transforms.CenterCrop(imgResize[0]),
        transforms.ToTensor(),
        transforms.Normalize(mean=MEAN, std=STD),
    ])

    logger.debug(f"[set_transform] imgResize={imgResize}")
    return trainTransform, valTransform


"""
class FocalLoss(nn.Module):
    def __init__(self, gamma=2.0, label_smoothing=0.1):
        super(FocalLoss, self).__init__()
        self.gamma = gamma
        self.label_smoothing = label_smoothing

    def forward(self, inputs, targets):
        ce_loss = F.cross_entropy(inputs, targets, 
                                  label_smoothing=self.label_smoothing, 
                                  reduction='none')
        pt = torch.exp(-ce_loss)
        
        focal_loss = ((1 - pt) ** self.gamma) * ce_loss
        return focal_loss.mean()
"""


class StarHead(nn.Module):
    def __init__(self, in_features, num_classes, dropout_rate=0.5):
        super(StarHead, self).__init__()
        
        self.fc1 = nn.Linear(in_features, in_features)
        self.fc2 = nn.Linear(in_features, in_features)
        
        # activate function
        self.act = nn.GELU()
        
        self.dropout = nn.Dropout(p=dropout_rate)
        self.classifier = nn.Linear(in_features, num_classes)

    def forward(self, x):
        # 1. Star Operation
        x1 = self.fc1(x)
        x2 = self.fc2(x)
        x_star = x1 * x2  
        
        # 2. activate func and Dropout
        x_out = self.act(x_star)
        x_out = self.dropout(x_out)
        
        return self.classifier(x_out)


def build_model_StarHead(model, numClasses=100, dropoutRate=0.5):
    inFeatures = model.fc.in_features

    model.fc = StarHead(in_features=inFeatures, 
                        num_classes=numClasses, 
                        dropout_rate=dropoutRate)
    
    logger.debug(f"[build_model] fc layer use StarHead:"
                 f"in_features={inFeatures}, num_classes={numClasses}")
    return model
