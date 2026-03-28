import os
import logging

import torch
import torch.nn as nn
from torch.utils.data import Dataset
from torchvision import transforms
from PIL import Image
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
        logger.debug(f"[ImageDataset] load {len(self.fileNames)} "
                     f"imgs from {imgDir}")

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

    logger.info("========================================")
    logger.info(f"Total Params (M): {totalParamsInMillion:.2f} M")
    logger.info("========================================")

    if totalParamsInMillion >= 100:
        logger.warning("model_size over 100M: "
                       f"now ({totalParamsInMillion:.2f}M)")
    else:
        logger.info("model_size correctly < 100M)")

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

    logger.debug("[build_model] fc layer change: "
                 f"in_features={inFeatures}, Dropout(p={dropoutRate}), "
                 f"num_classes={numClasses}")
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
    logger.info("[load_weight] load_weight finish")
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
    logger.debug(f"[train_one_epoch] loss={epochLoss:.5f}, "
                 f"acc={epochAcc:.5f}")
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
    logger.debug(f"[eval_one_epoch] loss={epochLoss:.5f}, "
                 f"acc={epochAcc:.5f}")
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

    logger.debug("[build_model] fc layer use StarHead: "
                 f"in_features={inFeatures}, num_classes={numClasses}")
    check_model_size(model)

    return model
