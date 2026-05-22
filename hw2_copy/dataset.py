"""COCO-format dataset for digit detection (HW2)."""

import json
import os
import random

import torch
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms


IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]


def _resize(img, max_side=640, min_side=480):
    """Resize image so shorter side >= min_side and longer side <= max_side."""
    w, h = img.size
    scale = min_side / min(w, h)
    if scale * max(w, h) > max_side:
        scale = max_side / max(w, h)
    new_w = max(1, int(round(w * scale)))
    new_h = max(1, int(round(h * scale)))
    return img.resize((new_w, new_h), Image.BILINEAR), scale


def get_train_transforms():
    return transforms.Compose([
        transforms.ColorJitter(
            brightness=0.4, contrast=0.4, saturation=0.4, hue=0.1
        ),
        transforms.ToTensor(),
        transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
    ])


def get_val_transforms():
    return transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
    ])


class COCODetectionDataset(Dataset):
    """Dataset for COCO-format object detection annotations."""

    def __init__(self, img_dir, ann_file, transforms=None, max_side=640, min_side=480, train_scales=None):
        self.img_dir = img_dir
        self.transforms = transforms
        self.max_side = max_side
        self.min_side = min_side
        self.train_scales = train_scales  # list → scale jitter; None → use min_side

        with open(ann_file) as f:
            data = json.load(f)

        self.id_to_info = {img["id"]: img for img in data["images"]}

        self.ann_by_image = {}
        for ann in data["annotations"]:
            iid = ann["image_id"]
            self.ann_by_image.setdefault(iid, []).append(ann)

        self.image_ids = list(self.id_to_info.keys())

    def __len__(self):
        return len(self.image_ids)

    def __getitem__(self, idx):
        img_id = self.image_ids[idx]
        info = self.id_to_info[img_id]

        img_path = os.path.join(self.img_dir, info["file_name"])
        img = Image.open(img_path).convert("RGB")
        orig_w, orig_h = img.size

        min_side = random.choice(self.train_scales) if self.train_scales else self.min_side
        img, scale = _resize(img, self.max_side, min_side)
        new_w, new_h = img.size

        anns = self.ann_by_image.get(img_id, [])
        boxes, labels = [], []
        for ann in anns:
            x, y, w, h = ann["bbox"]
            # Scale bbox, then convert to normalized [cx, cy, w, h]
            x = x * scale
            y = y * scale
            w = w * scale
            h = h * scale
            cx = (x + w / 2) / new_w
            cy = (y + h / 2) / new_h
            boxes.append([cx, cy, w / new_w, h / new_h])
            labels.append(ann["category_id"])  # 1-indexed (1–10)

        if self.transforms:
            img = self.transforms(img)

        target = {
            "image_id": img_id,
            "boxes": (
                torch.tensor(boxes, dtype=torch.float32)
                if boxes
                else torch.zeros((0, 4), dtype=torch.float32)
            ),
            "labels": (
                torch.tensor(labels, dtype=torch.long)
                if labels
                else torch.zeros(0, dtype=torch.long)
            ),
            "orig_size": torch.tensor([orig_h, orig_w]),
            "size": torch.tensor([new_h, new_w]),
        }
        return img, target


class TestDataset(Dataset):
    """Test dataset without annotations; image_id derived from filename."""

    def __init__(self, img_dir, transforms=None, max_side=640, min_side=480):
        self.img_dir = img_dir
        self.transforms = transforms
        self.max_side = max_side
        self.min_side = min_side
        self.files = sorted(
            f for f in os.listdir(img_dir)
            if f.lower().endswith((".png", ".jpg", ".jpeg"))
        )

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        fname = self.files[idx]
        img_path = os.path.join(self.img_dir, fname)
        img = Image.open(img_path).convert("RGB")
        orig_w, orig_h = img.size

        img, _ = _resize(img, self.max_side, self.min_side)
        new_w, new_h = img.size

        if self.transforms:
            img = self.transforms(img)

        image_id = int(os.path.splitext(fname)[0])
        return img, {
            "image_id": image_id,
            "orig_size": torch.tensor([orig_h, orig_w]),
            "size": torch.tensor([new_h, new_w]),
        }


def collate_fn(batch):
    """Pad images to the same spatial size within a batch."""
    images, targets = zip(*batch)

    max_h = max(img.shape[1] for img in images)
    max_w = max(img.shape[2] for img in images)

    padded = torch.zeros(len(images), 3, max_h, max_w)
    # mask: True = padding region (ignored)
    masks = torch.ones(len(images), max_h, max_w, dtype=torch.bool)

    for i, img in enumerate(images):
        h, w = img.shape[1], img.shape[2]
        padded[i, :, :h, :w] = img
        masks[i, :h, :w] = False

    return padded, masks, list(targets)


def build_dataloaders(
    data_dir, batch_size=4, num_workers=4, max_side=640, min_side=480, train_scales=None
):
    train_dataset = COCODetectionDataset(
        img_dir=os.path.join(data_dir, "train"),
        ann_file=os.path.join(data_dir, "train.json"),
        transforms=get_train_transforms(),
        max_side=max_side,
        min_side=min_side,
        train_scales=train_scales,
    )
    val_dataset = COCODetectionDataset(
        img_dir=os.path.join(data_dir, "valid"),
        ann_file=os.path.join(data_dir, "valid.json"),
        transforms=get_val_transforms(),
        max_side=max_side,
        min_side=min_side,
    )
    test_dataset = TestDataset(
        img_dir=os.path.join(data_dir, "test"),
        transforms=get_val_transforms(),
        max_side=max_side,
        min_side=min_side,
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        collate_fn=collate_fn,
        pin_memory=True,
        drop_last=True,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        collate_fn=collate_fn,
        pin_memory=True,
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        collate_fn=collate_fn,
        pin_memory=True,
    )

    return train_loader, val_loader, test_loader
