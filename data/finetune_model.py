"""
AgriFarm — Fine-tune ResNet18 on PlantVillage
Downloads dataset FREE from HuggingFace (no account needed).
Trains on CPU — ~45 min for 3000 samples / 5 epochs on t2.micro or c7i.xlarge.
Output: data/plantvillage_resnet18.pt  (auto-loaded by disease_agent.py)

Usage:
  PYTHONPATH=. python data/finetune_model.py --samples 3000 --epochs 5
  PYTHONPATH=. python data/finetune_model.py --samples 1000 --epochs 3  # quick test
  PYTHONPATH=. python data/finetune_model.py --synthetic               # offline test
"""

from __future__ import annotations
import argparse
import json
import time
from pathlib import Path

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, random_split
import torchvision.transforms as T
from torchvision import models
from PIL import Image
from loguru import logger

MODEL_SAVE = Path("./data/plantvillage_resnet18.pt")
HISTORY_F  = Path("./data/training_history.json")
NUM_CLASSES = 38
DEVICE      = torch.device("cpu")   # free tier — CPU only

PLANTVILLAGE_CLASSES = [
    "Apple__Apple_scab", "Apple__Black_rot", "Apple__Cedar_apple_rust", "Apple__healthy",
    "Blueberry__healthy", "Cherry__Powdery_mildew", "Cherry__healthy",
    "Corn__Cercospora_leaf_spot", "Corn__Common_rust", "Corn__Northern_Leaf_Blight", "Corn__healthy",
    "Grape__Black_rot", "Grape__Esca", "Grape__Leaf_blight", "Grape__healthy",
    "Orange__Haunglongbing",
    "Peach__Bacterial_spot", "Peach__healthy",
    "Pepper__Bacterial_spot", "Pepper__healthy",
    "Potato__Early_blight", "Potato__Late_blight", "Potato__healthy",
    "Raspberry__healthy", "Soybean__healthy", "Squash__Powdery_mildew",
    "Strawberry__Leaf_scorch", "Strawberry__healthy",
    "Tomato__Bacterial_spot", "Tomato__Early_blight", "Tomato__Late_blight",
    "Tomato__Leaf_Mold", "Tomato__Septoria_leaf_spot",
    "Tomato__Spider_mites", "Tomato__Target_Spot",
    "Tomato__Yellow_Leaf_Curl_Virus", "Tomato__mosaic_virus", "Tomato__healthy",
]
CLASS_TO_IDX = {c: i for i, c in enumerate(PLANTVILLAGE_CLASSES)}

TRAIN_TRANSFORM = T.Compose([
    T.RandomResizedCrop(224, scale=(0.7, 1.0)),
    T.RandomHorizontalFlip(),
    T.RandomVerticalFlip(),
    T.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.2),
    T.RandomRotation(15),
    T.ToTensor(),
    T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
])
VAL_TRANSFORM = T.Compose([
    T.Resize((224, 224)),
    T.ToTensor(),
    T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
])


# ─────────────────────────────────────────────
# Dataset wrappers
# ─────────────────────────────────────────────
class HFPlantVillage(Dataset):
    """Wraps a HuggingFace dataset split."""

    def __init__(self, hf_ds, transform=None):
        self.ds        = hf_ds
        self.transform = transform
        self.hf_labels = (
            hf_ds.features["label"].names
            if hasattr(hf_ds, "features") and "label" in hf_ds.features
            else PLANTVILLAGE_CLASSES
        )

    def __len__(self):
        return len(self.ds)

    def __getitem__(self, idx):
        item  = self.ds[idx]
        image = item["image"]
        if not isinstance(image, Image.Image):
            image = Image.fromarray(image)
        image = image.convert("RGB")

        hf_idx    = item["label"]
        cls_name  = self.hf_labels[hf_idx] if hf_idx < len(self.hf_labels) else "Tomato__healthy"
        our_idx   = CLASS_TO_IDX.get(cls_name, CLASS_TO_IDX.get("Tomato__healthy", 0))

        if self.transform:
            image = self.transform(image)
        return image, our_idx


class SyntheticLeaves(Dataset):
    """Random green-tinted images — for offline testing only."""

    def __init__(self, size=400, transform=None):
        self.size      = size
        self.transform = transform

    def __len__(self):
        return self.size

    def __getitem__(self, idx):
        import numpy as np
        label = idx % NUM_CLASSES
        r = int(40  + (label * 5)  % 80)
        g = int(110 + (label * 3)  % 90)
        b = int(20  + (label * 7)  % 50)
        arr = (np.random.rand(224, 224, 3) * 35 + [r, g, b]).clip(0, 255).astype("uint8")
        img = Image.fromarray(arr, "RGB")
        if self.transform:
            img = self.transform(img)
        return img, label


# ─────────────────────────────────────────────
# Model
# ─────────────────────────────────────────────
def build_model():
    m = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
    # Freeze all layers except layer4 + fc — saves memory and trains faster on CPU
    for name, param in m.named_parameters():
        if "layer4" not in name and "fc" not in name:
            param.requires_grad = False
    m.fc = nn.Sequential(
        nn.Dropout(0.4),
        nn.Linear(m.fc.in_features, NUM_CLASSES),
    )
    return m.to(DEVICE)


# ─────────────────────────────────────────────
# Train / eval loops
# ─────────────────────────────────────────────
def train_epoch(model, loader, opt, criterion):
    model.train()
    total_loss = correct = total = 0
    for imgs, labels in loader:
        imgs, labels = imgs.to(DEVICE), labels.to(DEVICE)
        opt.zero_grad()
        out  = model(imgs)
        loss = criterion(out, labels)
        loss.backward()
        opt.step()
        total_loss += loss.item() * imgs.size(0)
        correct    += (out.argmax(1) == labels).sum().item()
        total      += imgs.size(0)
    return total_loss / total, correct / total


def eval_epoch(model, loader, criterion):
    model.eval()
    total_loss = correct = total = 0
    with torch.no_grad():
        for imgs, labels in loader:
            imgs, labels = imgs.to(DEVICE), labels.to(DEVICE)
            out  = model(imgs)
            loss = criterion(out, labels)
            total_loss += loss.item() * imgs.size(0)
            correct    += (out.argmax(1) == labels).sum().item()
            total      += imgs.size(0)
    return total_loss / total, correct / total


# ─────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────
def main(samples: int = 3000, epochs: int = 5, batch_size: int = 16, synthetic: bool = False):
    logger.info(f"Device: {DEVICE} | Samples: {samples} | Epochs: {epochs}")

    # ── Dataset ───────────────────────────────
    if synthetic:
        logger.warning("Using SYNTHETIC data — for offline testing only, not real training")
        full_ds = SyntheticLeaves(size=400, transform=TRAIN_TRANSFORM)
        n_train = int(0.8 * len(full_ds))
        train_ds, val_ds = random_split(full_ds, [n_train, len(full_ds) - n_train])
    else:
        logger.info(f"Downloading PlantVillage from HuggingFace ({samples} samples) ...")
        try:
            from datasets import load_dataset   # type: ignore
            hf = load_dataset(
                "sasha/plant-disease-dataset-combined",
                split=f"train[:{samples}]",
                trust_remote_code=True,
            )
            logger.info(f"Downloaded {len(hf)} samples")
        except Exception as e:
            logger.error(f"HuggingFace download failed: {e}")
            logger.info("Falling back to synthetic data (add --synthetic flag next time)")
            return main(synthetic=True)

        full  = HFPlantVillage(hf, transform=None)
        n_tr  = int(0.85 * len(full))
        n_val = len(full) - n_tr
        raw_tr, raw_val = random_split(full, [n_tr, n_val])

        class WithTransform(Dataset):
            def __init__(self, subset, tfm):
                self.s, self.t = subset, tfm
            def __len__(self): return len(self.s)
            def __getitem__(self, i):
                img, lbl = self.s[i]
                if isinstance(img, torch.Tensor): return img, lbl
                return self.t(img), lbl

        train_ds = WithTransform(raw_tr,  TRAIN_TRANSFORM)
        val_ds   = WithTransform(raw_val, VAL_TRANSFORM)

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True,  num_workers=0)
    val_loader   = DataLoader(val_ds,   batch_size=batch_size, shuffle=False, num_workers=0)
    logger.info(f"Train: {len(train_ds)} | Val: {len(val_ds)}")

    # ── Model + optimiser ────────────────────
    model     = build_model()
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=1e-4, weight_decay=1e-4,
    )
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)

    best_val_acc = 0.0
    history      = []

    # ── Training loop ────────────────────────
    for epoch in range(1, epochs + 1):
        t0 = time.time()
        tr_loss, tr_acc = train_epoch(model, train_loader, optimizer, criterion)
        vl_loss, vl_acc = eval_epoch(model, val_loader, criterion)
        scheduler.step()
        elapsed = time.time() - t0

        row = {
            "epoch":     epoch,
            "train_acc": round(tr_acc, 4),
            "val_acc":   round(vl_acc, 4),
            "val_loss":  round(vl_loss, 4),
            "time_s":    round(elapsed, 1),
        }
        history.append(row)
        logger.info(
            f"Epoch {epoch}/{epochs} | "
            f"train {tr_acc:.1%} | val {vl_acc:.1%} | "
            f"loss {vl_loss:.4f} | {elapsed:.0f}s"
        )

        if vl_acc > best_val_acc:
            best_val_acc = vl_acc
            torch.save(model.state_dict(), MODEL_SAVE)
            logger.info(f"  ✓ Saved best model → {MODEL_SAVE}")

    # ── Summary ──────────────────────────────
    with open(HISTORY_F, "w") as f:
        json.dump(history, f, indent=2)

    logger.info(f"\n{'='*50}")
    logger.info(f"Training complete! Best val accuracy: {best_val_acc:.1%}")
    logger.info(f"Model saved to: {MODEL_SAVE}")
    logger.info("Restart the API to load the fine-tuned weights.")
    logger.info(f"{'='*50}")
    return best_val_acc


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Fine-tune ResNet18 on PlantVillage")
    parser.add_argument("--samples",   type=int,  default=3000,
                        help="Number of HuggingFace samples to download (default 3000)")
    parser.add_argument("--epochs",    type=int,  default=5,
                        help="Training epochs (default 5)")
    parser.add_argument("--batch",     type=int,  default=16,
                        help="Batch size (default 16, reduce to 8 if OOM)")
    parser.add_argument("--synthetic", action="store_true",
                        help="Use synthetic data (offline test, no real accuracy)")
    args = parser.parse_args()
    main(samples=args.samples, epochs=args.epochs,
         batch_size=args.batch, synthetic=args.synthetic)
