import os
import re
import csv
import math
import time
import glob
import random
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader, random_split

import albumentations as A
from albumentations.pytorch import ToTensorV2

import segmentation_models_pytorch as smp

from dotenv import load_dotenv
import os

load_dotenv()
ROOT = r"C:\Users\victo\Downloads\Mestrado\TNBC\preproc_unet_per_channel"
OUT_CSV = "results_per_channel_v2.csv"

IMG_SIZE = 512 
VAL_SPLIT = 0.2
BATCH_SIZE = 4 
EPOCHS = 5 
LR = 1e-4 
NUM_WORKERS = 2 
SEED = 1337

ENCODER = "resnet34" 
ENCODER_WEIGHTS = None 
ACTIVATION = None

IGNORE_DIRS = set(["Background", "Segmentation", "SegmentationInterior"])

# ROOT = os.getenv("ROOT")

# OUT_CSV = os.getenv("OUT_CSV")

# IMG_SIZE = int(os.getenv("IMG_SIZE", 512))
# VAL_SPLIT = float(os.getenv("VAL_SPLIT", 0.2))
# BATCH_SIZE = int(os.getenv("BATCH_SIZE", 4))
# EPOCHS = int(os.getenv("EPOCHS", 5))
# LR = float(os.getenv("LR", 1e-4))
# NUM_WORKERS = int(os.getenv("NUM_WORKERS", 2))
# SEED = int(os.getenv("SEED", 1337))

# ENCODER = os.getenv("ENCODER", "resnet18")
# ENCODER_WEIGHTS = os.getenv("ENCODER_WEIGHTS", None)
# ACTIVATION = os.getenv("ACTIVATION", None)

# ignore_dirs_str = os.getenv("IGNORE_DIRS", "")
# IGNORE_DIRS = set(d.strip() for d in ignore_dirs_str.split(",") if d.strip())

# IGNORE_DIRS = set([])  # ex.: {"Background", "Segmentation", "SegmentationInterior"}

def set_seed(seed: int = 1337):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def list_channels(root):
    chans = []
    for p in sorted(Path(root).iterdir()):
        if p.is_dir() and p.name not in IGNORE_DIRS:
            if (p / "images").is_dir() and (p / "masks").is_dir():
                chans.append(p.name)
    return chans


def build_pairs(images_dir, masks_dir):
    """
    Faz o pareamento por "nome base".
    Regras:
      - imagem: <nome>.(tif|tiff)
      - máscara: <nome>_mask.png  OU <nome>.png
    """
    img_paths = sorted(glob.glob(str(Path(images_dir) / "*.tif"))) + \
                sorted(glob.glob(str(Path(images_dir) / "*.tiff")))

    img_map = {}
    for ip in img_paths:
        stem = Path(ip).stem
        img_map[stem] = ip

    mask_paths = sorted(glob.glob(str(Path(masks_dir) / "*.png")))
    pairs = []
    for mp in mask_paths:
        mstem = Path(mp).stem
        # tenta remover sufixo _mask
        if mstem.endswith("_mask"):
            stem = mstem[:-5]
        else:
            stem = mstem
        if stem in img_map:
            pairs.append((img_map[stem], mp))
        else:
            # fallback: tenta casar por regex mais flexível
            base = re.sub(r"_mask$", "", mstem)
            if base in img_map:
                pairs.append((img_map[base], mp))
    return pairs


class SingleChannelSegDataset(Dataset):
    def __init__(self, pairs, img_size=512):
        self.pairs = pairs
        self.img_size = img_size

        self.resize = A.Resize(img_size, img_size, interpolation=1)

    def __len__(self):
        return len(self.pairs)

    def __getitem__(self, idx):
        img_path, mask_path = self.pairs[idx]
        
        with Image.open(img_path) as im:
            im = im.convert("I") if im.mode not in ("I;16", "I") else im
            img_np = np.array(im, dtype=np.float32)
        
        if img_np.max() > 0:
            img_np = img_np / img_np.max()
        
        with Image.open(mask_path) as mm:
            m_np = np.array(mm.convert("L"), dtype=np.uint8)
        m_np = (m_np > 0).astype(np.float32)

        resized = self.resize(image=img_np, mask=m_np)
        img_resized = resized["image"]
        mask_resized = resized["mask"]
        
        image = torch.from_numpy(img_resized).unsqueeze(0).float()
        mask = torch.from_numpy(mask_resized).unsqueeze(0).float()

        return image, mask


def dice_coeff(y_pred, y_true, eps=1e-7):
    # y_pred: logits -> sigmoid
    y_pred = torch.sigmoid(y_pred)
    y_pred = (y_pred > 0.5).float()
    inter = (y_pred * y_true).sum(dim=(1,2,3))
    union = y_pred.sum(dim=(1,2,3)) + y_true.sum(dim=(1,2,3))
    dice = (2 * inter + eps) / (union + eps)
    return dice.mean().item()


def iou_score(y_pred, y_true, eps=1e-7):
    y_pred = torch.sigmoid(y_pred)
    y_pred = (y_pred > 0.5).float()
    inter = (y_pred * y_true).sum(dim=(1,2,3))
    union = y_pred.sum(dim=(1,2,3)) + y_true.sum(dim=(1,2,3)) - inter
    iou = (inter + eps) / (union + eps)
    return iou.mean().item()


def precision_recall(y_pred, y_true, eps=1e-7):
    y_pred = torch.sigmoid(y_pred)
    y_pred = (y_pred > 0.5).float()
    tp = (y_pred * y_true).sum(dim=(1,2,3))
    fp = (y_pred * (1 - y_true)).sum(dim=(1,2,3))
    fn = ((1 - y_pred) * y_true).sum(dim=(1,2,3))
    prec = (tp + eps) / (tp + fp + eps)
    rec = (tp + eps) / (tp + fn + eps)
    return prec.mean().item(), rec.mean().item()


class DiceBCELoss(nn.Module):
    def __init__(self, smooth=1.0):
        super().__init__()
        self.bce = nn.BCEWithLogitsLoss()
        self.smooth = smooth

    def forward(self, logits, targets):
        bce = self.bce(logits, targets)
        probs = torch.sigmoid(logits)
        num = (2 * (probs * targets).sum(dim=(1,2,3)) + self.smooth)
        den = (probs.sum(dim=(1,2,3)) + targets.sum(dim=(1,2,3)) + self.smooth)
        dice = 1 - (num / den).mean()
        return 0.5 * bce + 0.5 * dice


def train_one_channel(channel_path, device):
    ch_name = Path(channel_path).name
    images_dir = Path(channel_path) / "images"
    masks_dir  = Path(channel_path) / "masks"

    pairs = build_pairs(images_dir, masks_dir)
    if len(pairs) < 4:
        return {
            "channel": ch_name, "n_total": len(pairs), "n_train": 0, "n_val": 0,
            "train_loss": np.nan, "train_iou": np.nan, "train_dice": np.nan,
            "train_prec": np.nan, "train_rec": np.nan,
            "val_loss": np.nan, "val_iou": np.nan, "val_dice": np.nan,
            "val_prec": np.nan, "val_rec": np.nan, "note": "skipped: poucos pares"
        }

    dataset = SingleChannelSegDataset(pairs, img_size=IMG_SIZE)
    n_total = len(dataset)
    n_val = max(1, int(n_total * VAL_SPLIT))
    n_train = n_total - n_val
    gen = torch.Generator().manual_seed(SEED)
    train_ds, val_ds = random_split(dataset, [n_train, n_val], generator=gen)

    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True,
                              num_workers=NUM_WORKERS, pin_memory=True)
    val_loader   = DataLoader(val_ds, batch_size=BATCH_SIZE, shuffle=False,
                              num_workers=NUM_WORKERS, pin_memory=True)

    model = smp.Unet(
        encoder_name=ENCODER,
        encoder_weights=ENCODER_WEIGHTS,
        in_channels=1,
        classes=1,
        activation=ACTIVATION
    ).to(device)

    opt = torch.optim.Adam(model.parameters(), lr=LR)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=EPOCHS)
    criterion = DiceBCELoss()

    for epoch in range(1, EPOCHS + 1):
        model.train()
        tr_loss = 0.0
        for imgs, masks in train_loader:
            imgs = imgs.to(device, non_blocking=True).float()
            masks = masks.to(device, non_blocking=True).float()
            opt.zero_grad()
            logits = model(imgs)
            loss = criterion(logits, masks)
            loss.backward()
            opt.step()
            tr_loss += loss.item() * imgs.size(0)

        tr_loss = tr_loss / n_train
        sched.step()

        if epoch % 5 == 0 or epoch == EPOCHS:
            print(f"  Epoch {epoch}/{EPOCHS} - train_loss: {tr_loss:.4f}")


    model.eval()
    train_loss = 0.0
    train_ious, train_dices, train_precs, train_recs = [], [], [], []
    with torch.no_grad():

        for imgs, masks in train_loader:

            imgs = imgs.to(device, non_blocking=True).float()
            masks = masks.to(device, non_blocking=True).float()
            logits = model(imgs)
            loss = criterion(logits, masks)
            train_loss += loss.item() * imgs.size(0)
            train_ious.append(iou_score(logits, masks))
            train_dices.append(dice_coeff(logits, masks))
            p, r = precision_recall(logits, masks)
            train_precs.append(p); train_recs.append(r)

    train_loss /= n_train
    train_iou = float(np.mean(train_ious)) if train_ious else 0.0
    train_dice = float(np.mean(train_dices)) if train_dices else 0.0
    train_prec = float(np.mean(train_precs)) if train_precs else 0.0
    train_rec = float(np.mean(train_recs)) if train_recs else 0.0

    val_loss = 0.0
    val_ious, val_dices, val_precs, val_recs = [], [], [], []
    with torch.no_grad():
        for imgs, masks in val_loader:
            imgs = imgs.to(device, non_blocking=True).float()
            masks = masks.to(device, non_blocking=True).float()
            logits = model(imgs)
            loss = criterion(logits, masks)
            val_loss += loss.item() * imgs.size(0)
            val_ious.append(iou_score(logits, masks))
            val_dices.append(dice_coeff(logits, masks))
            p, r = precision_recall(logits, masks)
            val_precs.append(p); val_recs.append(r)

    val_loss /= n_val
    val_iou = float(np.mean(val_ious)) if val_ious else 0.0
    val_dice = float(np.mean(val_dices)) if val_dices else 0.0
    val_prec = float(np.mean(val_precs)) if val_precs else 0.0
    val_rec = float(np.mean(val_recs)) if val_recs else 0.0

    return {
        "channel": ch_name,
        "n_total": n_total,
        "n_train": n_train,
        "n_val": n_val,
        "train_loss": train_loss,
        "train_iou": train_iou,
        "train_dice": train_dice,
        "train_prec": train_prec,
        "train_rec": train_rec,
        "val_loss": val_loss,
        "val_iou": val_iou,
        "val_dice": val_dice,
        "val_prec": val_prec,
        "val_rec": val_rec,
        "note": ""
    }


def main():
    set_seed(SEED)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Usando device: {device}")

    channels = list_channels(ROOT)
    if not channels:
        print("Nenhum canal encontrado. Verifique a estrutura de pastas.")
        return

    print("Canais encontrados:")
    for c in channels:
        print(" -", c)

    results = []
    t0 = time.time()
    for ch in channels:
        ch_path = Path(ROOT) / ch
        print(f"\n{'='*60}")
        print(f"Treinando canal: {ch}")
        print(f"{'='*60}")
        res = train_one_channel(ch_path, device)
        print(f"\n[{ch}] RESULTADOS FINAIS (época {EPOCHS}):")
        print(f"  Train - Loss: {res['train_loss']:.4f} | IoU: {res['train_iou']:.4f} | "
              f"Dice: {res['train_dice']:.4f} | Prec: {res['train_prec']:.4f} | Rec: {res['train_rec']:.4f}")
        print(f"  Val   - Loss: {res['val_loss']:.4f} | IoU: {res['val_iou']:.4f} | "
              f"Dice: {res['val_dice']:.4f} | Prec: {res['val_prec']:.4f} | Rec: {res['val_rec']:.4f}")
        print(f"  Dataset: n_total={res['n_total']}, n_train={res['n_train']}, n_val={res['n_val']}")
        results.append(res)

    df = pd.DataFrame(results, columns=[
        "channel", "n_total", "n_train", "n_val",
        "train_loss", "train_iou", "train_dice", "train_prec", "train_rec",
        "val_loss", "val_iou", "val_dice", "val_prec", "val_rec", "note"
    ])

    df = df.sort_values(by="val_iou", ascending=False)
    df.to_csv(OUT_CSV, index=False, encoding="utf-8")
    
    print(f"\n{'='*60}")
    print(f"RESUMO FINAL")
    print(f"{'='*60}")
    print(f"\nTop 5 canais por IoU de validação:")
    print(df[["channel", "val_iou", "val_dice", "val_prec", "val_rec"]].head())
    print(f"\nTodos os resultados salvos em: {OUT_CSV}")
    print(f"Tempo total: {time.time() - t0:.1f}s")


if __name__ == "__main__":
    main()