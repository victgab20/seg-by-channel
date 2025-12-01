import os
import re
import csv
import glob
import time
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

# =========================
# Hiperparâmetros originais
# =========================
ROOT = r"C:\Users\victo\Downloads\Mestrado\TNBC\preproc_unet_per_channel"
OUT_CSV = "results_multichannel_incremental_default_encoder_none.csv"

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

# Ranking trazido (por Dice)

#TOP_BY_DICE = ["dsDNA","Ta","Au","P","H3K27me3","Ca","C","Lag3","OX40","CD11c"]

TOP_BY_DICE = [
    "dsDNA",
    "CD4",
    "Beta_catenin",
    "HLA-DR",
    "H3K9ac",
    "Pan-Keratin",
    "H3K27me3"
]



# ==============
# Utilidades base
# ==============
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

def find_images(images_dir):
    return sorted(glob.glob(str(Path(images_dir) / "*.tif"))) + \
           sorted(glob.glob(str(Path(images_dir) / "*.tiff")))

def build_map_images(images_dir):
    img_paths = find_images(images_dir)
    m = {}
    for ip in img_paths:
        stem = Path(ip).stem
        m[stem] = ip
    return m

def build_map_masks(masks_dir):
    mask_paths = sorted(glob.glob(str(Path(masks_dir) / "*.png")))
    m = {}
    for mp in mask_paths:
        mstem = Path(mp).stem
        if mstem.endswith("_mask"):
            stem = mstem[:-5]
        else:
            stem = mstem
        m[stem] = mp
    return m

def intersect_stems(dicts):
    """Interseção de chaves (stems) de vários dicionários."""
    keys = None
    for d in dicts:
        keys = set(d.keys()) if keys is None else (keys & set(d.keys()))
    return sorted(list(keys)) if keys is not None else []

class MultiChannelSegDataset(Dataset):

    def __init__(self, root, channel_names, stems, img_size=512):
        self.root = Path(root)
        self.channel_names = channel_names
        self.stems = stems
        self.img_size = img_size

        self.img_maps = []
        self.mask_maps = []
        for ch in self.channel_names:
            ch_path = self.root / ch
            self.img_maps.append(build_map_images(ch_path / "images"))
            self.mask_maps.append(build_map_masks(ch_path / "masks"))

        self.resize = A.Resize(img_size, img_size, interpolation=1)

    def __len__(self):
        return len(self.stems)

    def _load_img_norm01(self, path):
        with Image.open(path) as im:
            im = im.convert("I") if im.mode not in ("I;16", "I") else im
            arr = np.array(im, dtype=np.float32)
        if arr.max() > 0:
            arr = arr / arr.max()
        return arr

    def __getitem__(self, idx):
        stem = self.stems[idx]

        chans = []
        for ch_idx, ch in enumerate(self.channel_names):
            img_path = self.img_maps[ch_idx][stem]
            arr = self._load_img_norm01(img_path)
            chans.append(arr)
        img_stack = np.stack(chans, axis=0)  # [k, H, W]

        mask_path = self.mask_maps[0][stem]
        with Image.open(mask_path) as mm:
            mask_np = np.array(mm.convert("L"), dtype=np.uint8)
        mask_np = (mask_np > 0).astype(np.float32)

        resized_chans = []
        for c in range(img_stack.shape[0]):
            r = self.resize(image=img_stack[c], mask=mask_np)
            resized_chans.append(r["image"])
            mask_np = r["mask"]

        img_resized = np.stack(resized_chans, axis=0)  # [k, H, W]
        image = torch.from_numpy(img_resized).float()
        mask = torch.from_numpy(mask_np).unsqueeze(0).float()  # [1, H, W]
        return image, mask

# ================
# Métricas e perdas
# ================
def dice_coeff(y_pred, y_true, eps=1e-7):
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

# ===========================
# Treino/avaliação por top-k
# ===========================
def get_common_stems_for_channels(root, channel_names):

    img_maps = []
    mask_maps = []
    for ch in channel_names:
        ch_path = Path(root) / ch
        img_maps.append(build_map_images(ch_path / "images"))
        mask_maps.append(build_map_masks(ch_path / "masks"))
    common_img = intersect_stems(img_maps)
    common_msk = intersect_stems(mask_maps)
    stems = sorted(list(set(common_img) & set(common_msk)))
    return stems

def split_indices(stems, val_split=0.2, seed=SEED):
    rng = np.random.RandomState(seed)
    idxs = np.arange(len(stems))
    rng.shuffle(idxs)
    n_val = max(1, int(len(stems) * val_split))
    val_idx = idxs[:n_val]
    tr_idx = idxs[n_val:]
    return tr_idx.tolist(), val_idx.tolist()

def train_eval_for_k(root, channels_k, stems, train_idx, val_idx, device):
    # Dataset e loaders
    train_stems = [stems[i] for i in train_idx]
    val_stems   = [stems[i] for i in val_idx]

    train_ds = MultiChannelSegDataset(root, channels_k, train_stems, img_size=IMG_SIZE)
    val_ds   = MultiChannelSegDataset(root, channels_k, val_stems,   img_size=IMG_SIZE)

    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True,
                              num_workers=NUM_WORKERS, pin_memory=True)
    val_loader   = DataLoader(val_ds, batch_size=BATCH_SIZE, shuffle=False,
                              num_workers=NUM_WORKERS, pin_memory=True)

    # Modelo com in_channels = k
    model = smp.Unet(
        encoder_name=ENCODER,
        encoder_weights=ENCODER_WEIGHTS,
        in_channels=len(channels_k),
        classes=1,
        activation=ACTIVATION
    ).to(device)

    opt = torch.optim.Adam(model.parameters(), lr=LR)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=EPOCHS)
    criterion = DiceBCELoss()

    # Treino
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
        tr_loss /= len(train_ds)
        sched.step()
        if epoch % 5 == 0 or epoch == EPOCHS:
            print(f"[k={len(channels_k)}] Epoch {epoch}/{EPOCHS} - train_loss: {tr_loss:.4f}")

    # Avaliação (train metrics para diagnóstico + val para decisão)
    def eval_loader(loader, ds_len):
        model.eval()
        tot_loss = 0.0
        ious, dices, precs, recs = [], [], [], []
        with torch.no_grad():
            for imgs, masks in loader:
                imgs = imgs.to(device, non_blocking=True).float()
                masks = masks.to(device, non_blocking=True).float()
                logits = model(imgs)
                loss = criterion(logits, masks)
                tot_loss += loss.item() * imgs.size(0)
                ious.append(iou_score(logits, masks))
                dices.append(dice_coeff(logits, masks))
                p, r = precision_recall(logits, masks)
                precs.append(p); recs.append(r)
        tot_loss /= ds_len
        return (tot_loss,
                float(np.mean(ious)) if ious else 0.0,
                float(np.mean(dices)) if dices else 0.0,
                float(np.mean(precs)) if precs else 0.0,
                float(np.mean(recs)) if recs else 0.0)

    train_loss, train_iou, train_dice, train_prec, train_rec = eval_loader(train_loader, len(train_ds))
    val_loss,   val_iou,   val_dice,   val_prec,   val_rec   = eval_loader(val_loader,   len(val_ds))

    return {
        "k": len(channels_k),
        "channels": "|".join(channels_k),
        "n_total": len(stems),
        "n_train": len(train_ds),
        "n_val": len(val_ds),
        "train_loss": train_loss, "train_iou": train_iou, "train_dice": train_dice,
        "train_prec": train_prec, "train_rec": train_rec,
        "val_loss": val_loss, "val_iou": val_iou, "val_dice": val_dice,
        "val_prec": val_prec, "val_rec": val_rec
    }

def main():
    set_seed(SEED)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Usando device: {device}")

    # Confere existência dos canais do ranking
    all_chs = list_channels(ROOT)
    missing = [c for c in TOP_BY_DICE if c not in all_chs]
    if missing:
        print("ATENÇÃO: canais ausentes no ROOT:", missing)
    base_list = [c for c in TOP_BY_DICE if c in all_chs]
    if not base_list:
        print("Nenhum dos canais do ranking foi encontrado.")
        return

    stems_base = get_common_stems_for_channels(ROOT, base_list)
    if len(stems_base) < 4:
        print("Poucos pares em comum entre os canais. Verifique sua estrutura.")
        return

    train_idx, val_idx = split_indices(stems_base, val_split=VAL_SPLIT, seed=SEED)

    print(f"Total de amostras comuns aos canais do ranking: {len(stems_base)}")
    print(f"Split -> train: {len(train_idx)} | val: {len(val_idx)}")

    results = []
    t0 = time.time()
    for k in range(1, len(base_list) + 1):
        channels_k = base_list[:k]
        print("\n" + "="*70)
        print(f"Treinando configuração k={k}  Canais: {channels_k}")
        print("="*70)

        res = train_eval_for_k(ROOT, channels_k, stems_base, train_idx, val_idx, device)
        print(f"\n[k={k}] RESULTADOS (época {EPOCHS}):")
        print(f"  Train - Loss: {res['train_loss']:.4f} | IoU: {res['train_iou']:.4f} | "
              f"Dice: {res['train_dice']:.4f} | Prec: {res['train_prec']:.4f} | Rec: {res['train_rec']:.4f}")
        print(f"  Val   - Loss: {res['val_loss']:.4f} | IoU: {res['val_iou']:.4f} | "
              f"Dice: {res['val_dice']:.4f} | Prec: {res['val_prec']:.4f} | Rec: {res['val_rec']:.4f}")

        results.append(res)

    df = pd.DataFrame(results, columns=[
        "k","channels","n_total","n_train","n_val",
        "train_loss","train_iou","train_dice","train_prec","train_rec",
        "val_loss","val_iou","val_dice","val_prec","val_rec"
    ])

    # Ordene por Dice de validação (clássico para célula tumoral)
    df = df.sort_values(by="val_dice", ascending=False)
    df.to_csv(OUT_CSV, index=False, encoding="utf-8")

    print("\n" + "="*70)
    print("RESUMO FINAL (ordenado por val_dice)")
    print("="*70)
    print(df[["k","channels","val_dice","val_iou","val_prec","val_rec"]].head(10))
    print(f"\nTodos os resultados salvos em: {OUT_CSV}")
    print(f"Tempo total: {time.time() - t0:.1f}s")

if __name__ == "__main__":
    main()
