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
from torch.utils.data import Dataset, DataLoader

import albumentations as A
from albumentations.pytorch import ToTensorV2

import segmentation_models_pytorch as smp

# =========================
# Hiperparâmetros
# =========================
ROOT = r"C:\Users\victo\Downloads\Mestrado\TNBC\preproc_unet_per_channel"
OUT_DIR = "output_single_training_v1_all"  # Pasta para salvar modelo e resultados

IMG_SIZE = 512
VAL_SPLIT = 0.2
BATCH_SIZE = 4
EPOCHS = 25
LR = 1e-4
NUM_WORKERS = 2
SEED = 1337

ENCODER = "resnet34"
ENCODER_WEIGHTS = "imagenet"
ACTIVATION = None

IGNORE_DIRS = set(["Background", "Segmentation", "SegmentationInterior"])

# =========================
# CONFIGURE SEUS CANAIS AQUI
# =========================
CHANNELS_TO_USE = [
    "B7H3",
    "Beta_catenin",
    "C",
    "Ca",
    "CD11b",
    "CD11c",
    "CD138",
    "CD16",
    "CD163",
    "CD20",
    "CD209",
    "CD3",
    "CD31",
    "CD4",
    "CD45",
    "CD45RO",
    "CD56",
    "CD63",
    "CD68",
    "CD8",
    "CSF-1R",
    "dsDNA",
    "EGFR",
    "Fe",
    "FoxP3",
    "H3K27me3",
    "H3K9ac",
    "HLA-DR",
    "HLA_Class_1",
    "IDO",
    "Keratin17",
    "Keratin6",
    "Ki67",
    "Lag3",
    "MPO",
    "Na",
    "OX40",
    "P",
    "p53",
    "Pan-Keratin",
    "PD-L1",
    "PD1",
    "phospho-S6",
    "SMA",
    "Vimentin"
]
# ==============
# Utilidades
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

        img_resized = np.stack(resized_chans, axis=0)
        image = torch.from_numpy(img_resized).float()
        mask = torch.from_numpy(mask_np).unsqueeze(0).float()
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
# Funções auxiliares
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

def evaluate_model(model, loader, criterion, device):
    """Avalia o modelo em um dataloader."""
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
            precs.append(p)
            recs.append(r)
    
    n_samples = len(loader.dataset)
    return {
        'loss': tot_loss / n_samples,
        'iou': float(np.mean(ious)) if ious else 0.0,
        'dice': float(np.mean(dices)) if dices else 0.0,
        'precision': float(np.mean(precs)) if precs else 0.0,
        'recall': float(np.mean(recs)) if recs else 0.0
    }

# ===========================
# Função principal de treinamento
# ===========================
def train_model(root, channels, device, save_dir):
    """Treina um modelo único com os canais especificados."""
    
    # Cria diretório de saída
    os.makedirs(save_dir, exist_ok=True)
    
    # Encontra stems comuns
    print(f"\nBuscando amostras comuns aos {len(channels)} canais...")
    stems = get_common_stems_for_channels(root, channels)
    
    if len(stems) < 4:
        print("❌ Poucos pares em comum entre os canais. Verifique sua estrutura.")
        return None
    
    print(f"✓ Encontradas {len(stems)} amostras comuns")
    
    # Split treino/validação
    train_idx, val_idx = split_indices(stems, val_split=VAL_SPLIT, seed=SEED)
    train_stems = [stems[i] for i in train_idx]
    val_stems = [stems[i] for i in val_idx]
    
    print(f"✓ Split: {len(train_stems)} treino | {len(val_stems)} validação")
    
    # Datasets e loaders
    train_ds = MultiChannelSegDataset(root, channels, train_stems, img_size=IMG_SIZE)
    val_ds = MultiChannelSegDataset(root, channels, val_stems, img_size=IMG_SIZE)
    
    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True,
                              num_workers=NUM_WORKERS, pin_memory=True)
    val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE, shuffle=False,
                            num_workers=NUM_WORKERS, pin_memory=True)
    
    # Modelo
    print(f"\n🔧 Criando modelo U-Net com {len(channels)} canais de entrada...")
    model = smp.Unet(
        encoder_name=ENCODER,
        encoder_weights=ENCODER_WEIGHTS,
        in_channels=len(channels),
        classes=1,
        activation=ACTIVATION
    ).to(device)
    
    # Otimizador, scheduler e loss
    optimizer = torch.optim.Adam(model.parameters(), lr=LR)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOCHS)
    criterion = DiceBCELoss()
    
    # Histórico de treinamento
    history = {
        'train_loss': [], 'train_dice': [], 'train_iou': [],
        'val_loss': [], 'val_dice': [], 'val_iou': []
    }
    
    best_val_dice = 0.0
    best_epoch = 0
    
    print(f"\n🚀 Iniciando treinamento por {EPOCHS} épocas...\n")
    print("="*80)
    
    t0 = time.time()
    
    for epoch in range(1, EPOCHS + 1):
        # ============ TREINO ============
        model.train()
        train_loss = 0.0
        
        for imgs, masks in train_loader:
            imgs = imgs.to(device, non_blocking=True).float()
            masks = masks.to(device, non_blocking=True).float()
            
            optimizer.zero_grad()
            logits = model(imgs)
            loss = criterion(logits, masks)
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item() * imgs.size(0)
        
        train_loss /= len(train_ds)
        scheduler.step()
        
        # ============ VALIDAÇÃO ============
        train_metrics = evaluate_model(model, train_loader, criterion, device)
        val_metrics = evaluate_model(model, val_loader, criterion, device)
        
        # Salva histórico
        history['train_loss'].append(train_metrics['loss'])
        history['train_dice'].append(train_metrics['dice'])
        history['train_iou'].append(train_metrics['iou'])
        history['val_loss'].append(val_metrics['loss'])
        history['val_dice'].append(val_metrics['dice'])
        history['val_iou'].append(val_metrics['iou'])
        
        # Salva melhor modelo
        # if val_metrics['dice'] > best_val_dice:
        #     best_val_dice = val_metrics['dice']
        #     best_epoch = epoch
        #     torch.save({
        #         'epoch': epoch,
        #         'model_state_dict': model.state_dict(),
        #         'optimizer_state_dict': optimizer.state_dict(),
        #         'val_dice': best_val_dice,
        #         'channels': channels
        #     }, os.path.join(save_dir, 'best_model.pth'))
        
        # Log a cada 5 épocas ou na última
        if epoch % 5 == 0 or epoch == EPOCHS:
            print(f"Época {epoch:3d}/{EPOCHS} | "
                  f"Train - Loss: {train_metrics['loss']:.4f} Dice: {train_metrics['dice']:.4f} | "
                  f"Val - Loss: {val_metrics['loss']:.4f} Dice: {val_metrics['dice']:.4f} IoU: {val_metrics['iou']:.4f}")
    
    total_time = time.time() - t0
    
    print("="*80)
    print(f"\n✅ Treinamento concluído em {total_time:.1f}s ({total_time/60:.1f} min)")
    print(f"🏆 Melhor Dice de validação: {best_val_dice:.4f} (época {best_epoch})")
    
    # Salva histórico
    history_df = pd.DataFrame(history)
    history_df.to_csv(os.path.join(save_dir, 'training_history.csv'), index=False)
    
    # Salva resumo final
    summary = {
        'channels': '|'.join(channels),
        'num_channels': len(channels),
        'total_samples': len(stems),
        'train_samples': len(train_stems),
        'val_samples': len(val_stems),
        'epochs': EPOCHS,
        'best_epoch': best_epoch,
        'best_val_dice': best_val_dice,
        'final_train_dice': train_metrics['dice'],
        'final_train_iou': train_metrics['iou'],
        'final_train_precision': train_metrics['precision'],
        'final_train_recall': train_metrics['recall'],
        'final_val_dice': val_metrics['dice'],
        'final_val_iou': val_metrics['iou'],
        'final_val_precision': val_metrics['precision'],
        'final_val_recall': val_metrics['recall'],
        'training_time_seconds': total_time
    }
    
    summary_df = pd.DataFrame([summary])
    summary_df.to_csv(os.path.join(save_dir, 'training_summary.csv'), index=False)
    
    return model, history, summary

# ===========================
# MAIN
# ===========================
def main():
    set_seed(SEED)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    print("="*80)
    print("TREINAMENTO ÚNICO DE SEGMENTAÇÃO MULTI-CANAL")
    print("="*80)
    print(f"Device: {device}")
    print(f"Root: {ROOT}")
    print(f"Output: {OUT_DIR}")
    print(f"Canais selecionados ({len(CHANNELS_TO_USE)}):")
    for i, ch in enumerate(CHANNELS_TO_USE, 1):
        print(f"   {i}. {ch}")
    
    # Verifica se os canais existem
    all_channels = list_channels(ROOT)
    missing = [c for c in CHANNELS_TO_USE if c not in all_channels]
    
    if missing:
        print(f"\n❌ ERRO: Canais não encontrados: {missing}")
        print(f"📋 Canais disponíveis: {all_channels}")
        return
    
    # Treina
    model, history, summary = train_model(ROOT, CHANNELS_TO_USE, device, OUT_DIR)
    
    if model is not None:
        print(f"\n📊 Resultados salvos em: {OUT_DIR}/")
        print(f"   - training_history.csv (histórico por época)")
        print(f"   - training_summary.csv (resumo final)")
        
        print(f"\n📈 Métricas finais:")
        print(f"   Val Dice: {summary['final_val_dice']:.4f}")
        print(f"   Val IoU:  {summary['final_val_iou']:.4f}")
        print(f"   Val Prec: {summary['final_val_precision']:.4f}")
        print(f"   Val Rec:  {summary['final_val_recall']:.4f}")

if __name__ == "__main__":
    main()