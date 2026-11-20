"""
Cross-validation por paciente (5 folds) + ranking dos 48 canais.

Versão MAC (Apple Silicon / MPS).
Resultados isolados em results_cv_patient_windows/ para não se misturarem
com os resultados dos outros experimentos.

Uso:
    python cv_patient_ranking_mps.py

Saídas (em OUT_DIR):
    - ranking_per_channel.csv : ranking final (média ± desvio dos folds, por canal)
    - fold_details.csv        : métricas brutas de cada canal × fold
    - checkpoints/best_overall_model.pth : pesos do CAMPEÃO global (melhor fold de todos
      os canais; formato compatível com inference.py/show_pred.py: model_state_dict,
      epoch, val_dice — campos 'channel' e 'fold' indicam a origem)

Retomada: canais com todos os folds presentes em fold_details.csv são pulados.
"""

import os
import re
import glob
import random
import time
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
ROOT = "/Users/victorgabriel/Downloads/Mestrado/TNBC/preproc_unet_per_channel"
OUT_DIR = "results_cv_patient_mps"

N_FOLDS = 5
EPOCHS = 10
IMG_SIZE = 512
BATCH_SIZE = 4
LR = 1e-4
NUM_WORKERS = 2
SEED = 1337

ENCODER = "resnet34"
ENCODER_WEIGHTS = "imagenet"
ACTIVATION = None

IGNORE_DIRS = set(["Background", "Segmentation", "SegmentationInterior"])

RANKING_CSV = os.path.join(OUT_DIR, "ranking_per_channel.csv")
FOLDS_CSV = os.path.join(OUT_DIR, "fold_details.csv")
CKPT_DIR = os.path.join(OUT_DIR, "checkpoints")
CHAMPION_CKPT = os.path.join(CKPT_DIR, "best_overall_model.pth")
TMP_CKPT = os.path.join(CKPT_DIR, "_tmp_best.pth")

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
    return {Path(ip).stem: ip for ip in img_paths}

def build_map_masks(masks_dir):
    mask_paths = sorted(glob.glob(str(Path(masks_dir) / "*.png")))
    m = {}
    for mp in mask_paths:
        mstem = Path(mp).stem
        stem = mstem[:-5] if mstem.endswith("_mask") else mstem
        m[stem] = mp
    return m

def intersect_stems(dicts):
    keys = None
    for d in dicts:
        keys = set(d.keys()) if keys is None else (keys & set(d.keys()))
    return sorted(list(keys)) if keys is not None else []

def patient_id_from_stem(stem):
    """Extrai o ID do paciente do stem. Ex.: 'Point41' -> '41'."""
    match = re.match(r"Point(\d+)$", stem)
    if match is None:
        raise ValueError(
            f"Stem '{stem}' não segue o padrão esperado 'Point<numero>'. "
            f"Confira a nomeação dos arquivos em {ROOT}."
        )
    return match.group(1)

def split_folds_by_patient(stems, n_folds, seed=SEED):
    """Divide os stems em n_folds folds agrupados por paciente.

    Mesmo paciente nunca fica simultaneamente em treino e validação.
    Retorna lista de (train_stems, val_stems).
    """
    patients = sorted({patient_id_from_stem(s) for s in stems})
    if len(patients) < n_folds:
        raise ValueError(
            f"Pacientes insuficientes ({len(patients)}) para {n_folds} folds."
        )
    rng = np.random.RandomState(seed)
    rng.shuffle(patients)
    folds_patients = np.array_split(np.array(patients), n_folds)

    by_patient = {}
    for s in stems:
        by_patient.setdefault(patient_id_from_stem(s), []).append(s)

    folds = []
    for fold_patients in folds_patients:
        val_set = set(fold_patients.tolist())
        val_stems = [s for p in val_set for s in by_patient[p]]
        train_stems = [s for p in patients if p not in val_set for s in by_patient[p]]
        folds.append((sorted(train_stems), sorted(val_stems)))
    return folds

# ==============
# Dataset (1 canal)
# ==============
class SingleChannelSegDataset(Dataset):

    def __init__(self, root, channel_name, stems, img_size=512):
        self.root = Path(root)
        self.channel_name = channel_name
        self.stems = stems
        self.img_size = img_size

        ch_path = self.root / channel_name
        self.img_map = build_map_images(ch_path / "images")
        self.mask_map = build_map_masks(ch_path / "masks")

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

        img = self._load_img_norm01(self.img_map[stem])

        with Image.open(self.mask_map[stem]) as mm:
            mask_np = np.array(mm.convert("L"), dtype=np.uint8)
        mask_np = (mask_np > 0).astype(np.float32)

        r = self.resize(image=img, mask=mask_np)
        image = torch.from_numpy(r["image"]).float().unsqueeze(0)  # [1, H, W]
        mask = torch.from_numpy(r["mask"]).unsqueeze(0).float()    # [1, H, W]
        return image, mask

# ================
# Métricas e perdas
# ================
def dice_coeff(y_pred, y_true, eps=1e-7):
    y_pred = (torch.sigmoid(y_pred) > 0.5).float()
    inter = (y_pred * y_true).sum(dim=(1, 2, 3))
    union = y_pred.sum(dim=(1, 2, 3)) + y_true.sum(dim=(1, 2, 3))
    dice = (2 * inter + eps) / (union + eps)
    return dice.mean().item()

def iou_score(y_pred, y_true, eps=1e-7):
    y_pred = (torch.sigmoid(y_pred) > 0.5).float()
    inter = (y_pred * y_true).sum(dim=(1, 2, 3))
    union = y_pred.sum(dim=(1, 2, 3)) + y_true.sum(dim=(1, 2, 3)) - inter
    iou = (inter + eps) / (union + eps)
    return iou.mean().item()

def precision_recall(y_pred, y_true, eps=1e-7):
    y_pred = (torch.sigmoid(y_pred) > 0.5).float()
    tp = (y_pred * y_true).sum(dim=(1, 2, 3))
    fp = (y_pred * (1 - y_true)).sum(dim=(1, 2, 3))
    fn = ((1 - y_pred) * y_true).sum(dim=(1, 2, 3))
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
        num = (2 * (probs * targets).sum(dim=(1, 2, 3)) + self.smooth)
        den = (probs.sum(dim=(1, 2, 3)) + targets.sum(dim=(1, 2, 3)) + self.smooth)
        dice = 1 - (num / den).mean()
        return 0.5 * bce + 0.5 * dice

# ===========================
# Treino/avaliação de um fold
# ===========================
@torch.no_grad()
def evaluate(model, loader, criterion, device):
    model.eval()
    tot_loss = 0.0
    ious, dices, precs, recs = [], [], [], []
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
    n_batches = max(len(loader), 1)
    return {
        "val_loss": tot_loss / n_batches,
        "val_iou": float(np.mean(ious)) if ious else 0.0,
        "val_dice": float(np.mean(dices)) if dices else 0.0,
        "val_prec": float(np.mean(precs)) if precs else 0.0,
        "val_rec": float(np.mean(recs)) if recs else 0.0,
    }

def train_eval_one_fold(root, channel, fold_idx, train_stems, val_stems, device):
    train_ds = SingleChannelSegDataset(root, channel, train_stems, img_size=IMG_SIZE)
    val_ds = SingleChannelSegDataset(root, channel, val_stems, img_size=IMG_SIZE)

    pin_memory = (device.type == "cuda")
    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True,
                              num_workers=NUM_WORKERS, pin_memory=pin_memory)
    val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE, shuffle=False,
                            num_workers=NUM_WORKERS, pin_memory=pin_memory)

    model = smp.Unet(
        encoder_name=ENCODER,
        encoder_weights=ENCODER_WEIGHTS,
        in_channels=1,
        classes=1,
        activation=ACTIVATION
    ).to(device)

    opt = torch.optim.Adam(model.parameters(), lr=LR)
    criterion = DiceBCELoss()

    os.makedirs(CKPT_DIR, exist_ok=True)
    best = {"val_dice": -1.0, "best_epoch": 0, "metrics": None}

    for epoch in range(1, EPOCHS + 1):
        model.train()
        tr_loss = 0.0
        for imgs, masks in train_loader:
            imgs = imgs.to(device, non_blocking=True).float()
            masks = masks.to(device, non_blocking=True).float()
            opt.zero_grad()
            loss = criterion(model(imgs), masks)
            loss.backward()
            opt.step()
            tr_loss += loss.item() * imgs.size(0)
        tr_loss /= len(train_ds)

        metrics = evaluate(model, val_loader, criterion, device)
        improved = metrics["val_dice"] > best["val_dice"]
        if improved:
            best = {"val_dice": metrics["val_dice"],
                    "best_epoch": epoch,
                    "metrics": metrics}
            # Salva no temporário; o main() promove a campeão se o fold bater o recorde
            # Formato compatível com inference.py/show_pred.py (weights_only=True)
            torch.save({
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "val_dice": metrics["val_dice"],
                "val_iou": metrics["val_iou"],
                "val_prec": metrics["val_prec"],
                "val_rec": metrics["val_rec"],
                "val_loss": metrics["val_loss"],
                "channels": [channel],
                "channel": channel,
                "fold": fold_idx,
                "best_epoch": epoch,
            }, TMP_CKPT)
        if epoch % 5 == 0 or epoch == EPOCHS or improved:
            print(f"    [fold {fold_idx}] epoch {epoch}/{EPOCHS} - "
                  f"train_loss: {tr_loss:.4f} | val_dice: {metrics['val_dice']:.4f}"
                  f"{' *best*' if improved else ''}")

    metrics = best["metrics"]
    return {
        "channel": channel,
        "fold": fold_idx,
        "n_train": len(train_ds),
        "n_val": len(val_ds),
        "patient_ids_val": "|".join(sorted({patient_id_from_stem(s) for s in val_stems})),
        "best_epoch": best["best_epoch"],
        "val_loss": metrics["val_loss"],
        "val_dice": metrics["val_dice"],
        "val_iou": metrics["val_iou"],
        "val_prec": metrics["val_prec"],
        "val_rec": metrics["val_rec"],
    }

# ===========================
# CSVs
# ===========================
def load_done_channels():
    """Canais com todos os N_FOLDS folds registrados no fold_details.csv."""
    if not os.path.exists(FOLDS_CSV):
        return set()
    df = pd.read_csv(FOLDS_CSV)
    counts = df.groupby("channel")["fold"].nunique()
    return set(counts[counts >= N_FOLDS].index)

def append_fold_row(row):
    new_df = pd.DataFrame([row])
    if os.path.exists(FOLDS_CSV):
        old_df = pd.read_csv(FOLDS_CSV)
        # Remove folds anteriores deste canal (canal parcial é refeito do zero)
        old_df = old_df[old_df["channel"] != row["channel"]]
        df = pd.concat([old_df, new_df], ignore_index=True)
    else:
        df = new_df
    df.to_csv(FOLDS_CSV, index=False, encoding="utf-8")

def load_champion_dice():
    """Lê o campeão existente para conhecer o recorde atual (retomada)."""
    if not os.path.exists(CHAMPION_CKPT):
        return -1.0
    ckpt = torch.load(CHAMPION_CKPT, map_location="cpu", weights_only=True)
    dice = float(ckpt.get("val_dice", -1.0))
    print(f"Campeão atual: {ckpt.get('channel')} fold {ckpt.get('fold')} "
          f"(epoch {ckpt.get('best_epoch')}) val_dice: {dice:.4f}")
    return dice

def rebuild_ranking_csv():
    if not os.path.exists(FOLDS_CSV):
        return
    df = pd.read_csv(FOLDS_CSV)
    agg = df.groupby("channel").agg(
        mean_val_dice=("val_dice", "mean"),
        std_val_dice=("val_dice", "std"),
        mean_val_iou=("val_iou", "mean"),
        std_val_iou=("val_iou", "std"),
        mean_val_prec=("val_prec", "mean"),
        mean_val_rec=("val_rec", "mean"),
        n_folds_ok=("fold", "nunique"),
    ).reset_index()
    fold_agg = (
        df.sort_values(["channel", "fold"])
          .groupby("channel")
          .apply(lambda g: pd.Series({
              "fold_dices": "|".join(f"{v:.4f}" for v in g["val_dice"]),
              "fold_ious": "|".join(f"{v:.4f}" for v in g["val_iou"]),
          }))
          .reset_index()
    )
    agg = agg.merge(fold_agg, on="channel")
    agg = agg.sort_values(by="mean_val_dice", ascending=False)
    agg.to_csv(RANKING_CSV, index=False, encoding="utf-8")

# ================
# Main
# ================
def main():
    set_seed(SEED)
    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    print(f"Usando device: {device}")

    Path(OUT_DIR).mkdir(parents=True, exist_ok=True)

    global_best_dice = load_champion_dice()

    channels = list_channels(ROOT)
    print(f"Canais encontrados: {len(channels)}")
    if not channels:
        print("Nenhum canal encontrado. Verifique ROOT.")
        return

    done = load_done_channels()
    if done:
        print(f"Pulando {len(done)} canal(is) já concluídos: {sorted(done)}")

    t0 = time.time()
    total_trainings = len(channels) * N_FOLDS

    for ch_idx, channel in enumerate(channels, start=1):
        if channel in done:
            continue

        img_map = build_map_images(Path(ROOT) / channel / "images")
        mask_map = build_map_masks(Path(ROOT) / channel / "masks")
        stems = intersect_stems([img_map, mask_map])
        if len(stems) < N_FOLDS * 2:
            print(f"[{channel}] Apenas {len(stems)} amostras; pulando.")
            continue

        folds = split_folds_by_patient(stems, N_FOLDS, seed=SEED)
        print("\n" + "=" * 70)
        print(f"[{ch_idx}/{len(channels)}] Canal: {channel} | "
              f"{len(stems)} amostras | {N_FOLDS} folds (pacientes agrupados)")
        print("=" * 70)

        for fold_idx, (train_stems, val_stems) in enumerate(folds, start=1):
            print(f"  Fold {fold_idx}/{N_FOLDS}: "
                  f"train={len(train_stems)} val={len(val_stems)} "
                  f"(pacientes val: {sorted({patient_id_from_stem(s) for s in val_stems})})")
            row = train_eval_one_fold(ROOT, channel, fold_idx,
                                      train_stems, val_stems, device)
            print(f"  -> fold {fold_idx} concluído | "
                  f"val_dice: {row['val_dice']:.4f} (best epoch {row['best_epoch']})")
            append_fold_row(row)

            # Promove o melhor epoch do fold a campeão global, ou descarta o temporário
            if row["val_dice"] > global_best_dice:
                global_best_dice = row["val_dice"]
                os.replace(TMP_CKPT, CHAMPION_CKPT)
                print(f"  -> NOVO CAMPEÃO GLOBAL: {channel} fold {fold_idx} "
                      f"val_dice: {row['val_dice']:.4f}")
            elif os.path.exists(TMP_CKPT):
                os.remove(TMP_CKPT)

        rebuild_ranking_csv()
        print(f"[{channel}] Ranking atualizado: {RANKING_CSV}")

    rebuild_ranking_csv()

    print("\n" + "=" * 70)
    print("RANKING FINAL (top 10 por mean_val_dice)")
    print("=" * 70)
    df = pd.read_csv(RANKING_CSV)
    print(df[["channel", "mean_val_dice", "std_val_dice", "n_folds_ok"]].head(10).to_string(index=False))
    print(f"\nTempo total: {(time.time() - t0) / 3600:.2f}h "
          f"({total_trainings} treinos de {EPOCHS} épocas)")
    print(f"Ranking: {RANKING_CSV}")
    print(f"Folds:   {FOLDS_CSV}")
    print(f"Checkpoint campeão: {CHAMPION_CKPT}")

if __name__ == "__main__":
    main()
