import os, glob
import numpy as np
import tifffile
import torch
import matplotlib.pyplot as plt
import segmentation_models_pytorch as smp

# ========= AJUSTE =========
CKPT_PATH = r"output_single_training_default\best_model.pth"
SAMPLE_FOLDER = r"C:\Users\victo\Downloads\Mestrado\TNBC\TNBCShareData\Point41"

# Mesmos parâmetros do treino:
ENCODER = "resnet34"          # <- ajuste
ENCODER_WEIGHTS = "imagenet"  # <- ajuste (ou None)
ACTIVATION = None            # <- ajuste (no treino você passou ACTIVATION)
# ==========================

device = "cuda" if torch.cuda.is_available() else "cpu"

# --- carregamento seguro dos pesos (evita o warning e reduz risco de pickle) ---
ckpt = torch.load(CKPT_PATH, map_location="cpu", weights_only=True)

channels = ["dsDNA", "CD4", "Beta catenin", "HLA-DR", "H3K9ac", "Pan-Keratin", "H3K27me3"]

print("Checkpoint epoch:", ckpt.get("epoch"))
print("Checkpoint val_dice:", ckpt.get("val_dice"))
print("Channels no ckpt:", channels)

# --- instancia o mesmo modelo do treino ---
model = smp.Unet(
    encoder_name=ENCODER,
    encoder_weights=ENCODER_WEIGHTS,
    in_channels=len(channels),
    classes=1,
    activation=ACTIVATION,   # normalmente None para inferência (logits)
).to(device)
model.eval()

# --- carrega pesos ---
sd = ckpt["model_state_dict"]
sd = {k.replace("module.", ""): v for k, v in sd.items()}
missing, unexpected = model.load_state_dict(sd, strict=True)
print("missing:", missing)
print("unexpected:", unexpected)

def robust_norm01(x: np.ndarray) -> np.ndarray:
    x = x.astype(np.float32, copy=False)
    lo = np.percentile(x, 1)
    hi = np.percentile(x, 99)
    if hi <= lo:
        return np.zeros_like(x, dtype=np.float32)
    x = (x - lo) / (hi - lo)
    return np.clip(x, 0, 1)

def find_channel_file(folder, ch):
    tiffs = glob.glob(os.path.join(folder, "*.tif")) + glob.glob(os.path.join(folder, "*.tiff"))
    ch_low = ch.lower()
    matches = [p for p in tiffs if ch_low in os.path.basename(p).lower()]
    if len(matches) == 0:
        raise FileNotFoundError(f"Não achei arquivo para canal '{ch}' em: {folder}")
    if len(matches) > 1:
        matches = sorted(matches, key=lambda p: len(os.path.basename(p)))
    return matches[0]

def load_stack_7ch(folder, channels):
    imgs = []
    used_files = []
    for ch in channels:
        fp = find_channel_file(folder, ch)
        im = tifffile.imread(fp)

        # reduz para 2D se vier multi-dim
        if im.ndim == 2:
            im2d = im
        elif im.ndim == 3:
            im2d = im[0]
        elif im.ndim == 4:
            im2d = im[0, 0]
        else:
            raise ValueError(f"{ch}: ndim inesperado {im.ndim}, shape={im.shape}")

        imgs.append(robust_norm01(im2d))
        used_files.append(fp)

    x = np.stack(imgs, axis=0)  # (C,H,W)
    return x, used_files

def infer(model, x_chw, thr=0.5):
    x = torch.from_numpy(x_chw).unsqueeze(0).to(device)  # (1,C,H,W)
    with torch.no_grad():
        out = model(x)  # logits ou prob dependendo de activation
    out = out.detach().cpu()

    # se activation=None, precisa sigmoid
    if out.shape[1] == 1:
        logits = out[0, 0].numpy()
        prob = 1 / (1 + np.exp(-logits))
        pred = (prob >= thr).astype(np.uint8)
        return prob, pred
    else:
        prob = torch.softmax(out, dim=1)[0].numpy()  # (K,H,W)
        pred = prob.argmax(axis=0).astype(np.uint8)
        return prob, pred

# --- roda em uma amostra ---
x, used_files = load_stack_7ch(SAMPLE_FOLDER, channels)
print("Stack shape (C,H,W):", x.shape)
print("Arquivos usados (na ordem):")
for ch, fp in zip(channels, used_files):
    print(f"  {ch:12s} -> {os.path.basename(fp)}")

prob, pred = infer(model, x, thr=0.5)

# --- PLOT APENAS DA MÁSCARA PREDITA (SEM OVERLAY) ---
plt.figure(figsize=(8, 8))
plt.imshow(pred, cmap="gray", vmin=0, vmax=1)
plt.title("Máscara predita (binária)")
plt.axis("off")
plt.tight_layout()
plt.show()