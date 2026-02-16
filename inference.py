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

channels = ["dsDNA"]
# if channels is None:
#     # fallback (se não vier do checkpoint)
#     print("caiu aqui")

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
# remove prefixo module. se existir
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
        # heurística: pega o menor nome
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
        # multi-classe (não é seu caso pelo classes=1, mas deixado por segurança)
        prob = torch.softmax(out, dim=1)[0].numpy()  # (K,H,W)
        pred = prob.argmax(axis=0).astype(np.uint8)
        return prob, pred

def make_view_image(x_chw, mode="mean"):
    return x_chw.mean(axis=0) if mode == "mean" else x_chw.max(axis=0)

def mask_edges(mask: np.ndarray) -> np.ndarray:
    """Borda 1px sem dependências extras."""
    m = mask.astype(bool)
    # pixels que mudam em relação aos vizinhos (aprox contorno)
    e = np.zeros_like(m, dtype=bool)
    e |= m ^ np.roll(m,  1, axis=0)
    e |= m ^ np.roll(m, -1, axis=0)
    e |= m ^ np.roll(m,  1, axis=1)
    e |= m ^ np.roll(m, -1, axis=1)
    return e.astype(np.uint8)

# --- roda em uma amostra ---
x, used_files = load_stack_7ch(SAMPLE_FOLDER, channels)
print("Stack shape (C,H,W):", x.shape)
print("Arquivos usados (na ordem):")
for ch, fp in zip(channels, used_files):
    print(f"  {ch:12s} -> {os.path.basename(fp)}")

prob, pred = infer(model, x, thr=0.5)

base = make_view_image(x, mode="mean")
base_norm = base.astype(np.float32)
base_norm = (base_norm - base_norm.min()) / (base_norm.max() - base_norm.min() + 1e-8)

edges = mask_edges(pred)

# Overlay com preenchimento + contorno
alpha_fill = 0.25
alpha_edge = 0.95

plt.figure(figsize=(18, 5))

plt.subplot(1, 3, 1)
plt.imshow(base_norm, cmap="gray")
plt.title("Entrada (normalizada)")
plt.axis("off")

plt.subplot(1, 3, 2)
plt.imshow(base_norm, cmap="gray")

# preenchimento em verde
fill = np.zeros((*pred.shape, 4), dtype=np.float32)
fill[..., 1] = 1.0                 # G
fill[..., 3] = pred * alpha_fill   # alpha só onde pred=1
plt.imshow(fill)

# contorno em amarelo
edge_rgba = np.zeros((*pred.shape, 4), dtype=np.float32)
edge_rgba[..., 0] = 1.0            # R
edge_rgba[..., 1] = 1.0            # G
edge_rgba[..., 3] = edges * alpha_edge
plt.imshow(edge_rgba)

plt.title("Overlay: preenchimento (verde) + contorno (amarelo)")
plt.axis("off")

plt.subplot(1, 3, 3)
plt.imshow(prob, cmap="magma", vmin=0, vmax=1)
plt.title("Probabilidade (0–1)")
plt.axis("off")

plt.tight_layout()
plt.show()