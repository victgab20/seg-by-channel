import os, re, glob
import numpy as np
import torch
from torch.utils.data import Dataset
from tifffile import imread
from PIL import Image

def natural_key(s: str):
    return [int(t) if t.isdigit() else t for t in re.split(r'(\d+)', s)]

def list_pairs(channel_dir):
    img_dir = os.path.join(channel_dir, "images")
    msk_dir = os.path.join(channel_dir, "masks")
    img_paths = sorted(glob.glob(os.path.join(img_dir, "*.tif")), key=natural_key)
    pairs = []
    for ip in img_paths:
        base = os.path.splitext(os.path.basename(ip))[0]  # "PointX"
        mp = os.path.join(msk_dir, f"{base}_mask.png")
        if os.path.exists(mp):
            pairs.append((ip, mp))
    return pairs

def percentile_norm(x, p_low=1.0, p_high=99.0, eps=1e-6):
    lo, hi = np.percentile(x, [p_low, p_high])
    x = (x - lo) / max(hi - lo, eps)
    return np.clip(x, 0.0, 1.0)

class TNBCChannelDataset(Dataset):
    def __init__(self, channel_dir, split="train", val_ratio=0.2, img_size=512, seed=42):
        """
        channel_dir: .../<CANAL>/{images,masks}
        split: 'train' ou 'val'
        val_ratio: fração para validação
        img_size: lado do crop central (int) ou None para usar a imagem inteira
        """
        super().__init__()
        all_pairs = list_pairs(channel_dir)
        if not all_pairs:
            raise RuntimeError(f"Nenhum par imagem/máscara em {channel_dir}")

        rng = np.random.RandomState(seed)
        idx = np.arange(len(all_pairs))
        rng.shuffle(idx)
        n_val = int(len(all_pairs) * val_ratio)
        if split == "train":
            sel = idx[n_val:]
        else:
            sel = idx[:n_val]
        self.samples = [all_pairs[i] for i in sel]

        self.img_size = img_size

    def __len__(self): 
        return len(self.samples)

    def _center_crop(self, img: np.ndarray, s: int) -> np.ndarray:
        H, W = img.shape
        if H < s or W < s:
            img_u8 = (img * 255.0).astype(np.uint8)
            img = np.array(Image.fromarray(img_u8).resize((s, s), Image.BILINEAR)) / 255.0
            return img.astype(np.float32)
        top = (H - s) // 2
        left = (W - s) // 2
        return img[top:top+s, left:left+s]

    def __getitem__(self, idx):
        ip, mp = self.samples[idx]

        img = imread(ip).astype(np.float32)
        img = percentile_norm(img)

        msk = np.array(Image.open(mp))
        msk = (msk > 0).astype(np.float32)

        if self.img_size is not None:
            s = int(self.img_size)
            img = self._center_crop(img, s)
            msk = self._center_crop(msk, s)

        img = np.ascontiguousarray(img, dtype=np.float32)
        msk = np.ascontiguousarray(msk, dtype=np.float32)

        img = torch.from_numpy(img).unsqueeze(0)
        msk = torch.from_numpy(msk).unsqueeze(0)

        return img, msk
