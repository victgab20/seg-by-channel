import os
import glob
import numpy as np
import tifffile
import matplotlib.pyplot as plt



canais =[ "SegmentationInterior"]

img_path = r"C:\Users\victo\Downloads\Mestrado\TNBC\TNBCShareData\Point41"

def normalize01(x: np.ndarray) -> np.ndarray:
    x = x.astype(np.float32, copy=False)
    lo = np.nanpercentile(x, 1)   # robusto a outliers
    hi = np.nanpercentile(x, 99)
    if hi <= lo:
        return np.zeros_like(x, dtype=np.float32)
    x = (x - lo) / (hi - lo)
    return np.clip(x, 0, 1)

# 1) Lista todos os tiffs do diretório
tiffs = glob.glob(os.path.join(img_path, "*.tif")) + glob.glob(os.path.join(img_path, "*.tiff"))
tiffs_lower = [os.path.basename(p).lower() for p in tiffs]

# 2) Mapeia cada canal -> arquivo cujo nome contém o canal
canal_para_arquivo = {}
nao_encontrados = []
for c in canais:
    c_low = c.lower()
    matches = [tiffs[i] for i, name in enumerate(tiffs_lower) if c_low in name]
    if len(matches) == 0:
        nao_encontrados.append(c)
    elif len(matches) == 1:
        canal_para_arquivo[c] = matches[0]
    else:
        # Se houver mais de um, pega o menor nome (heurística simples)
        canal_para_arquivo[c] = sorted(matches, key=lambda p: len(os.path.basename(p)))[0]

if nao_encontrados:
    print("Canais não encontrados no diretório (por substring no nome do arquivo):")
    print(nao_encontrados)
    print("\nDica: confira como os arquivos estão nomeados em Point41.")
    # Você pode optar por abortar:
    # raise SystemExit

# 3) Carrega, normaliza e agrega (max projection)
agg = None
usados = []

for c, fp in canal_para_arquivo.items():
    img = tifffile.imread(fp)

    # Se vier com mais dimensões, tenta reduzir para 2D (primeira fatia)
    if img.ndim == 2:
        img2d = img
    elif img.ndim == 3:
        img2d = img[0]
    elif img.ndim == 4:
        img2d = img[0, 0]
    else:
        print(f"Pulando {c}: ndim inesperado {img.ndim}, shape={img.shape}")
        continue

    img01 = normalize01(img2d)

    if agg is None:
        agg = img01
    else:
        # max projection pixel-a-pixel
        agg = np.maximum(agg, img01)

    usados.append(c)

if agg is None:
    raise RuntimeError("Nenhuma imagem foi agregada. Verifique nomes de arquivos e o diretório.")

print(f"Agregado gerado com {len(usados)} canais.")

# 4) Plota a imagem única agregada
plt.figure(figsize=(7, 7))
plt.imshow(agg, cmap="gray")
plt.title(f"Imagem agregada (max) - {len(usados)} canais")
plt.axis("off")
plt.tight_layout()
plt.show()
