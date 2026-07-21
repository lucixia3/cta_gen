"""Entrena un segmentador 2D de infarto en NCCT -- el motor del filtro de cribado.

Datos (ya en disco, reales de ISLES'24): ISLES_train_package/spade2d_native/{train,val}/*.npz
    img : (X, Y, Z) float16 en [-1,1]  = NCCT, ventana [0,80] HU -> [-1,1], 256 in-plane
    lab : (X, Y, Z) uint8 0..7         = CTseg (clase 7 = LESION)

Modelo: BasicUNet 2D (1->1), DiceCE con sigmoide. Se entrena por CORTES axiales.

Clave para un FILTRO fiable: se sobre-muestrean los cortes con lesion (recall) PERO se
incluyen tantos cortes sanos como positivos (precision). Lo que hunde un cribado no es no
ver un infarto, es marcar de infarto un cerebro normal -- por eso el negativo pesa igual.
Como todos los pacientes de ISLES tienen lesion, los "negativos" son cortes sanos por
encima/debajo del infarto: ensenan "tejido normal -> sin lesion", justo lo que protege a
los controles sanos de un falso positivo.

Salida: cta_gen/ckpt/infarct_seg2d.pt  {model, epoch, val_dice, window=(0,80), inplane=256}

Uso:  python train_infarct_seg.py [--epochs 60] [--batch 24]
"""
import os
import sys
import glob
import argparse
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from monai.networks.nets import BasicUNet
from monai.losses import DiceCELoss

ROOT = Path(__file__).parent
ISLES = Path(r"C:\Desactivar_Respaldo\ISLES_train_package\spade2d_native")
CKPT = ROOT / "ckpt" / "infarct_seg2d.pt"
LESION = 7
WINDOW = (0.0, 80.0)
INPLANE = 256
DEV = "cuda" if torch.cuda.is_available() else "cpu"


def load_split(split, min_les=50):
    """Carga los volumenes de un split a RAM y lista sus cortes (cid, z, positivo?)."""
    vols, pos, neg = {}, [], []
    for f in sorted(glob.glob(str(ISLES / split / "*.npz"))):
        cid = Path(f).stem
        d = np.load(f)
        img = d["img"].astype(np.float16)          # (X,Y,Z) [-1,1]
        les = (d["lab"] == LESION)                 # (X,Y,Z) bool
        vols[cid] = (img, les)
        for z in range(img.shape[2]):
            (pos if les[:, :, z].sum() >= min_les else neg).append((cid, z))
    return vols, pos, neg


def batch_slices(vols, items, aug=False, rng=None):
    """(cid,z) -> tensores (N,1,256,256) img y mascara, con aumentado opcional."""
    xs, ys = [], []
    for cid, z in items:
        img, les = vols[cid]
        x = img[:, :, z].astype(np.float32)
        y = les[:, :, z].astype(np.float32)
        if aug:
            if rng.random() < 0.5:                 # espejo L-R (el infarto puede ir a cualquier lado)
                x, y = x[::-1].copy(), y[::-1].copy()
            if rng.random() < 0.7:                 # jitter de intensidad (robustez a otro escaner)
                x = np.clip(x * rng.uniform(0.9, 1.1) + rng.uniform(-0.05, 0.05), -1, 1)
            if rng.random() < 0.3:
                x = np.clip(x + rng.normal(0, 0.03, x.shape).astype(np.float32), -1, 1)
        xs.append(x); ys.append(y)
    X = torch.from_numpy(np.stack(xs))[:, None].to(DEV)
    Y = torch.from_numpy(np.stack(ys))[:, None].to(DEV)
    return X, Y


@torch.no_grad()
def evaluate(net, vols, pos):
    """Dice de lesion sobre los cortes con lesion del val (recall/solape)."""
    net.eval()
    inter = union = 0.0
    for i in range(0, len(pos), 32):
        X, Y = batch_slices(vols, pos[i:i + 32])
        p = (torch.sigmoid(net(X)) > 0.5).float()
        inter += (p * Y).sum().item()
        union += (p.sum() + Y.sum()).item()
    return 2 * inter / (union + 1e-6)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--epochs", type=int, default=60)
    ap.add_argument("--batch", type=int, default=24)
    ap.add_argument("--lr", type=float, default=1e-3)
    a = ap.parse_args()

    print(f"dispositivo: {DEV}", flush=True)
    tr_vols, tr_pos, tr_neg = load_split("train")
    va_vols, va_pos, _ = load_split("val")
    print(f"train: {len(tr_vols)} vols | pos {len(tr_pos)} neg {len(tr_neg)} | "
          f"val: {len(va_vols)} vols, {len(va_pos)} cortes con lesion", flush=True)

    net = BasicUNet(spatial_dims=2, in_channels=1, out_channels=1,
                    features=(32, 32, 64, 128, 256, 32)).to(DEV)
    loss_fn = DiceCELoss(sigmoid=True)
    opt = torch.optim.Adam(net.parameters(), lr=a.lr)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, a.epochs)
    rng = np.random.default_rng(0)

    CKPT.parent.mkdir(exist_ok=True)
    best = -1.0
    for ep in range(a.epochs):
        net.train()
        # epoca balanceada: todos los positivos + igual numero de negativos al azar
        neg = [tr_neg[i] for i in rng.permutation(len(tr_neg))[:len(tr_pos)]]
        items = tr_pos + neg
        items = [items[i] for i in rng.permutation(len(items))]

        tot = 0.0
        for i in range(0, len(items), a.batch):
            X, Y = batch_slices(tr_vols, items[i:i + a.batch], aug=True, rng=rng)
            opt.zero_grad()
            l = loss_fn(net(X), Y)
            l.backward(); opt.step()
            tot += l.item()
        sched.step()

        dice = evaluate(net, va_vols, va_pos)
        flag = ""
        if dice > best:
            best = dice
            torch.save({"model": net.state_dict(), "epoch": ep, "val_dice": dice,
                        "window": WINDOW, "inplane": INPLANE,
                        "features": (32, 32, 64, 128, 256, 32)}, CKPT)
            flag = "  <- best (guardado)"
        print(f"[{ep+1:2d}/{a.epochs}] loss {tot/(len(items)//a.batch+1):.4f} | "
              f"val lesion Dice {dice:.4f}{flag}", flush=True)

    print(f"\nlisto. mejor val Dice {best:.4f} -> {CKPT}", flush=True)


if __name__ == "__main__":
    main()
