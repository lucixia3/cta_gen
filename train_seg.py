"""Segmentador de vasos de 36 clases, entrenado con el GT de TopAneu (58 CTA).

Su unico proposito es completar las etiquetas de TopCoW e ISLES (que solo traen
las 15 clases del CoW) para poder entrenar el generador con el arbol arterial
entero. No forma parte del modelo desplegado.

Uso: python train_seg.py [--epochs 400] [--patch 128] [--bs 2]
"""
import sys
import time
import argparse
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from monai.networks.nets import UNet
from monai.losses import DiceCELoss
from monai.inferers import sliding_window_inference

sys.path.insert(0, str(Path(__file__).parent))
import common as C

ROOT = Path(__file__).parent
PREP = ROOT / "prep" / "topaneu"
CKPT = ROOT / "ckpt"
DEV = "cuda"
NCLS = C.N_VESSEL + 1     # 0 fondo + 36 vasos


def load_all():
    fs = sorted(PREP.glob("*.npz"))
    data = []
    for f in fs:
        d = np.load(f)
        data.append((f.stem, d["img"].astype(np.float32), d["ves"]))
    return data


def sample_patch(img, ves, P, rng):
    """Parche PxPxP, 85% centrado en un voxel de vaso (son <1% del volumen)."""
    if rng.random() < 0.85:
        idx = np.array(np.nonzero(ves > 0))
        c = idx[:, rng.integers(idx.shape[1])]
    else:
        c = np.array([rng.integers(s) for s in img.shape])
    lo = np.clip(c - P // 2, 0, np.array(img.shape) - P)
    s = tuple(slice(l, l + P) for l in lo)
    return img[s], ves[s]


def augment(img, ves, rng):
    if rng.random() < 0.5:
        img, ves = C.flip_lr(img, ves)
    img = img * rng.normal(1.0, 0.06) + rng.normal(0.0, 0.05)
    if rng.random() < 0.3:
        img = img + rng.normal(0, rng.uniform(0.01, 0.05), img.shape).astype(np.float32)
    return img.astype(np.float32), ves


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--epochs", type=int, default=400)
    ap.add_argument("--patch", type=int, default=128)
    ap.add_argument("--bs", type=int, default=2)
    ap.add_argument("--iters", type=int, default=40, help="iteraciones por epoca")
    ap.add_argument("--lr", type=float, default=1e-3)
    a = ap.parse_args()
    CKPT.mkdir(exist_ok=True)

    data = load_all()
    rng = np.random.default_rng(0)
    val_ids = {d[0] for d in data[::8]}          # 1 de cada 8 -> ~7 casos
    tr = [d for d in data if d[0] not in val_ids]
    va = [d for d in data if d[0] in val_ids]
    print(f"train {len(tr)} | val {len(va)}", flush=True)

    net = UNet(spatial_dims=3, in_channels=1, out_channels=NCLS,
               channels=(32, 64, 128, 256, 320), strides=(2, 2, 2, 2),
               num_res_units=2, dropout=0.0).to(DEV)
    loss_fn = DiceCELoss(to_onehot_y=True, softmax=True, include_background=False,
                         lambda_dice=1.0, lambda_ce=1.0)
    opt = torch.optim.AdamW(net.parameters(), lr=a.lr, weight_decay=1e-5)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, a.epochs)
    scaler = torch.amp.GradScaler()

    best = -1.0
    for ep in range(a.epochs):
        net.train()
        tl, t0 = 0.0, time.time()
        for _ in range(a.iters):
            xs, ys = [], []
            for _ in range(a.bs):
                _, img, ves = tr[rng.integers(len(tr))]
                pi, pv = sample_patch(img, ves, a.patch, rng)
                pi, pv = augment(pi, pv, rng)
                xs.append(pi[None])
                ys.append(pv[None])
            x = torch.from_numpy(np.stack(xs)).to(DEV)
            y = torch.from_numpy(np.stack(ys).astype(np.int64)).to(DEV)
            with torch.autocast("cuda", dtype=torch.bfloat16):
                loss = loss_fn(net(x), y)
            opt.zero_grad(set_to_none=True)
            scaler.scale(loss).backward()
            scaler.step(opt)
            scaler.update()
            tl += loss.item()
        sched.step()

        if (ep + 1) % 10 == 0 or ep == a.epochs - 1:
            net.eval()
            dices = []
            with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
                for _, img, ves in va:
                    x = torch.from_numpy(img[None, None]).to(DEV)
                    lo = sliding_window_inference(x, (a.patch,) * 3, 2, net, overlap=0.5)
                    pr = lo.argmax(1)[0].cpu().numpy().astype(np.uint8)
                    per = []
                    for k in range(1, NCLS):
                        g, p = ves == k, pr == k
                        if g.sum() < 20:
                            continue          # clase ausente en este caso
                        per.append(2 * (g & p).sum() / (g.sum() + p.sum() + 1e-6))
                    dices.append(np.mean(per))
            d = float(np.mean(dices))
            print(f"ep {ep+1:4d} loss {tl/a.iters:.4f} valDice {d:.4f} "
                  f"({time.time()-t0:.0f}s)", flush=True)
            if d > best:
                best = d
                torch.save({"model": net.state_dict(), "dice": d, "epoch": ep,
                            "ncls": NCLS, "patch": a.patch}, CKPT / "seg_best.pt")
                print(f"   -> guardado seg_best.pt (Dice {d:.4f})", flush=True)
        else:
            print(f"ep {ep+1:4d} loss {tl/a.iters:.4f} ({time.time()-t0:.0f}s)", flush=True)

    print(f"\nMejor Dice medio (clases presentes): {best:.4f}")


if __name__ == "__main__":
    main()
