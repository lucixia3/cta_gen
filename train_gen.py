"""Entrena el generador unico: mapa semantico -> CTA.

Un solo modelo para todo. La variante de CoW y el sitio de oclusion no son
entradas del modelo: son ediciones del mapa semantico que entra por SPADE. Por
eso generaliza a combinaciones variante x oclusion que no existen en los datos.

Datos (209 casos): TopAneu 58 (arbol completo, GT), TopCoW 125 (pseudo), ISLES 26
(CTA con oclusion real -> el modelo aprende como se ve de verdad un cut-off).

Uso: python train_gen.py [--epochs 800] [--patch 96] [--bs 6]
"""
import sys
import time
import argparse
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).parent))
import common as C
import model as M

ROOT = Path(__file__).parent
DEV = "cuda"


def load_all():
    data = []
    for ds in ("topaneu", "topcow", "isles"):
        for f in sorted((ROOT / "sem" / ds).glob("*.npz")):
            s = np.load(f)
            p = np.load(ROOT / "prep" / ds / f.name)
            data.append((f.stem, ds, p["img"].astype(np.float32), s["sem"]))
    return data


def sample_patch(img, sem, P, rng):
    """70% de los parches centrados en un vaso: son <1% del volumen pero son
    lo unico que importa aqui."""
    ves = (sem >= C.VESSEL0) & (sem <= C.VESSEL0 + C.N_VESSEL - 1)
    if rng.random() < 0.70 and ves.any():
        idx = np.array(np.nonzero(ves))
        c = idx[:, rng.integers(idx.shape[1])]
    else:
        c = np.array([rng.integers(s) for s in img.shape])
    lo = np.clip(c - P // 2, 0, np.array(img.shape) - P)
    s = tuple(slice(l, l + P) for l in lo)
    return img[s], sem[s]


def augment(img, sem, rng):
    if rng.random() < 0.5:
        img = img[::-1].copy()
        sem = sem[::-1].copy()
        v = (sem >= C.VESSEL0) & (sem <= C.VESSEL0 + C.N_VESSEL - 1)
        sem[v] = C.VESSEL0 + C.LR_SWAP[sem[v] - C.VESSEL0 + 1] - 1   # swap R<->L
    if rng.random() < 0.3:                       # variacion del realce de contraste
        img = img * rng.normal(1.0, 0.05) + rng.normal(0.0, 0.03)
    return img.astype(np.float32), sem.astype(np.uint8)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--epochs", type=int, default=800)
    ap.add_argument("--patch", type=int, default=96)
    ap.add_argument("--bs", type=int, default=6)
    ap.add_argument("--iters", type=int, default=50)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--vessel_w", type=float, default=4.0,
                    help="peso extra de la perdida dentro de los vasos")
    ap.add_argument("--resume", action="store_true")
    a = ap.parse_args()
    (ROOT / "ckpt").mkdir(exist_ok=True)

    data = load_all()
    rng = np.random.default_rng(0)
    val = {d[0] for d in data[::12]}
    tr = [d for d in data if d[0] not in val]
    va = [d for d in data if d[0] in val]
    from collections import Counter
    print(f"train {len(tr)} | val {len(va)}  {Counter(d[1] for d in tr)}", flush=True)

    net = M.build().to(DEV)
    sched = M.scheduler()
    opt = torch.optim.AdamW(net.parameters(), lr=a.lr, weight_decay=1e-6)
    lrs = torch.optim.lr_scheduler.CosineAnnealingLR(opt, a.epochs)
    scaler = torch.amp.GradScaler()
    ep0, best = 0, 1e9

    ck = ROOT / "ckpt" / "gen_last.pt"
    if a.resume and ck.exists():
        st = torch.load(ck, map_location=DEV, weights_only=False)
        net.load_state_dict(st["model"])
        opt.load_state_dict(st["opt"])
        lrs.load_state_dict(st["lrs"])
        ep0, best = st["epoch"] + 1, st["best"]
        print(f"reanudado en la epoca {ep0} (best {best:.4f})", flush=True)

    nT = sched.num_train_timesteps
    for ep in range(ep0, a.epochs):
        net.train()
        tl, t0 = 0.0, time.time()
        for _ in range(a.iters):
            xs, ss = [], []
            for _ in range(a.bs):
                _, _, img, sem = tr[rng.integers(len(tr))]
                pi, ps = sample_patch(img, sem, a.patch, rng)
                pi, ps = augment(pi, ps, rng)
                xs.append(pi[None])
                ss.append(ps)
            x0 = C.normalize(torch.from_numpy(np.stack(xs)).to(DEV))
            sem_t = torch.from_numpy(np.stack(ss).astype(np.int64)).to(DEV)
            seg = F.one_hot(sem_t, C.N_CLASSES).permute(0, 4, 1, 2, 3).float()

            noise = torch.randn_like(x0)
            t = torch.randint(0, nT, (x0.shape[0],), device=DEV)
            xt = sched.add_noise(x0, noise, t)
            target = sched.get_velocity(x0, noise, t)     # v-prediction

            with torch.autocast("cuda", dtype=torch.bfloat16):
                pred = net(x=xt, timesteps=t, seg=seg)
                # los vasos son finisimos: sin este peso, la MSE los ignora
                w = torch.ones_like(x0)
                ves = ((sem_t >= C.VESSEL0) &
                       (sem_t <= C.VESSEL0 + C.N_VESSEL - 1))[:, None]
                w = w + a.vessel_w * ves.float()
                w = w + a.vessel_w * (sem_t == C.THROMBUS)[:, None].float()
                loss = (w * (pred.float() - target) ** 2).sum() / w.sum()

            opt.zero_grad(set_to_none=True)
            scaler.scale(loss).backward()
            scaler.step(opt)
            scaler.update()
            tl += loss.item()
        lrs.step()
        tl /= a.iters

        if (ep + 1) % 20 == 0 or ep == a.epochs - 1:
            net.eval()
            vl, nb = 0.0, 0
            g = torch.Generator(device=DEV).manual_seed(0)
            with torch.no_grad():
                for _, _, img, sem in va:
                    for _ in range(4):
                        pi, ps = sample_patch(img, sem, a.patch, np.random.default_rng(nb))
                        x0 = C.normalize(torch.from_numpy(pi[None, None]).to(DEV))
                        st_ = torch.from_numpy(ps[None].astype(np.int64)).to(DEV)
                        seg = F.one_hot(st_, C.N_CLASSES).permute(0, 4, 1, 2, 3).float()
                        noise = torch.randn(x0.shape, device=DEV, generator=g)
                        t = torch.randint(0, nT, (1,), device=DEV, generator=g)
                        xt = sched.add_noise(x0, noise, t)
                        target = sched.get_velocity(x0, noise, t)
                        with torch.autocast("cuda", dtype=torch.bfloat16):
                            pred = net(x=xt, timesteps=t, seg=seg)
                        vl += F.mse_loss(pred.float(), target).item()
                        nb += 1
            vl /= nb
            print(f"ep {ep+1:4d} loss {tl:.4f} val {vl:.4f} ({time.time()-t0:.0f}s)",
                  flush=True)
            if vl < best:
                best = vl
                torch.save({"model": net.state_dict(), "epoch": ep, "val": vl},
                           ROOT / "ckpt" / "gen_best.pt")
                print(f"   -> gen_best.pt (val {vl:.4f})", flush=True)
        else:
            print(f"ep {ep+1:4d} loss {tl:.4f} ({time.time()-t0:.0f}s)", flush=True)

        torch.save({"model": net.state_dict(), "opt": opt.state_dict(),
                    "lrs": lrs.state_dict(), "epoch": ep, "best": best}, ck)


if __name__ == "__main__":
    main()
