"""Cribado de infarto en NCCT -- filtro de imagen para donantes de control.

No hay que fiarse de las etiquetas del dataset: "sin hemorragia" NO es "sin infarto"
(CQ500/RSNA solo anotan hemorragia; el infarto isquemico ni se mira). Se criba por la
IMAGEN, con un SEGMENTADOR de infarto entrenado (`train_infarct_seg.py`) sobre los pares
reales NCCT+mascara de ISLES'24.

(Se probo antes un cribado por SIMETRIA L-R y se descarto: un cerebro sano tiene ~9 HU de
asimetria media, del mismo orden que la profundidad de un infarto, asi que marcaba todos
los controles. El segmentador aprendido si separa lesion de asimetria normal.)

Cada NCCT candidato -> se windowa [0,80] HU -> [-1,1], se pasa por el U-Net 2D corte a
corte -> se suma el volumen de lesion. Si supera `--min-ml`, se MARCA (posible infarto) y
no entra al banco.

Uso:
    python screen_infarct.py <archivo.nii.gz | carpeta> [--min-ml 2] [--thr 0.5] [--out rep.json]
"""
import sys
import json
import argparse
from pathlib import Path

import numpy as np
import nibabel as nib
import torch
import torch.nn.functional as F
from monai.networks.nets import BasicUNet

ROOT = Path(__file__).parent
CKPT = ROOT / "ckpt" / "infarct_seg2d.pt"
DEV = "cuda" if torch.cuda.is_available() else "cpu"
MIN_ML = 2.0          # volumen de lesion detectado por encima del cual se marca
THR = 0.5             # umbral de probabilidad por voxel

_NET = None
_META = None


def _load_net():
    global _NET, _META
    if _NET is None:
        if not CKPT.exists():
            raise FileNotFoundError(f"falta el checkpoint {CKPT}: entrena antes con "
                                    f"train_infarct_seg.py")
        ck = torch.load(CKPT, map_location=DEV, weights_only=False)
        net = BasicUNet(spatial_dims=2, in_channels=1, out_channels=1,
                        features=tuple(ck.get("features", (32, 32, 64, 128, 256, 32)))).to(DEV)
        net.load_state_dict(ck["model"]); net.eval()
        _NET, _META = net, ck
    return _NET, _META


@torch.no_grad()
def segment(ncct, vox_ml=0.001, thr=THR, batch=32):
    """NCCT (array HU) -> (mascara_lesion bool, volumen_ml). Corte axial = ultimo eje."""
    net, meta = _load_net()
    lo, hi = meta.get("window", (0.0, 80.0))
    n = int(meta.get("inplane", 256))
    ncct = np.asarray(ncct, np.float32)
    x01 = np.clip(ncct, lo, hi)
    x = (x01 - lo) / (hi - lo) * 2.0 - 1.0                # [-1,1], igual que en el entreno
    X, Y, Z = x.shape
    out = np.zeros((X, Y, Z), bool)
    for z0 in range(0, Z, batch):
        sl = x[:, :, z0:z0 + batch]                       # (X,Y,b)
        t = torch.from_numpy(sl.transpose(2, 0, 1))[:, None].to(DEV)   # (b,1,X,Y)
        tr = F.interpolate(t, size=(n, n), mode="bilinear", align_corners=False)
        p = torch.sigmoid(net(tr))
        p = F.interpolate(p, size=(X, Y), mode="bilinear", align_corners=False)  # vuelve al grid nativo
        m = (p[:, 0] > thr).cpu().numpy().transpose(1, 2, 0)             # (X,Y,b)
        out[:, :, z0:z0 + m.shape[2]] = m
    return out, float(out.sum()) * vox_ml


def screen(ncct, vox_ml=0.001, thr=THR, min_ml=MIN_ML):
    """Veredicto para un NCCT (array HU). `flagged=True` -> posible infarto -> descartar."""
    mask, ml = segment(ncct, vox_ml=vox_ml, thr=thr)
    return {"flagged": ml >= min_ml, "lesion_ml": round(ml, 1)}


def screen_file(path, thr=THR, min_ml=MIN_ML):
    im = nib.load(str(path))
    ncct = np.asanyarray(im.dataobj).astype(np.float32)
    vox_ml = float(abs(np.linalg.det(im.affine[:3, :3]))) / 1000.0
    return screen(ncct, vox_ml=vox_ml, thr=thr, min_ml=min_ml)


def _iter(target):
    p = Path(target)
    if p.is_file():
        return [p]
    return sorted(p.glob("*ncct*.nii.gz")) or sorted(p.glob("*.nii.gz"))


def main():
    ap = argparse.ArgumentParser(description="Criba NCCT de control con el segmentador de "
                                             "infarto entrenado; descarta los que tengan lesion.")
    ap.add_argument("target", help="archivo .nii.gz o carpeta de NCCT")
    ap.add_argument("--min-ml", type=float, default=MIN_ML, help="ml de lesion para marcar")
    ap.add_argument("--thr", type=float, default=THR, help="umbral de probabilidad por voxel")
    ap.add_argument("--out", default=None, help="escribe el reporte JSON aqui")
    a = ap.parse_args()

    files = _iter(a.target)
    if not files:
        print(f"no hay NCCT en {a.target}"); sys.exit(1)

    _, meta = _load_net()
    print(f"segmentador de infarto: val Dice {meta.get('val_dice', float('nan')):.3f} "
          f"(epoca {meta.get('epoch', '?')})\n")

    rep, npass, nflag = {}, 0, 0
    for f in files:
        r = screen_file(f, thr=a.thr, min_ml=a.min_ml)
        rep[f.name] = r
        if r["flagged"]:
            nflag += 1
            print(f"  MARCADO  {f.name:42s} -> infarto {r['lesion_ml']} ml")
        else:
            npass += 1
            print(f"  pasa     {f.name:42s} -> lesion {r['lesion_ml']} ml (< {a.min_ml})")

    print(f"\n{npass} pasan (sin infarto) | {nflag} marcados / {len(files)}")
    if a.out:
        Path(a.out).write_text(json.dumps(rep, indent=1, ensure_ascii=False))
        print(f"reporte -> {a.out}")


if __name__ == "__main__":
    main()
