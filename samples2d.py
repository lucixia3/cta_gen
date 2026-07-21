"""Muestras en 2D: cortes axiales de CTA sinteticas, listos para mirar.

    python samples2d.py --n 6            # una lamina por caso + una hoja resumen

Cada lamina lleva el corte al nivel de la arteria ocluida y, al lado, el mismo
corte del paciente SANO, para que se vea que desaparece.
"""
import sys
import argparse
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, str(Path(__file__).parent))
import common as C
import cow_edit as E
import hybrid as H

ROOT = Path(__file__).parent
OUT = ROOT / "samples2d"

# combinaciones que enseñan el acoplamiento variante x oclusion
CASES = [
    ("CoW completo",           "R-M1",        0.45, "Oclusion M1 derecha"),
    ("CoW completo",           "R-ICA-C6-C7", 0.95, "ICA derecha, T carotideo\n(el Acom salva la ACA)"),
    ("Acom ausente",           "R-ICA-C6-C7", 0.95, "ICA derecha, T carotideo\nSIN Acom (cae tambien la ACA)"),
    ("CoW completo",           "R-ICA-C6-C7", 0.25, "ICA derecha PROXIMAL al Pcom\n(el Pcom la rellena: no cae nada)"),
    ("Pcom bilateral ausente", "BA",          0.60, "Basilar SIN Pcom\n(cae todo el territorio posterior)"),
    ("CoW completo",           "L-M2",        0.50, "Oclusion M2 izquierda"),
]


def axial(ax, hu, z, title, lo=0, hi=350):
    ax.imshow(np.clip(hu[:, :, z].T[::-1], lo, hi), cmap="gray", vmin=lo, vmax=hi)
    ax.set_title(title, fontsize=9)
    ax.axis("off")


def z_of(ves, artery, fallback=95):
    m = np.nonzero(ves == E.V[artery])[2]
    return int(np.median(m)) if m.size else fallback


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=len(CASES))
    ap.add_argument("--seed", type=int, default=7)
    a = ap.parse_args()
    OUT.mkdir(exist_ok=True)

    sheet, rows = [], []
    for i, (var, art, frac, tit) in enumerate(CASES[:a.n]):
        seed = a.seed + i * 101
        rng = np.random.default_rng(seed)

        # donante compatible
        cand = sorted((ROOT / "sem" / "topaneu").glob("*.npz"))
        rng.shuffle(cand)
        pick = None
        for p in cand:
            d = np.load(p)
            if var in E.available_variants(d["ves"]) and art in E.occludable(d["ves"]):
                pick = (p.stem, d["sem"], d["ves"])
                break
        case, sem0, ves0 = pick

        hu0 = C.to_hu(np.load(ROOT / "prep" / "topaneu" / f"{case}.npz")["img"]
                      .astype(np.float32))
        hu0, ves0, sem0 = H.randomize_anatomy(hu0, ves0, sem0, rng)
        if art not in E.occludable(ves0):
            print(f"  (saltado {tit}: {art} se perdio al deformar)")
            continue

        v, _ = E.apply_variant(ves0, var)
        v2, thr, info = E.occlude(v, art, frac=frac, rng=rng)
        hu1 = H.render(hu0, ves0, v2, thr, rng=rng)

        z = z_of(ves0, art)
        z = int(np.clip(z, 2, hu0.shape[2] - 3))

        fig, ax = plt.subplots(1, 3, figsize=(12.5, 4.6))
        axial(ax[0], hu0, z, f"SANO (mismo paciente)\nz={z}")
        axial(ax[1], hu1, z, f"{tit}\nz={z}")
        # el arbol, para leer que se ha apagado
        keep = np.max((v2 > 0).astype(float), axis=2).T[::-1]
        lost = np.max(((ves0 > 0) & (v2 == 0)).astype(float), axis=2).T[::-1]
        rgb = np.zeros((*keep.shape, 3))
        rgb[..., 1] = keep
        rgb[..., 0] = lost
        ax[2].imshow(rgb)
        ap_ = ", ".join(info["apagados"]) or "nada"
        ax[2].set_title(f"arbol: verde=contraste, rojo=apagado\napaga: {ap_}", fontsize=8)
        ax[2].axis("off")

        fig.suptitle(f"[{i+1}] {var}  |  oclusion {art} (frac {frac})  |  donante {case}",
                     fontsize=11)
        fig.tight_layout()
        f = OUT / f"sample_{i+1:02d}_{art}_{var.replace(' ','-')}.png"
        fig.savefig(f, dpi=110)
        plt.close(fig)

        sheet.append((hu1, z, f"{tit}\napaga: {ap_[:40]}"))
        rows.append((var, art, info["apagados"]))
        print(f"[{i+1}] {var} / {art} -> apaga {info['apagados']}  -> {f.name}", flush=True)

    # hoja resumen con todos los casos
    n = len(sheet)
    fig, axes = plt.subplots(1, n, figsize=(3.6 * n, 4.4))
    for j, (hu, z, t) in enumerate(sheet):
        axial(axes[j], hu, z, t)
    fig.suptitle("CTA sinteticas — cortes axiales (ventana 0-350 HU)", fontsize=13)
    fig.tight_layout()
    fig.savefig(OUT / "00_resumen.png", dpi=110)
    print(f"\n-> {OUT}  ({n} laminas + 00_resumen.png)")


if __name__ == "__main__":
    main()
