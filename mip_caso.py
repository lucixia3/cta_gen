"""MIP angiografico del CTA con la oclusion senalada.

Tres paneles por caso, axial y coronal (que es como se mira un poligono de Willis):

    CTA (MIP)            el angiograma tal cual: se ve el corte de contraste
    + oclusion           el mismo MIP con el trombo marcado y el arbol perdido en rojo
    arbol                verde = sigue con contraste | rojo = apagado | amarillo = trombo

El craneo se sustrae con `qc.bone_subtract`: sin eso el MIP de cabeza completa sale
blanco (el hueso satura la proyeccion) y no se ve el arbol. No basta con quitar la
clase BONE -- el hueso trabecular vive en 250-500 HU -- hay que quitar todo lo brillante
que no sea vaso.

    python mip_caso.py --dir dataset_casos                # un PNG por caso
    python mip_caso.py --dir dataset_casos --caso 02_ICA-der_SIN-Acom
"""
import sys
import json
import argparse
from pathlib import Path

import numpy as np
import nibabel as nib
from scipy import ndimage as ndi
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, str(Path(__file__).parent))
import common as C
import qc as Q

ROOT = Path(__file__).parent
EJE = {"axial": 2, "coronal": 1}     # slab canonico RAS: x=D-I, y=P-A, z=I-S


def marca(ax, thr, axis, color="yellow"):
    """Circulo + flecha donde esta el trombo, sobre el MIP."""
    if thr is None or not thr.any():
        return
    m = Q.mip(thr.astype(np.float32), axis) > 0
    if not m.any():
        return
    yy, xx = np.nonzero(m)
    cy, cx = yy.mean(), xx.mean()
    ax.add_patch(plt.Circle((cx, cy), 13, fill=False, color=color, lw=1.6))
    ax.annotate("oclusión", xy=(cx, cy), xytext=(cx + 42, cy - 34),
                color=color, fontsize=8, ha="left",
                arrowprops=dict(arrowstyle="->", color=color, lw=1.2))


def panel(d, out=None):
    info = json.loads((d / "caso.json").read_text(encoding="utf-8"))
    hu = np.asanyarray(nib.load(d / "cta_oclusion.nii.gz").dataobj).astype(np.float32)
    ves2 = np.asanyarray(nib.load(d / "vasos.nii.gz").dataobj).astype(np.uint8)
    thr = np.asanyarray(nib.load(d / "trombo.nii.gz").dataobj) > 0
    # el arbol sano DE ESTE CASO (reflejado/deformado), no el crudo del banco
    ves0 = np.asanyarray(nib.load(d / "vasos_sano.nii.gz").dataobj).astype(np.uint8)

    # el mapa semantico del CASO (arbol ya editado + trombo): es lo que necesita
    # bone_subtract para saber que es vaso y que es hueso
    sem = C.semantic_map(hu, ves2, None, thr)
    ang = Q.bone_subtract(hu, sem)

    fig, axs = plt.subplots(2, 3, figsize=(11.5, 8))
    for r, (nombre, axis) in enumerate(EJE.items()):
        # 1. el angiograma tal cual
        axs[r, 0].imshow(Q.mip(np.clip(ang, 100, 450), axis), cmap="gray")
        axs[r, 0].set_title(f"CTA — MIP {nombre}", fontsize=9)

        # 2. el mismo, con la oclusion marcada y lo perdido en rojo
        base = Q.mip(np.clip(ang, 100, 450), axis)
        base = (base - base.min()) / (np.ptp(base) + 1e-6)
        rgb = np.dstack([base] * 3)
        lost = Q.vessel_mip((ves0 > 0) & (ves2 == 0), axis) > 0
        rgb[lost] = 0.45 * rgb[lost] + 0.55 * np.array([1.0, 0.2, 0.2])
        axs[r, 1].imshow(rgb)
        axs[r, 1].set_title("rojo = árbol sin contraste", fontsize=9)
        marca(axs[r, 1], thr, axis)

        # 3. solo el arbol
        Q.show_diff(axs[r, 2], ves0, ves2, thr, axis=axis,
                    title="verde=contraste | rojo=apagado | blanco=trombo")
        marca(axs[r, 2], thr, axis, color="cyan")

        for ax in axs[r]:
            ax.axis("off")
            ax.text(0.97, 0.04, "R", color="yellow", fontsize=10,
                    ha="right", transform=ax.transAxes)

    ap = ", ".join(info["apagados"]) or "nada"
    # `territorios: null` (lote sin NCCT de `batch_cta.py`) no es una lista vacia:
    # null = no se ha calculado; vacia = no infarta nada.
    terr = info.get("territorios")
    vol = info.get("volumen_infarto_ml")
    pie = ("infarto: no calculado (lote sin NCCT)" if terr is None else
           f"infarto: {', '.join(terr) or 'SIN INFARTO'}"
           + (f"  ({vol} ml)" if vol is not None else ""))
    fig.suptitle(f"{d.name}   —   {info['variante']}  +  "
                 f"{info['arteria'] or 'sin oclusión'}"
                 + (f" (frac {info['frac']})" if info["frac"] else "")
                 + f"\napaga: {ap}   →   {pie}",
                 fontsize=10)
    fig.tight_layout(rect=(0, 0, 1, 0.93))
    out = out or d / "mip.png"
    fig.savefig(out, dpi=110)
    plt.close(fig)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", default="dataset_casos")
    ap.add_argument("--caso", default=None, help="solo uno (nombre de la subcarpeta)")
    a = ap.parse_args()

    base = Path(a.dir)
    dirs = ([base / a.caso] if a.caso
            else sorted(p for p in base.iterdir() if (p / "caso.json").exists()))
    for d in dirs:
        if not (d / "trombo.nii.gz").exists():
            print(f"  {d.name}: sin trombo.nii.gz (regenera el lote)")
            continue
        print(f"-> {panel(d)}")


if __name__ == "__main__":
    main()
