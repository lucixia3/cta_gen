"""Lamina comparativa: la misma oclusion, distinta variante -> distinto infarto.

Es la figura que demuestra el punto del generador. Cada fila es un caso; a la
izquierda el arbol arterial del CTA (verde = con contraste, rojo = apagado por la
oclusion), a la derecha el NCCT con el infarto que produce ese arbol apagado.

    python samples_caso.py            # usa casos/*  ya generados

Los dos volumenes se llevan a RAS antes de pintarlos. Sin eso salen espejados el uno
respecto al otro -- el CTA se guarda en el slab canonico (RAS) y el NCCT en MNI (LAS),
que tienen el eje x al reves -- y la lamina mostraria la oclusion y el infarto en
lados opuestos aunque el caso este perfecto.
"""
import sys
import json
import argparse
from pathlib import Path

import numpy as np
import nibabel as nib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, str(Path(__file__).parent))
import common as C

ROOT = Path(__file__).parent


def ras(arr, aff):
    """A RAS: eje 0 = x creciente hacia la DERECHA del paciente, siempre."""
    im = nib.as_closest_canonical(nib.Nifti1Image(np.asarray(arr), aff))
    return np.asanyarray(im.dataobj), im.affine


def axial(sl):
    """Corte (x, y) -> lamina con anterior arriba y la derecha del paciente a la derecha."""
    return sl.T[::-1]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", default="dataset_casos", help="carpeta del lote")
    a = ap.parse_args()

    base = Path(a.dir)
    dirs = sorted(p for p in base.iterdir() if (p / "caso.json").exists())
    if not dirs:
        raise SystemExit(f"no hay casos en {base}: corre primero batch_casos.py")

    fig, axs = plt.subplots(len(dirs), 2, figsize=(8, 3.0 * len(dirs)))
    axs = np.atleast_2d(axs)

    for r, d in enumerate(dirs):
        name = d.name
        info = json.loads((d / "caso.json").read_text(encoding="utf-8"))

        # --- izquierda: el arbol arterial. verde = con contraste, rojo = apagado
        # el arbol sano DE ESTE CASO (reflejado/deformado), no el crudo del banco
        sano = nib.load(d / "vasos_sano.nii.gz")
        v0, vaff = ras(np.asanyarray(sano.dataobj), sano.affine)
        v2, _ = ras(np.asanyarray(nib.load(d / "vasos.nii.gz").dataobj), sano.affine)
        keep = np.max((v2 > 0).astype(float), axis=2)
        lost = np.max(((v0 > 0) & (v2 == 0)).astype(float), axis=2)
        rgb = np.zeros((*axial(keep).shape, 3))
        rgb[..., 1] = axial(keep)
        rgb[..., 0] = axial(lost)
        axs[r, 0].imshow(rgb)
        apg = ", ".join(info["apagados"]) or "nada"
        axs[r, 0].set_title(f"{name}\n{info['variante']} + {info['arteria'] or 'sin oclusion'}\n"
                            f"apaga: {apg}", fontsize=7)

        # --- derecha: el NCCT con el infarto
        nc, naff = ras(np.asanyarray(nib.load(d / "ncct_infarto.nii.gz").dataobj),
                       nib.load(d / "ncct_infarto.nii.gz").affine)
        mim = nib.load(d / "mask.nii.gz")
        mk, _ = ras(np.asanyarray(mim.dataobj).astype(np.uint8), mim.affine)
        mk = mk > 0
        z = int(np.array(np.nonzero(mk)).mean(1)[2]) if mk.any() else nc.shape[2] // 2
        gris = np.clip(axial(nc[:, :, z]) / 80.0, 0, 1)     # ventana de parenquima 0-80 HU
        img = np.dstack([gris] * 3)
        m = axial(mk[:, :, z]).astype(bool)
        img[m] = 0.65 * img[m] + 0.35 * np.array([1.0, 0.15, 0.15])
        axs[r, 1].imshow(img)
        terr = ", ".join(info["territorios"]) or "SIN INFARTO"
        axs[r, 1].set_title(f"NCCT — {terr}\n{info['volumen_infarto_ml']} ml", fontsize=7)

        for ax in axs[r]:
            ax.axis("off")
            ax.text(0.97, 0.5, "R", color="yellow", fontsize=10, va="center",
                    ha="right", transform=ax.transAxes)

    fig.suptitle("árbol arterial (verde=contraste, rojo=apagado)   |   NCCT con el infarto",
                 fontsize=10)
    fig.tight_layout(rect=(0, 0, 1, 0.995))
    out = base / "qc_lote.png"
    fig.savefig(out, dpi=85)
    print(f"-> {out}  ({len(dirs)} casos)")


if __name__ == "__main__":
    main()
