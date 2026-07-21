"""Render 3D del arbol arterial: que sigue con contraste y que se ha apagado.

El MIP de `mip_caso.py` proyecta y aplasta: en el poligono, donde todo se cruza, no
deja ver que rama nace de donde. Aqui la superficie se reconstruye con marching cubes
y se mira desde tres angulos, que es lo que hace falta para leer una variante.

    verde    sigue perfundido
    rojo     ha perdido el contraste (lo que apaga la oclusion)
    amarillo el trombo

    python arbol3d.py --dir dataset_casos                     # un PNG por caso
    python arbol3d.py --dir dataset_casos --caso 02_ICA-der_SIN-Acom
    python arbol3d.py --dir dataset_casos --montaje           # todos en una lamina

El coste esta en el numero de triangulos, no en el volumen: `--paso` decima la malla
(2 por defecto). Con paso 1 la superficie es mas fina y tarda ~4x mas.
"""
import sys
import json
import argparse
from pathlib import Path

import numpy as np
import nibabel as nib
from scipy import ndimage as ndi
from skimage import measure
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d.art3d import Poly3DCollection

sys.path.insert(0, str(Path(__file__).parent))
import common as C

ROOT = Path(__file__).parent

# slab canonico RAS: x = D-I, y = P-A, z = I-S
VISTAS = [
    ("superior",  dict(elev=78,  azim=-90)),   # como se mira un poligono de Willis
    ("oblicua",   dict(elev=22,  azim=-62)),
    ("lateral I", dict(elev=6,   azim=0)),
]

VERDE    = np.array([0.25, 0.88, 0.42])
ROJO     = np.array([1.00, 0.26, 0.26])
AMARILLO = np.array([1.00, 0.86, 0.12])
FONDO    = "#0d1117"      # el arbol se lee mucho mejor sobre oscuro, como un angiograma
LUZ = np.array([0.4, -0.7, 0.6]);  LUZ = LUZ / np.linalg.norm(LUZ)


def _malla(mask, paso, sigma=0.7):
    """Superficie de una mascara binaria. None si no hay nada que dibujar."""
    if not mask.any():
        return None
    # marching cubes sobre la binaria da escalones; un suavizado corto los quita
    # sin mover el tubo (los vasos son de 1-2 voxeles y no aguantan mas sigma).
    vol = ndi.gaussian_filter(mask.astype(np.float32), sigma)
    if vol.max() < 0.5:                      # vaso tan fino que el filtro lo borra
        vol = mask.astype(np.float32)
    try:
        v, f, n, _ = measure.marching_cubes(vol, level=0.5, step_size=paso)
    except (ValueError, RuntimeError):
        return None
    if len(f) == 0:
        return None
    return v * C.SPACING, f, n


def _infarto(info):
    """Pie del titulo. `territorios: null` (lote sin NCCT, `batch_cta.py`) NO es lo mismo
    que una lista vacia: null = no se ha calculado, vacia = no infarta nada."""
    t, v = info.get("territorios"), info.get("volumen_infarto_ml")
    if t is None:
        return "infarto: no calculado (lote sin NCCT)"
    return f"infarto: {', '.join(t) or 'SIN INFARTO'}" + (f"  ({v} ml)" if v is not None else "")


def _gordo(thr, r=3):
    """El trombo mide 4 mm y queda enterrado entre los vasos: se engorda para que asome."""
    if not thr.any():
        return thr
    return ndi.binary_dilation(thr, ndi.generate_binary_structure(3, 1), iterations=r)


def _marca(ax, thr):
    """La malla del trombo queda tapada por los vasos que la rodean (matplotlib no tiene
    z-buffer real entre colecciones). Un marcador plano encima siempre se ve."""
    if not thr.any():
        return
    c = np.array(np.nonzero(thr)).mean(axis=1) * C.SPACING
    ax.scatter(*c, s=190, facecolor="none", edgecolor=AMARILLO,
               linewidths=1.8, depthshade=False, zorder=10)


def _sombra(verts, faces, normals, color):
    """Lambertiano simple por cara: sin esto la malla sale como una silueta plana."""
    fn = normals[faces].mean(axis=1)
    fn /= (np.linalg.norm(fn, axis=1, keepdims=True) + 1e-9)
    lam = np.clip(fn @ LUZ, 0, 1)
    tono = 0.42 + 0.58 * lam                       # ambiente + difusa
    return np.clip(color[None, :] * tono[:, None], 0, 1)


def _pinta(ax, mallas, lims, zoom=1.55):
    for (v, f, n), color, alpha in mallas:
        pc = Poly3DCollection(v[f], alpha=alpha, linewidths=0)
        pc.set_facecolor(_sombra(v, f, n, color))
        pc.set_edgecolor("none")
        ax.add_collection3d(pc)
    (x0, x1), (y0, y1), (z0, z1) = lims
    ax.set_xlim(x0, x1); ax.set_ylim(y0, y1); ax.set_zlim(z0, z1)
    # sin zoom, matplotlib deja tanto margen que el arbol sale diminuto
    ax.set_box_aspect((x1 - x0, y1 - y0, z1 - z0), zoom=zoom)
    ax.set_axis_off()
    ax.set_facecolor(FONDO)


def _limites(mask, margen=6.0):
    """Caja del arbol SANO, compartida por todas las vistas y por todos los paneles."""
    idx = np.array(np.nonzero(mask))
    lo = idx.min(axis=1) * C.SPACING - margen
    hi = idx.max(axis=1) * C.SPACING + margen
    return list(zip(lo, hi))


def caso3d(d, paso=2, out=None):
    info = json.loads((d / "caso.json").read_text(encoding="utf-8"))
    ves0 = np.asanyarray(nib.load(d / "vasos_sano.nii.gz").dataobj).astype(np.uint8)
    ves2 = np.asanyarray(nib.load(d / "vasos.nii.gz").dataobj).astype(np.uint8)
    thr = np.asanyarray(nib.load(d / "trombo.nii.gz").dataobj) > 0

    perm = ves2 > 0
    lost = (ves0 > 0) & (ves2 == 0)
    lims = _limites(ves0 > 0)

    mallas = []
    for m, col, al in ((perm, VERDE, 1.0), (lost, ROJO, 1.0), (_gordo(thr), AMARILLO, 1.0)):
        g = _malla(m, paso)
        if g is not None:
            mallas.append((g, col, al))

    fig = plt.figure(figsize=(13.5, 5.4), facecolor=FONDO)
    for k, (nombre, ang) in enumerate(VISTAS, 1):
        ax = fig.add_subplot(1, len(VISTAS), k, projection="3d")
        _pinta(ax, mallas, lims)
        _marca(ax, thr)
        ax.view_init(**ang)
        ax.set_title(nombre, fontsize=9, color="0.85")

    ap = ", ".join(info["apagados"]) or "nada"
    fig.suptitle(f"{d.name}   —   {info['variante']}  +  "
                 f"{info['arteria'] or 'sin oclusión'}"
                 + (f" (frac {info['frac']})" if info["frac"] else "")
                 + f"\napaga: {ap}   →   {_infarto(info)}"
                 + "\nverde = con contraste | rojo = apagado | amarillo = trombo",
                 fontsize=9.5, color="0.92")
    fig.tight_layout(rect=(0, 0, 1, 0.86))
    out = out or d / "arbol3d.png"
    fig.savefig(out, dpi=115)
    plt.close(fig)
    return out


def montaje(dirs, paso=3, out="arbol3d_lote.png", vista=1):
    """Una vista por caso, todos en la misma lamina: el barrido de un vistazo."""
    ncol = 4
    nfil = int(np.ceil(len(dirs) / ncol))
    fig = plt.figure(figsize=(3.6 * ncol, 3.3 * nfil), facecolor=FONDO)
    nombre, ang = VISTAS[vista]
    for k, d in enumerate(dirs, 1):
        info = json.loads((d / "caso.json").read_text(encoding="utf-8"))
        ves0 = np.asanyarray(nib.load(d / "vasos_sano.nii.gz").dataobj).astype(np.uint8)
        ves2 = np.asanyarray(nib.load(d / "vasos.nii.gz").dataobj).astype(np.uint8)
        thr = np.asanyarray(nib.load(d / "trombo.nii.gz").dataobj) > 0

        mallas = []
        for m, col in (((ves2 > 0), VERDE), (((ves0 > 0) & (ves2 == 0)), ROJO), (thr, AMARILLO)):
            g = _malla(m, paso)
            if g is not None:
                mallas.append((g, col, 1.0))

        ax = fig.add_subplot(nfil, ncol, k, projection="3d")
        _pinta(ax, mallas, _limites(ves0 > 0))
        ax.view_init(**ang)
        # con territorio se muestra el volumen; sin el (lote sin NCCT) el donante, que
        # ahi es el dato que importa: cada caso cae en un cerebro distinto
        v = info.get("volumen_infarto_ml")
        pie = f"{v} ml" if v is not None else info.get("donante", "")
        ax.set_title(f"{d.name}\n{info['variante']} + {info['arteria'] or 'sano'}"
                     f"  ({pie})", fontsize=7.5, color="0.85")
        print(f"  [{k}/{len(dirs)}] {d.name}", flush=True)

    fig.suptitle(f"Árbol arterial 3D — vista {nombre}   "
                 "[verde = con contraste | rojo = apagado | amarillo = trombo]",
                 fontsize=11, color="0.92")
    fig.tight_layout(rect=(0, 0, 1, 0.965))
    fig.savefig(out, dpi=105)
    plt.close(fig)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", default="dataset_casos")
    ap.add_argument("--caso", default=None, help="solo uno (nombre de la subcarpeta)")
    ap.add_argument("--paso", type=int, default=2, help="decimado de la malla (1 = fino)")
    ap.add_argument("--montaje", action="store_true", help="todos en una sola lamina")
    a = ap.parse_args()

    base = Path(a.dir)
    dirs = ([base / a.caso] if a.caso
            else sorted(p for p in base.iterdir() if (p / "caso.json").exists()))
    dirs = [d for d in dirs if (d / "vasos_sano.nii.gz").exists()]
    if not dirs:
        print(f"sin casos con vasos_sano.nii.gz en {base}")
        return

    if a.montaje:
        print(f"-> {montaje(dirs, paso=max(a.paso, 3), out=base / 'arbol3d_lote.png')}")
    else:
        for d in dirs:
            print(f"-> {caso3d(d, paso=a.paso)}", flush=True)


if __name__ == "__main__":
    main()
