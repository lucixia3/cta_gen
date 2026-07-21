"""Control de calidad visual: MIP de la CTA generada frente a su mapa de vasos.

    python qc.py --grid          # matriz variante x oclusion (lo que importa)
    python qc.py --labels        # solo los mapas editados, sin generar (rapido)
"""
import sys
import argparse
from pathlib import Path

import numpy as np
from scipy import ndimage as ndi
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, str(Path(__file__).parent))
import common as C
import cow_edit as E

ROOT = Path(__file__).parent


def mip(vol, axis=2):
    return np.max(vol, axis=axis).T[::-1]


def vessel_mip(ves, axis=2):
    """MIP de 'hay vaso' (no de la etiqueta, que no es ordinal)."""
    return mip((ves > 0).astype(np.float32), axis)


def bone_subtract(hu, sem, thr_hu=230.0):
    """Angiograma con el craneo sustraido.

    Sin esto el MIP de cabeza completa sale blanco: el hueso satura la proyeccion
    y tapa el arbol. No basta con la clase BONE (umbral >500 HU): el hueso
    trabecular y el calcio viven en 250-500 HU y ensucian el MIP de moteado. Se
    quita todo lo brillante que NO sea vaso etiquetado.
    """
    vessel = (sem >= C.VESSEL0) & (sem <= C.VESSEL0 + C.N_VESSEL - 1)
    vessel |= (sem == C.THROMBUS)
    keep = ndi.binary_dilation(vessel, iterations=2)
    bone = ((sem == C.BONE) | (hu > thr_hu)) & ~keep
    out = hu.copy()
    out[bone] = 0.0
    return ndi.median_filter(out, size=2)                # borra el moteado residual


def show_cta(ax, hu, axis=2, title="", sem=None, lo=100, hi=450):
    v = bone_subtract(hu, sem) if sem is not None else hu
    ax.imshow(mip(np.clip(v, lo, hi), axis), cmap="gray")
    ax.set_title(title, fontsize=8)
    ax.axis("off")


def show_ves(ax, ves, thr=None, axis=2, title=""):
    rgb = np.zeros((*vessel_mip(ves, axis).shape, 3))
    rgb[..., 1] = vessel_mip(ves, axis)                 # verde = con contraste
    if thr is not None and np.any(thr):
        rgb[..., 0] = mip(thr.astype(np.float32), axis)  # rojo = trombo
    ax.imshow(rgb)
    ax.set_title(title, fontsize=8)
    ax.axis("off")


def show_diff(ax, base, ves, thr=None, axis=2, title=""):
    """verde = sigue con contraste | rojo = se ha apagado | blanco = trombo."""
    keep = vessel_mip((ves > 0), axis)
    lost = vessel_mip((base > 0) & (ves == 0), axis)
    rgb = np.zeros((*keep.shape, 3))
    rgb[..., 1] = keep
    rgb[..., 0] = lost
    if thr is not None and np.any(thr):
        t = mip(thr.astype(np.float32), axis)
        rgb[t > 0] = [1, 1, 1]
    ax.imshow(rgb)
    ax.set_title(title, fontsize=8)
    ax.axis("off")


def full_cow_donor():
    """Un donante con Acom y ambas Pcom: sin ellas, la variante no cambia nada."""
    for p in sorted((ROOT / "prep" / "topaneu").glob("*.npz")):
        v = np.load(p)["ves"]
        if all(E.present(v, n) for n in ("Acom", "R-Pcom", "L-Pcom")):
            return p
    raise RuntimeError("ningun donante con CoW completo")


def grid_labels(donor=None, out="qc_labels.png"):
    """Matriz variante x oclusion a nivel de MAPA (sin generador): comprueba la
    hemodinamica -- que se apaga y que se rescata en cada combinacion."""
    p = donor or full_cow_donor()
    ves0 = np.load(p)["ves"]

    variants = ["CoW completo", "Acom ausente", "Pcom bilateral ausente"]
    occl = [("R-ICA-C6-C7", 0.95), ("R-ICA-C6-C7", 0.3), ("R-M1", 0.5), ("BA", 0.6)]

    fig, axes = plt.subplots(len(variants), len(occl) + 1,
                             figsize=(3 * (len(occl) + 1), 3.2 * len(variants)))
    for i, var in enumerate(variants):
        vv, _ = E.apply_variant(ves0, var)
        show_diff(axes[i, 0], ves0, vv, title=f"{var}\n(sin ocluir)")
        for j, (art, fr) in enumerate(occl, 1):
            try:
                v2, thr, info = E.occlude(vv, art, frac=fr)
                ap = ", ".join(info["apagados"]) or "nada"
                if len(ap) > 34:
                    ap = ap[:32] + "..."
                show_diff(axes[i, j], vv, v2, thr,
                          title=f"{art} f={fr}\napaga: {ap}")
            except Exception as ex:
                axes[i, j].text(0.5, 0.5, str(ex)[:40], ha="center", fontsize=6)
                axes[i, j].axis("off")
    fig.suptitle(f"hemodinamica del CoW (donante {Path(p).stem}) — verde: mantiene "
                 f"contraste | rojo: se apaga | blanco: trombo", fontsize=11)
    fig.tight_layout()
    fig.savefig(ROOT / out, dpi=110)
    print(f"-> {ROOT / out}")


def grid_generate(steps=50, out="qc_grid.png"):
    """Igual pero generando la CTA de verdad con el modelo."""
    import torch
    import generate as G
    import model as M

    net = G.load_gen()
    sched = M.scheduler()
    combos = [("CoW completo", None, None),
              ("CoW completo", "R-M1", 0.5),
              ("CoW completo", "R-ICA-C6-C7", 0.95),
              ("Acom ausente", "R-ICA-C6-C7", 0.95),
              ("CoW completo", "BA", 0.6)]

    fig, axes = plt.subplots(2, len(combos), figsize=(3.2 * len(combos), 6.4))
    rng = np.random.default_rng(0)
    for j, (var, art, fr) in enumerate(combos):
        case, sem, ves, aff = G.pick_donor(var, art, np.random.default_rng(7))
        v2, _ = E.apply_variant(ves, var)
        thr = None
        if art:
            v2, thr, _ = E.occlude(v2, art, frac=fr, rng=rng)
        sem2 = G.rebuild_sem(sem, v2, thr)
        x = M.sample(net, sched, sem2, "cuda", steps=steps, seed=7)
        hu = C.to_hu(x)
        show_cta(axes[0, j], hu, title=f"{var}\n{art or 'sin oclusion'}")
        show_ves(axes[1, j], v2, thr, title="mapa de vasos")
        print(f"  generado {var} / {art}", flush=True)
    fig.suptitle("CTA generada (MIP, ventana 100-500 HU)", fontsize=11)
    fig.tight_layout()
    fig.savefig(ROOT / out, dpi=110)
    print(f"-> {ROOT / out}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--labels", action="store_true")
    ap.add_argument("--grid", action="store_true")
    ap.add_argument("--steps", type=int, default=50)
    a = ap.parse_args()
    if a.labels:
        grid_labels()
    if a.grid:
        grid_generate(a.steps)
