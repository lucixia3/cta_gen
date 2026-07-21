"""API de generacion: variante de CoW + sitio de oclusion -> CTA sintetica.

    python generate.py --variant "Acom ausente" --artery R-ICA-C6-C7 --frac 0.9
    python generate.py --list                     # variantes y arterias disponibles

Un unico modelo. El pipeline es: se toma un donante, se edita su mapa de vasos
(variante + oclusion, con la hemodinamica del grafo del CoW en cow_edit) y el
generador pinta la CTA a partir del mapa editado.
"""
import sys
import json
import argparse
from pathlib import Path

import numpy as np
import torch
from scipy import ndimage as ndi

sys.path.insert(0, str(Path(__file__).parent))
import common as C
import cow_edit as E
import model as M

ROOT = Path(__file__).parent
DEV = "cuda"


def donors():
    return sorted((ROOT / "sem" / "topaneu").glob("*.npz")) + \
           sorted((ROOT / "sem" / "topcow").glob("*.npz"))


def load_donor(path):
    d = np.load(path)
    return d["sem"], d["ves"], d["affine"]


def pick_donor(variant, artery, rng):
    """Un donante que tenga los segmentos que la variante y la oclusion tocan."""
    cand = donors()
    rng.shuffle(cand)
    for p in cand:
        sem, ves, aff = load_donor(p)
        if variant not in E.available_variants(ves):
            continue
        if artery and artery not in E.occludable(ves):
            continue
        return p.stem, sem, ves, aff
    raise RuntimeError(f"ningun donante compatible con {variant} + {artery}")


def rebuild_sem(sem, ves, thr=None):
    """Mapa semantico nuevo: tejido del donante + vasos editados + trombo.

    Todo lo que era vaso/aneurisma y ya no esta pasa a parenquima, que es lo que
    el generador debe pintar donde el contraste ha desaparecido.
    """
    out = sem.copy()
    out[out >= C.VESSEL0] = C.SOFT
    m = ves > 0
    out[m] = C.VESSEL0 + ves[m].astype(np.uint8) - 1
    if thr is not None and np.any(thr):
        out[thr] = C.THROMBUS
    return out


def deform(sem, rng, alpha=6.0, sigma=12.0):
    """Deformacion elastica del mapa -> 'otro paciente' con la misma variante."""
    g = [ndi.gaussian_filter(rng.normal(0, 1, sem.shape), sigma) * alpha
         for _ in range(3)]
    idx = np.indices(sem.shape, dtype=np.float32)
    co = [idx[i] + g[i] for i in range(3)]
    return ndi.map_coordinates(sem, co, order=0, mode="nearest").astype(np.uint8)


def load_gen():
    ck = torch.load(ROOT / "ckpt" / "gen_best.pt", map_location=DEV, weights_only=False)
    net = M.build().to(DEV)
    net.load_state_dict(ck["model"])
    net.eval()
    print(f"generador: epoca {ck['epoch']+1}, val {ck['val']:.4f}")
    return net


def generate(variant="CoW completo", artery=None, frac=None, seed=None,
             steps=50, do_deform=False, out=None, net=None):
    rng = np.random.default_rng(seed)
    case, sem, ves, aff = pick_donor(variant, artery, rng)

    ves2, vinfo = E.apply_variant(ves, variant)
    thr, oinfo = None, {}
    if artery:
        if frac is None:
            frac = float(rng.uniform(0.35, 0.9))
        ves2, thr, oinfo = E.occlude(ves2, artery, frac=frac, rng=rng)

    sem2 = rebuild_sem(sem, ves2, thr)
    if do_deform:
        sem2 = deform(sem2, rng)

    net = net or load_gen()
    sched = M.scheduler()
    x = M.sample(net, sched, sem2, DEV, steps=steps, seed=seed,
                 progress=lambda i, n: print(f"  paso {i}/{n}", flush=True))
    hu = C.to_hu(x)

    tag = variant.replace(" ", "-")
    if artery:
        tag += f"_ocl-{artery}-f{frac:.2f}"
    out = Path(out or ROOT / "out" / f"{tag}.nii.gz")
    out.parent.mkdir(parents=True, exist_ok=True)
    C.save_nifti(out, hu.astype(np.float32), aff)
    C.save_nifti(str(out).replace(".nii.gz", "_sem.nii.gz"), sem2.astype(np.int16), aff)

    info = {"donante": case, "variante": variant, "cambios": vinfo,
            "oclusion": oinfo, "salida": str(out)}
    print(json.dumps(info, indent=1, ensure_ascii=False))
    return hu, sem2, info


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--variant", default="CoW completo")
    ap.add_argument("--artery", default=None)
    ap.add_argument("--frac", type=float, default=None,
                    help="posicion del trombo en el segmento (0=proximal, 1=distal)")
    ap.add_argument("--seed", type=int, default=None)
    ap.add_argument("--steps", type=int, default=50)
    ap.add_argument("--deform", action="store_true")
    ap.add_argument("--out", default=None)
    ap.add_argument("--list", action="store_true")
    a = ap.parse_args()

    if a.list:
        ds = donors()
        print(f"{len(ds)} donantes\n\nvariantes:")
        for v in E.VARIANTS:
            n = sum(1 for p in ds if v in E.available_variants(load_donor(p)[1]))
            print(f"  {v:28s} ({n} donantes)")
        print("\narterias ocluibles:")
        for x in C.OCCLUDABLE:
            print(f"  {x}")
        return

    generate(a.variant, a.artery, a.frac, a.seed, a.steps, a.deform, a.out)


if __name__ == "__main__":
    main()
