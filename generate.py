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
    # `tis` (encefalo / extracraneal / LCR) es IMPRESCINDIBLE al muestrear: son los
    # canales de tejido que el modelo aprendio. Sin ellos el parenquima y el LCR
    # colapsan al mismo gris (~24 HU). Ver el porque en `generate`.
    if "tis" not in d:
        raise SystemExit(f"{path.name} no tiene 'tis'. Ejecuta: python tejidos.py")
    return d["sem"], d["ves"], d["tis"], d["affine"]


def pick_donor(variant, artery, rng):
    """Un donante que tenga los segmentos que la variante y la oclusion tocan."""
    cand = donors()
    rng.shuffle(cand)
    for p in cand:
        sem, ves, tis, aff = load_donor(p)
        if variant not in E.available_variants(ves):
            continue
        if artery and artery not in E.occludable(ves):
            continue
        return p.stem, sem, ves, tis, aff
    raise RuntimeError(f"ningun donante compatible con {variant} + {artery}")


def rebuild_sem(sem, ves):
    """Mapa semantico nuevo: tejido del donante + vasos editados.

    Todo lo que era vaso/aneurisma y ya no esta pasa a parenquima, que es lo que
    el generador debe pintar donde el contraste ha desaparecido. La oclusion es
    justamente eso -- el vaso distal deja de verse -- y NO se marca ningun coagulo.
    """
    out = sem.copy()
    out[out >= C.VESSEL0] = C.SOFT
    m = ves > 0
    out[m] = C.VESSEL0 + ves[m].astype(np.uint8) - 1
    return out


def deform(rng, *mapas, alpha=6.0, sigma=12.0):
    """Deformacion elastica -> 'otro paciente' con la misma variante.

    Aplica EL MISMO campo a todos los mapas (sem, tis, ...) para que sigan
    alineados; devolverlos por separado si se pasa mas de uno.
    """
    shape = mapas[0].shape
    g = [ndi.gaussian_filter(rng.normal(0, 1, shape), sigma) * alpha for _ in range(3)]
    idx = np.indices(shape, dtype=np.float32)
    co = [idx[i] + g[i] for i in range(3)]
    out = [ndi.map_coordinates(m, co, order=0, mode="nearest").astype(m.dtype)
           for m in mapas]
    return out[0] if len(out) == 1 else out


def suaviza_parenquima(hu, sem, sigma=0.5):
    """Quita el moteado fino del parenquima SIN tocar vasos ni hueso.

    La difusion deja un grano de un voxel (|laplaciano| ~2x el del ruido de CT
    real) mientras que el ruido de alta frecuencia global ya casa. Un gaussiano
    suave (sigma 0.5) baja ese moteado hasta el nivel real; sigma mayor emborrona
    y da aspecto plastico. Se protege todo lo brillante (vaso/hueso etiquetado y
    >230 HU) para no perder nitidez donde importa. sigma=0 lo desactiva.
    """
    if not sigma:
        return hu
    ves = (sem >= C.VESSEL0) & (sem <= C.VESSEL0 + C.N_VESSEL - 1)
    protege = ndi.binary_dilation(ves | (sem == C.BONE) | (hu > 230), iterations=1)
    sm = ndi.gaussian_filter(hu, sigma)
    out = hu.copy()
    out[~protege] = sm[~protege]
    return out


def load_gen(ckpt="gen_best.pt"):
    ck = torch.load(ROOT / "ckpt" / ckpt, map_location=DEV, weights_only=False)
    net = M.build().to(DEV)
    net.load_state_dict(ck["model"])
    net.eval()
    val = ck.get("val") or ck.get("best")
    print(f"generador: {ckpt}, epoca {ck['epoch']+1}, val {val:.4f}")
    return net


def generate(variant="CoW completo", artery=None, frac=None, seed=None,
             steps=50, do_deform=False, out=None, net=None, ckpt="gen_best.pt",
             guidance=1.3, denoise=0.5):
    rng = np.random.default_rng(seed)
    case, sem, ves, tis, aff = pick_donor(variant, artery, rng)

    ves2, vinfo = E.apply_variant(ves, variant)
    oinfo = {}
    if artery:
        if frac is None:
            frac = float(rng.uniform(0.35, 0.9))
        # E.occlude devuelve tambien una mascara de trombo; NO se usa: la oclusion
        # es la ausencia de contraste distal, sin coagulo hiperdenso.
        ves2, _, oinfo = E.occlude(ves2, artery, frac=frac, rng=rng)

    sem2 = rebuild_sem(sem, ves2)
    if do_deform:
        sem2, tis = deform(rng, sem2, tis)      # el mismo campo a los dos

    net = net or load_gen(ckpt)
    sched = M.sampler()          # DDIM: con DDPM los vasos salen a +14 HU (ver model.sampler)
    # `tis` va SIEMPRE: son los canales de tejido del modelo. Omitirlo colapsaba el
    # parenquima y el LCR al mismo gris (bug historico de generate).
    x = M.sample(net, sched, sem2, DEV, tis=tis, steps=steps, seed=seed,
                 guidance=guidance,
                 progress=lambda i, n: print(f"  paso {i}/{n}", flush=True))
    hu = suaviza_parenquima(C.to_hu(x), sem2, sigma=denoise)

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
    ap.add_argument("--guidance", type=float, default=1.3,
                    help="amplifica la condicion en el muestreo -> intensidad del "
                         "vaso. 1.0=normal (~66%% del real), 1.3=iguala el real. "
                         "Ver model.sample")
    ap.add_argument("--denoise", type=float, default=0.5,
                    help="sigma del suavizado del parenquima (protege vaso/hueso). "
                         "0.5 iguala el grano al real, 0=desactivado")
    ap.add_argument("--deform", action="store_true")
    ap.add_argument("--out", default=None)
    ap.add_argument("--ckpt", default="gen_best.pt")
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

    generate(a.variant, a.artery, a.frac, a.seed, a.steps, a.deform, a.out,
             ckpt=a.ckpt, guidance=a.guidance, denoise=a.denoise)


if __name__ == "__main__":
    main()
