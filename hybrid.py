"""Renderizador HIBRIDO: CTA real del donante + el arbol editado.

El valor del proyecto esta en el grafo hemodinamico (`cow_edit`), no en como se
pinte la imagen. Este renderizador toma la CTA REAL del donante y le aplica la
misma edicion (variante + oclusion): calidad angiografica real, en CPU, al
instante, sin depender de que la difusion converja.

Frente al hibrido de PYTHIA: alli el mapa solo tenia las 15 clases del poligono,
asi que al ocluir un M1 no se podia apagar el M2/M3 (no estaban etiquetados). Con
las 36 clases de TopAneu se suprime el arbol distal ENTERO, que es justo el
hallazgo radiologico de una LVO.

    python hybrid.py --variant "Acom ausente" --artery R-ICA-C6-C7 --frac 0.9
"""
import sys
import json
import argparse
from pathlib import Path

import numpy as np
from scipy import ndimage as ndi

sys.path.insert(0, str(Path(__file__).parent))
import common as C
import cow_edit as E
import generate as G

ROOT = Path(__file__).parent

PARENCHYMA_HU = 38.0      # sustituto del lumen suprimido
THROMBUS_HU = 58.0        # coagulo agudo: hiperdenso frente al parenquima


def _capture_vessel(hu, mask, halo=2):
    """Region a borrar: el lumen + su borde brillante de volumen parcial.

    La etiqueta cubre el lumen, pero el contraste se difumina 1-3 voxeles mas alla; si
    solo se borra el interior queda un anillo brillante que dibuja el vaso. Un `halo`
    fijo no vale: la ICA (5 mm) tiene mas penumbra que una M3 fina. Se captura el borde
    de verdad -- los voxeles vecinos que siguen claros frente al fondo local -- ademas
    de una dilatacion minima de seguridad.
    """
    grow = ndi.binary_dilation(mask, iterations=1)
    # fondo local: tejido/LCR alrededor (ni vaso ni hueso)
    around = ndi.binary_dilation(grow, iterations=6) & ~ndi.binary_dilation(grow, iterations=2)
    ref = hu[around]
    ref = ref[(ref > -50) & (ref < 120)]
    bg = float(np.median(ref)) if ref.size > 50 else PARENCHYMA_HU
    # crece mientras el vecino siga claramente por encima del fondo (borde del contraste)
    region = grow.copy()
    for _ in range(halo + 3):
        shell = ndi.binary_dilation(region, iterations=1) & ~region
        bright = shell & (hu > bg + 35.0)
        if not bright.any():
            break
        region |= bright
    region |= ndi.binary_dilation(mask, iterations=halo)   # minimo de seguridad
    return ndi.binary_closing(region, iterations=1), bg, ref


def suppress_lumen(hu, mask, halo=2, rng=None):
    """Borra el vaso de modo que quede COMO SI NUNCA HUBIERA ESTADO AHI.

    Relleno biharmonico (membrana suave que iguala EXACTAMENTE el borde de la region):
    no deja costura y, sobre todo, hereda el nivel del entorno sea cual sea -- si el vaso
    iba por una cisterna oscura, el hueco queda oscuro como el LCR; si iba entre
    parenquima, queda a nivel de parenquima. El ajuste de brillo GLOBAL de la version
    anterior fallaba justo ahi: subia el hueco al nivel del parenquima y dejaba un parche
    claro en mitad de la cisterna. Encima se le devuelve el grano del TC.
    """
    if not mask.any():
        return hu
    rng = rng or np.random.default_rng()
    from skimage.restoration import inpaint_biharmonic

    region, bg, ref = _capture_vessel(hu, mask, halo)

    # GRANO real del TC, no la varianza entre tejidos. `ref` mezcla LCR oscuro y
    # parenquima, asi que su std es enorme (~15-20 HU): usarla mete un moteado blanco
    # que delata el parche. El grano de verdad es el residuo de alta frecuencia del
    # parenquima homogeneo (quantum mottle, ~5-7 HU): se estima quitandole la baja
    # frecuencia a un anillo de parenquima.
    par = (ndi.binary_dilation(region, iterations=6) & ~ndi.binary_dilation(region, iterations=2)
           & (hu > 20) & (hu < 55))
    if par.sum() > 80:
        hp = hu - ndi.gaussian_filter(hu, 2.0)        # paso alto: se queda el grano
        sd = float(np.clip(np.std(hp[par]), 3.0, 8.0))
    else:
        sd = 6.0

    # se trabaja en la caja de la region (+ margen): el biharmonico necesita ver el
    # borde sano alrededor, y asi no se procesa el volumen entero
    z, y, x = np.nonzero(region)
    m = 6
    sl = tuple(slice(max(0, c.min() - m), min(s, c.max() + 1 + m))
               for c, s in zip((z, y, x), hu.shape))
    sub, sub_m = hu[sl].astype(np.float32), region[sl]

    filled = inpaint_biharmonic(sub, sub_m).astype(np.float32)

    # grano fino a la sigma del parenquima, solo dentro del hueco y sin tocar el borde
    noise = ndi.gaussian_filter(rng.normal(0, 1, sub.shape).astype(np.float32), 0.6)
    noise *= sd / (noise.std() + 1e-6)
    inner = ndi.binary_erosion(sub_m, iterations=1)
    filled = filled + noise * inner

    out = hu.copy()
    # fundido suave de 1 voxel para que no se note ni el borde de la caja
    w = ndi.gaussian_filter(sub_m.astype(np.float32), 0.8)
    w = np.clip(w * 1.3, 0, 1)
    out[sl] = sub * (1 - w) + filled * w
    return out.astype(np.float32)


def randomize_anatomy(hu, ves, sem, rng, alpha=4.0, sigma=16.0, flip=True):
    """Convierte al donante en OTRO paciente: el arbol arterial tambien cambia.

    Deformacion elastica con un campo suave aplicado a la imagen y a las etiquetas
    con EL MISMO campo (imagen con interpolacion lineal, etiquetas con vecino mas
    proximo) para que no se descuadren. Mas un reflejo izquierda-derecha opcional,
    que intercambia las etiquetas lateralizadas (R-M1 <-> L-M1).

    Asi, dos llamadas con la misma variante y la misma oclusion no dan la misma
    anatomia.
    """
    if flip and rng.random() < 0.5:
        hu = hu[::-1].copy()
        ves = C.LR_SWAP[ves[::-1]].astype(np.uint8)
        sem = sem[::-1].copy()
        v = (sem >= C.VESSEL0) & (sem <= C.VESSEL0 + C.N_VESSEL - 1)
        sem[v] = C.VESSEL0 + C.LR_SWAP[sem[v] - C.VESSEL0 + 1] - 1

    # Campo de desplazamiento suave (el MISMO para imagen y etiquetas).
    # OJO: hay que RENORMALIZAR tras el filtrado. Un gaussiano de sigma=14 aplasta
    # la amplitud del ruido ~350x, asi que sin esto el desplazamiento es de ~0.02
    # voxeles y la deformacion no hace NADA (Dice 1.000 entre generaciones).
    g = []
    for _ in range(3):
        f = ndi.gaussian_filter(rng.normal(0, 1, hu.shape), sigma).astype(np.float32)
        f *= alpha / (f.std() + 1e-8)          # alpha = desplazamiento tipico, en voxeles
        g.append(f)
    idx = np.indices(hu.shape, dtype=np.float32)
    co = [idx[i] + g[i] for i in range(3)]
    hu2 = ndi.map_coordinates(hu, co, order=1, mode="nearest")
    ves2 = ndi.map_coordinates(ves, co, order=0, mode="nearest").astype(np.uint8)
    sem2 = ndi.map_coordinates(sem, co, order=0, mode="nearest").astype(np.uint8)
    return hu2.astype(np.float32), ves2, sem2


def render(hu, ves0, ves, thr, jitter=True, rng=None):
    """CTA real -> CTA con el arbol editado."""
    rng = rng or np.random.default_rng()
    lost = (ves0 > 0) & (ves == 0)          # vasos que pierden el contraste
    out = suppress_lumen(hu, lost, rng=rng)
    if thr is not None and np.any(thr):
        # coagulo: hiperdenso frente al parenquima pero muy por debajo del
        # contraste, con grano propio y el borde fundido con el vaso vecino
        clot = THROMBUS_HU + rng.normal(0, 5.0, hu.shape).astype(np.float32)
        clot = ndi.gaussian_filter(clot, 0.6)
        w = ndi.gaussian_filter(thr.astype(np.float32), 0.8)
        w = np.clip(w * 1.2, 0, 1)
        out = out * (1 - w) + clot * w
    return out.astype(np.float32)


def donors(sets=("topaneu", "topcow")):
    """Todos los donantes disponibles.

    TopAneu (58) trae el arbol con GT de 36 clases; TopCoW (125) lo trae
    pseudo-etiquetado, pero su IMAGEN es igual de real. Usar los dos da 183
    anatomias, x2 con el reflejo L-R.
    """
    out = []
    for ds in sets:
        out += [(ds, p) for p in sorted((ROOT / "sem" / ds).glob("*.npz"))]
    return out


def generate_hybrid(variant="CoW completo", artery=None, frac=None, seed=None,
                    out=None, randomize=True, sets=("topaneu", "topcow")):
    rng = np.random.default_rng(seed)
    cand = donors(sets)
    rng.shuffle(cand)
    case = sem = ves = aff = src = None
    for ds, p in cand:
        d = np.load(p)
        v = d["ves"]
        if variant not in E.available_variants(v):
            continue
        if artery and artery not in E.occludable(v):
            continue
        case, sem, ves, aff, src = p.stem, d["sem"], v, d["affine"], ds
        break
    if case is None:
        raise RuntimeError(f"ningun donante compatible con {variant} + {artery}")

    img = np.load(ROOT / "prep" / src / f"{case}.npz")["img"].astype(np.float32)
    hu = C.to_hu(img)

    # el arbol arterial tambien es aleatorio: cada llamada es otro paciente
    if randomize:
        hu, ves, sem = randomize_anatomy(hu, ves, sem, rng)
        if artery and artery not in E.occludable(ves):
            raise RuntimeError(f"{artery} se perdio al deformar; reintenta con otra semilla")

    ves2, vinfo = E.apply_variant(ves, variant)
    thr, oinfo = None, {}
    if artery:
        if frac is None:
            frac = float(rng.uniform(0.35, 0.9))
        ves2, thr, oinfo = E.occlude(ves2, artery, frac=frac, rng=rng)

    hu2 = render(hu, ves, ves2, thr, rng=rng)

    # el nombre lleva donante y semilla: si no, dos pacientes distintos con la
    # misma orden se sobrescriben el uno al otro
    tag = variant.replace(" ", "-") + (f"_ocl-{artery}-f{frac:.2f}" if artery else "_sano")
    sid = "s" + (str(seed) if seed is not None else "rnd")
    out = Path(out or ROOT / "out" / f"hybrid_{tag}_{case}_{sid}.nii.gz")
    out.parent.mkdir(parents=True, exist_ok=True)
    C.save_nifti(out, hu2.astype(np.float32), aff)
    # el mapa de vasos editado, para saber que esta ocluido en cada caso
    C.save_nifti(str(out).replace(".nii.gz", "_vasos.nii.gz"),
                 ves2.astype(np.int16), aff)

    v = (ves2 > 0)
    lost = (ves > 0) & (ves2 == 0)
    info = {"donante": case, "variante": variant, "cambios": vinfo,
            "oclusion": oinfo, "salida": str(out),
            "HU_vaso_permeable": round(float(hu2[v].mean()), 0) if v.any() else None,
            "HU_vaso_apagado": round(float(hu2[lost].mean()), 0) if lost.any() else None,
            "HU_trombo": round(float(hu2[thr].mean()), 0) if thr is not None and thr.any() else None}
    print(json.dumps(info, indent=1, ensure_ascii=False))
    return hu2, ves2, thr, info


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--variant", default="CoW completo")
    ap.add_argument("--artery", default=None)
    ap.add_argument("--frac", type=float, default=None)
    ap.add_argument("--seed", type=int, default=None)
    ap.add_argument("--out", default=None)
    a = ap.parse_args()
    generate_hybrid(a.variant, a.artery, a.frac, a.seed, a.out)
