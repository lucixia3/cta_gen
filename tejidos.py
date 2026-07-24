"""Parte la clase SOFT en encefalo / extracraneal / LCR.

El mapa semantico de `common.semantic_map` tiene un problema de fondo: **el 92.6%
del volumen es una unica clase** (`SOFT`), que mezcla cuero cabelludo, musculo
temporal, encefalo, surcos y ventriculos. El generador tiene que adivinar de
memoria que pinta en cada sitio, y de ahi sale el parenquima granulado.

Aqui se derivan tres clases mas, **sin anotar nada a mano**, de la propia imagen:

    ENCEFALO      dentro del craneo
    EXTRACRANEAL  fuera del craneo (cuero cabelludo, musculo, orbita, globo ocular)
    LCR           dentro del craneo y por debajo de LCR_HU (ventriculos, cisternas)

(El aire NO se trata aqui: es la clase `AIR` de `sem`, que estaba muerta por el
umbral de -300 y se arregla en `common.semantic_map`. Ver `reparar_aire`.)

Se entregan en un mapa APARTE (`tis`), no sustituyendo a `sem`. Es deliberado: el
tensor que entra por SPADE es **multi-hot**, no one-hot — un voxel de ventriculo
es a la vez `SOFT` (canal 1, el que el modelo ya tiene aprendido) y `LCR` (canal
nuevo, inicializado a cero). Asi el warm-start desde la epoca 107 es EXACTO: con
los canales nuevos a cero el modelo se comporta bit a bit como antes, y aprende a
usarlos encima de lo que ya sabia en vez de tener que reaprenderlo.

## Como se separa dentro de fuera

Tres cosas que la version ingenua hacia mal, todas vistas en el QC:

- **El craneo no es un anillo cerrado en todos los cortes** (base craneal, orbitas,
  agujero magno), asi que rellenar corte a corte no vale. Se cierra el hueso en 3D
  para tapar forámenes, suturas y hueso trabeculado poroso.
- **Aun cerrado, la componente conexa se escapa a las orbitas** por el canal optico
  y la hendidura esfenoidal: los globos oculares salian etiquetados como encefalo.
  Se corta con una APERTURA (erosion -> componente conexa -> dilatacion): el cuello
  de esas comunicaciones es de 1-2 mm y se rompe, mientras que el encefalo, que es
  macizo, sobrevive a la erosion.
- **El umbral de LCR sobre la imagen cruda captura ruido**, no liquido: en los casos
  de TopCoW, mas ruidosos, salia moteado verde por todo el parenquima. Se filtra la
  mediana antes de umbralizar y se descartan las componentes menores de LCR_MIN_VOX.
"""
from pathlib import Path

import numpy as np
from scipy import ndimage as ndi

import common as C

_ROOT = Path(__file__).parent

# ids nuevos, DETRAS de los 41 existentes: asi los canales 0..40 conservan su
# significado y el checkpoint viejo se copia tal cual (ver model.warm_start).
NONE, ENCEFALO, EXTRA, LCR, AIRE = 0, 1, 2, 3, 4
N_TIS = 5                                  # 4 canales utiles + el 0 (nada)
N_CLASSES_IN = C.N_CLASSES + N_TIS - 1     # 41 + 4 = 45 canales de entrada

TIS_NAMES = {ENCEFALO: "encefalo", EXTRA: "extracraneal", LCR: "lcr", AIRE: "aire"}

LCR_HU = 18.0        # parenquima ~30-40 HU en CTA; ventriculo y cisternas <20
LCR_MEDIANA = 5      # filtro previo al umbral (ver barrido abajo)
LCR_MIN_VOX = 150    # componentes menores = ruido, no liquido
CIERRE = 3           # cierre del craneo, en voxeles de 0.6 mm (~1.8 mm)
APERTURA = 7         # corta el canal optico, la hendidura esfenoidal y el agujero magno

# Barrido medido sobre topaneu_center4_ct_002 (fuga orbitaria) y topcow_ct_001
# (moteado), % del volumen:
#
#   apertura  mediana  min_vox | encefalo  extracraneal   lcr   coste
#      3         3        40   |   49.30       5.08      6.32    3.4 s
#      5         3        40   |   47.70       8.19      4.81    4.6 s
#      5         5       150   |   48.10       8.19      4.41   10.2 s
#      7         5       150   |   42.35      17.15      1.19   15.3 s   <- elegido
#
# Los 6 puntos que apertura=7 mueve de encefalo a extracraneal NO son cerebro
# perdido: son el musculo temporal, la fosa infratemporal y el tejido del cuello,
# que con apertura menor entraban por el agujero magno y salian etiquetados como
# encefalo. Se ve en el corte coronal: con 7 la mitad inferior pasa a naranja, que
# es lo correcto. El cerebelo y la corteza sobreviven a la erosion (son macizos) y
# la dilatacion posterior los recupera enteros.
#
# El filtro de mediana 5 + min_vox 150 es lo que convierte el LCR de moteado a
# estructura: en TopCoW, mas ruidoso, el umbral crudo sembraba verde por todo el
# parenquima; con el filtro quedan los ventriculos y las cisternas como blobs
# coherentes. Cuesta 5 s por caso y merece la pena.


def _bola(r):
    z, y, x = np.ogrid[-r:r + 1, -r:r + 1, -r:r + 1]
    return (z * z + y * y + x * x) <= r * r


def _semilla_cow(sem):
    """Centroide del poligono de Willis: intracraneal por construccion del slab."""
    v = (sem >= C.VESSEL0) & (sem <= C.VESSEL0 + C.N_VESSEL - 1)
    core = np.isin(sem.astype(np.int16) - C.VESSEL0 + 1, C.COW_CORE) & v
    m = core if core.sum() > 50 else v
    if m.sum() == 0:
        return None
    return tuple(np.array(np.nonzero(m)).mean(axis=1).round().astype(int))


def intracraneal(sem, aire, semilla=None):
    """Mascara del espacio intracraneal (hueso y aire excluidos)."""
    hueso = sem == C.BONE
    cerrado = ndi.binary_closing(hueso, structure=_bola(CIERRE))
    dentro = ~cerrado & ~aire

    if semilla is None:
        semilla = _semilla_cow(sem)
    if semilla is None:
        return np.zeros_like(hueso)

    # apertura: erosionar rompe los cuellos finos (canal optico, hendidura
    # esfenoidal) antes de elegir la componente, y luego se recupera el volumen.
    b = _bola(APERTURA)
    nucleo = ndi.binary_erosion(dentro, structure=b)
    lab, n = ndi.label(nucleo)
    if n == 0:
        return dentro
    l0 = lab[semilla]
    if l0 == 0:                       # la semilla cayo en un voxel erosionado
        _, idx = ndi.distance_transform_edt(lab == 0, return_indices=True)
        l0 = lab[tuple(i[semilla] for i in idx)]
    if l0 == 0:
        return dentro
    return ndi.binary_dilation(lab == l0, structure=b) & dentro


def _lcr(hu, intra):
    """LCR: umbral sobre la imagen filtrada + limpieza de componentes minusculas."""
    suave = ndi.median_filter(hu, size=LCR_MEDIANA)
    m = intra & (suave < LCR_HU)
    lab, n = ndi.label(m)
    if n == 0:
        return m
    cuenta = np.bincount(lab.ravel())
    cuenta[0] = 0
    return np.isin(lab, np.nonzero(cuenta >= LCR_MIN_VOX)[0])


def derivar(hu, sem, semilla=None):
    """(HU, mapa semantico) -> mapa de tejido `tis` (uint8, 0..4).

    Solo se etiquetan los voxeles que hoy son SOFT: hueso, vasos y trombo
    conservan su clase en `sem` y aqui van a NONE. El AIRE va aqui y NO en `sem`
    (ver el umbral muerto en `common.AIR_HU`): asi el canal 1 sigue encendido
    donde el modelo espera y la informacion nueva entra por un canal a cero.
    """
    tis = np.zeros(sem.shape, np.uint8)
    soft = sem == C.SOFT
    aire = hu <= C.AIR_HU
    intra = intracraneal(sem, aire, semilla)
    tis[soft & intra] = ENCEFALO
    tis[soft & ~intra] = EXTRA
    tis[soft & _lcr(hu, intra)] = LCR
    tis[soft & aire] = AIRE               # ultimo: manda sobre EXTRA
    return tis


def resumen(tis):
    n = tis.size
    return {v: round(100.0 * float((tis == k).sum()) / n, 2)
            for k, v in TIS_NAMES.items()}


# --- CLI: derivar el tejido de los 209 mapas ----------------------------------
#   python tejidos.py --workers 6
# Anade la clave `tis` a cada sem/*/*.npz y de paso repara la clase AIR. Escribe
# a un temporal y renombra: si se interrumpe no deja un npz a medias (regenerarlos
# costaria volver a pasar el segmentador en GPU).

def _procesar(args):
    import os
    ds, f = args
    root = _ROOT
    try:
        d = np.load(f)
        hu = C.to_hu(np.load(root / "prep" / ds / f.name)["img"].astype(np.float32))
        # `sem` tiene que quedar EXACTAMENTE como lo dejo pseudolabel.py: si una
        # pasada anterior escribio la clase AIR, se revierte a SOFT (la reparacion
        # solo hacia SOFT->AIR, asi que es reversible). El aire vive en `tis`.
        sem = d["sem"].copy()
        sem[sem == C.AIR] = C.SOFT
        tis = derivar(hu, sem)
        # OJO: el temporal TIENE que acabar en .npz. `np.savez_compressed` anade
        # ".npz" a cualquier ruta que no lo lleve ya, asi que un ".npz.tmp" acaba
        # escrito como ".npz.tmp.npz" y el renombrado falla.
        tmp = f.parent / (f.stem + ".tmp.npz")
        np.savez_compressed(tmp, sem=sem, ves=d["ves"], tis=tis,
                            affine=d["affine"], src=d["src"])
        d.close()
        os.replace(tmp, f)
        return f.stem, {"aire": round(100.0 * float((sem == C.AIR).mean()), 2),
                        **resumen(tis)}
    except Exception as e:                      # noqa: BLE001
        return f.stem, {"error": f"{type(e).__name__}: {e}"}


def _main():
    import json
    import argparse
    from pathlib import Path
    from concurrent.futures import ProcessPoolExecutor

    ap = argparse.ArgumentParser()
    ap.add_argument("--workers", type=int, default=6)
    ap.add_argument("--ds", nargs="*", default=["topaneu", "topcow", "isles"])
    a = ap.parse_args()

    tareas = [(ds, f) for ds in a.ds
              for f in sorted((_ROOT / "sem" / ds).glob("*.npz"))]
    print(f"{len(tareas)} mapas, {a.workers} workers", flush=True)

    rep, malos = {}, 0
    with ProcessPoolExecutor(a.workers) as ex:
        for k, (name, r) in enumerate(ex.map(_procesar, tareas), 1):
            rep[name] = r
            if "error" in r:
                malos += 1
                print(f"  [{k}/{len(tareas)}] {name}  {r['error']}", flush=True)
            elif k % 20 == 0 or k == len(tareas):
                print(f"  [{k}/{len(tareas)}] {name}  {r}", flush=True)

    (_ROOT / "sem" / "tejidos.json").write_text(json.dumps(rep, indent=1))
    ok = [r for r in rep.values() if "error" not in r]
    if ok:
        for c in ("aire", "encefalo", "extracraneal", "lcr"):
            v = np.array([r[c] for r in ok])
            print(f"  {c:14s} mediana {np.median(v):5.2f}%  "
                  f"rango [{v.min():.2f}, {v.max():.2f}]")
    print(f"\n{len(ok)} ok, {malos} con error -> sem/tejidos.json")


if __name__ == "__main__":
    _main()
