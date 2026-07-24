"""Corrige las etiquetas THROMBUS de ISLES, que estaban puestas sobre contraste.

## El fallo

`pseudolabel.detect_occlusion` localiza la arteria ocluida por asimetria y luego
marca el trombo asi:

    m = ves == E.V[prox]              # prox = ultimo segmento CON contraste
    gd = E.geodesic(m, E._seed_of(ves, prox))
    thr = valid & (gd > dmax - 4mm)   # sus ultimos 4 mm

Es decir, etiqueta los ultimos 4 mm del segmento **permeable**. Por construccion
esos voxeles estan DENTRO de la columna de contraste -- se segmentaron como vaso
justamente porque estan brillantes. El coagulo real empieza donde terminan.

Medido sobre los 17 casos ISLES con oclusion localizada (1011 voxeles):

    mediana 185.2 HU  |  65.7% por encima de 150 HU
    peor caso sub-stroke_0014_ct: mediana 250.2 HU

Y `train_gen.py` les da peso x5 en la perdida. O sea que los unicos ejemplos
reales de coagulo que ha visto el generador le ensenaban que un trombo son ~185
HU. Eso explica que los pintara a ~174 en vez de a ~58, sin que fuera cuestion de
ver mas ejemplos.

## La correccion

Esos voxeles SON lumen con contraste, asi que su etiqueta correcta es el vaso al
que pertenecen. Y no hay que adivinarlo: `ves` se guardo intacto en el mismo npz
(el trombo solo se aplico sobre `sem`), asi que la etiqueta buena esta ahi.

    sem[thr] = C.VESSEL0 + ves[thr] - 1

El generador sigue aprendiendo de ISLES lo que ISLES aporta de verdad -- el
aspecto de un cut-off real, que es la AUSENCIA de contraste distal -- y el trombo
lo aprende de las oclusiones sinteticas de `oclusiones.py`, que se pintan a
THROMBUS_HU por construccion.

Uso: python arreglar_trombo.py [--dry]
"""
import argparse
import shutil
from pathlib import Path

import numpy as np

import common as C

ROOT = Path(__file__).parent


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dry", action="store_true", help="solo informar, no escribir")
    ap.add_argument("--ds", default="isles")
    a = ap.parse_args()

    files = sorted((ROOT / "sem" / a.ds).glob("*.npz"))
    tot_vox, tocados = 0, 0
    print(f"{'caso':<24} {'vox':>6} {'HU antes':>9}  accion")
    for f in files:
        d = dict(np.load(f))
        sem, ves = d["sem"], d["ves"]
        m = sem == C.THROMBUS
        if not m.any():
            continue

        hu = C.to_hu(np.load(ROOT / "prep" / a.ds / f.name)["img"].astype(np.float32))
        med = float(np.median(hu[m]))

        # los que no tienen vaso debajo no se pueden reetiquetar asi; no deberia
        # pasar (thr es un subconjunto del segmento proximal) pero se comprueba.
        sin = m & (ves == 0)
        if sin.any():
            print(f"{f.stem:<24} {m.sum():6d} {med:9.1f}  AVISO: {sin.sum()} vox "
                  f"sin vaso debajo -> a SOFT")

        sem2 = sem.copy()
        sem2[m & (ves > 0)] = (C.VESSEL0 + ves[m & (ves > 0)].astype(np.int16) - 1)
        sem2[sin] = C.SOFT

        tot_vox += int(m.sum())
        tocados += 1
        print(f"{f.stem:<24} {m.sum():6d} {med:9.1f}  -> vaso "
              f"{sorted(set(np.unique(ves[m]).tolist()))}")

        if not a.dry:
            bak = f.with_suffix(".npz.bak")
            if not bak.exists():
                shutil.copy2(f, bak)
            d["sem"] = sem2.astype(sem.dtype)
            np.savez_compressed(f, **d)

    print(f"\n{tocados} casos, {tot_vox} voxeles reetiquetados"
          f"{'  (DRY: no se ha escrito nada)' if a.dry else '  (copias en *.npz.bak)'}")


if __name__ == "__main__":
    main()
