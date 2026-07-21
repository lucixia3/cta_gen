"""QC de un caso: ¿el infarto del NCCT cae en el lado del vaso ocluido del CTA?

Es LA comprobacion que importa, y la mas facil de romper sin enterarse: el CTA vive
en el slab canonico del donante y el NCCT/atlas en MNI. Si el puente se equivoca de
lado, el infarto sale en el hemisferio contrario a la oclusion -- y el caso *parece*
perfecto en cualquier visor, porque cada volumen por separado es plausible.

OJO con como se mide el lado. En MNI, x = 0 ES la linea media. En el slab canonico NO:
el grid se centra en el centroide del poligono de Willis (`common.cow_center`), no en
la linea media anatomica, asi que el paciente entero puede estar desplazado 20 mm en x.
Medir "derecha = x > 0" ahi da falsos positivos de lado invertido. La linea media del
donante se estima con el centroide de todo el arbol arterial, que es simetrico.

    python qc_caso.py casos/B_sin_acom
"""
import sys
import json
from pathlib import Path

import numpy as np
import nibabel as nib

sys.path.insert(0, str(Path(__file__).parent))
import common as C
import cow_edit as E

ROOT = Path(__file__).parent
PARES = ["ICA-C6-C7", "M1", "M2", "A1A2", "P1P2", "VA", "ICA-C1-C5"]


def cx(mask, affine):
    """Coordenada x (mundo) del centroide de `mask`."""
    if not mask.any():
        return None
    idx = np.array(np.nonzero(mask)).mean(axis=1)
    return float((affine[:3, :3] @ idx + affine[:3, 3])[0])


def linea_media(ves, aff):
    """x de la linea media: media de los puntos medios de cada par R/L.

    NO vale el centroide del arbol: el pseudo-etiquetado no cubre igual los dos lados y
    lo corre varios mm -- lo bastante para que un M1 perfectamente colocado caiga del
    lado equivocado y el QC de un falso positivo. Tampoco vale la simetria de la cabeza:
    el slab canonico esta DENTRO del craneo y no hay borde que centrar. Y x=0 tampoco es
    la linea media, que el grid se centra en el poligono.
    """
    mids = []
    for p in PARES:
        r, l = cx(ves == E.V[f"R-{p}"], aff), cx(ves == E.V[f"L-{p}"], aff)
        if r is not None and l is not None:
            mids.append((r + l) / 2.0)
    return float(np.mean(mids)) if mids else cx(ves > 0, aff)


def lado_de(ves, aff, art):
    """Lado de `art`, comparandola con su par contralateral cuando existe.

    Es la medida invariante: no depende de donde este el origen ni de lo asimetrico que
    sea el etiquetado. Si el par no existe (lo ha borrado la variante), se cae a la linea
    media estimada.
    """
    a = cx(ves == E.V[art], aff)
    if a is None:
        return None, None
    par = ("L-" if art.startswith("R-") else "R-") + art[2:]
    b = cx(ves == E.V[par], aff) if par[2:] in PARES else None
    ref = b if b is not None else linea_media(ves, aff)
    return ("DERECHA" if a > ref else "IZQUIERDA"), a


def coherencia_lr(ves, aff):
    """¿Cada 'R-X' esta de verdad a la derecha de su 'L-X'? (invariante al desplazamiento)"""
    ok, mal, sin = [], [], []
    for p in PARES:
        r, l = cx(ves == E.V[f"R-{p}"], aff), cx(ves == E.V[f"L-{p}"], aff)
        if r is None or l is None:
            sin.append(p)
        elif r > l:
            ok.append(p)
        else:
            mal.append(f"{p} (R={r:+.1f} L={l:+.1f})")
    return ok, mal, sin


def main(d):
    d = Path(d)
    info = json.loads((d / "caso.json").read_text(encoding="utf-8"))
    art = info["arteria"]

    # el arbol sano DE ESTE CASO, no el del banco: la anatomia va reflejada y deformada
    sano = nib.load(d / "vasos_sano.nii.gz")
    vsano, vaff = np.asanyarray(sano.dataobj), sano.affine
    ves2 = np.asanyarray(nib.load(d / "vasos.nii.gz").dataobj)
    mim = nib.load(d / "mask.nii.gz")
    mask = np.asanyarray(mim.dataobj) > 0

    print(f"{d.name}: {info['variante']} + {art}")
    print(f"  territorios: {info['territorios'] or 'ninguno'}  ({info['volumen_infarto_ml']} ml)")

    ok, mal, sin = coherencia_lr(vsano, vaff)
    print(f"  coherencia R/L del arbol: {len(ok)} pares OK, {len(mal)} invertidos"
          + (f"  *** {mal} ***" if mal else ""))

    lm = linea_media(vsano, vaff)
    veredicto = None
    if art and not art.startswith(("BA", "3rd")):
        lado_occ, x = lado_de(vsano, vaff, art)
        print(f"  vaso ocluido {art:12s} x={x:+6.1f}  (linea media {lm:+.1f}) -> {lado_occ}")

        if mask.any():
            xi = cx(mask, mim.affine)          # MNI: aqui x=0 SI es la linea media
            lado_inf = "DERECHA" if xi > 0 else "IZQUIERDA"
            print(f"  infarto (MNI)             x={xi:+6.1f}  (linea media   0.0) -> {lado_inf}")
            esperado = "DERECHA" if art.startswith("R-") else "IZQUIERDA"
            veredicto = (lado_occ == esperado and lado_inf == esperado)
            print(f"  esperado {esperado}  ->  "
                  + ("OK: oclusion e infarto en el mismo lado" if veredicto
                     else "*** LADO EQUIVOCADO ***"))
        else:
            print("  infarto: NINGUNO (el CoW rescata el arbol distal)")

    if mask.any():
        ncct = np.asanyarray(nib.load(d / "ncct_infarto.nii.gz").dataobj)
        hu_i, hu_c = float(ncct[mask].mean()), float(ncct[mask[::-1]].mean())
        print(f"  HU infarto {hu_i:5.1f} | contralateral {hu_c:5.1f} | delta {hu_i - hu_c:+5.1f}")

    n0, n1 = int((vsano > 0).sum()), int((ves2 > 0).sum())
    print(f"  arbol: {n0} vox con contraste -> {n1}  ({100*(n0-n1)/max(n0,1):.0f}% apagado)")
    return veredicto


if __name__ == "__main__":
    res = [main(a) for a in (sys.argv[1:] or ["casos/B_sin_acom"])]
    print()
    malos = [i for i, r in enumerate(res) if r is False]
    print(f"{sum(1 for r in res if r)} casos con lateralidad OK, {len(malos)} equivocados")
