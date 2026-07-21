"""Genera un lote de casos (CTA + NCCT) barriendo variantes de CoW y sitios de oclusion.

Cada caso va a su propia subcarpeta y todo se resume en `manifest.csv` (la verdad de
terreno: variante, arteria, que se apago, que territorio se infarto, volumen, HU).

    python batch_casos.py                       # la rejilla de abajo -> dataset_casos/
    python batch_casos.py --out mis_casos --fuerza 0.8

Cada caso cae en una ANATOMIA DISTINTA a proposito: `pid=None` deja que `caso.candidatos`
elija donante dentro de la familia de CoW, y la `seed` unica por caso (el indice) hace que
el reflejo + la deformacion elastica no se repitan. Asi el arbol arterial cambia de un caso
a otro, no solo la variante y el sitio de oclusion.

    OJO con el techo real: el caso completo (CTA + NCCT) solo puede usar el banco de 8
    donantes -- el pool de 183 no trae el NCCT+atlas que necesita el territorio del infarto
    -- y la familia CoW 0 (CoW completo / Acom / Pcom) tiene solo 2. La variedad de esos
    casos sale de 2 donantes x reflejo x deformacion, que es cuanto da el banco. Si hace
    falta un arbol distinto de verdad por caso sin NCCT, ese es el lote de `batch_cta.py`
    (183 donantes).

Antes esta rejilla FIJABA `pat_cow0_00` en casi todo para poder comparar pares que solo
cambiaban en la variante (mismo arbol): esa comparacion se pierde a cambio de la variedad.
"""
import csv
import sys
import argparse
import traceback
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
import caso

ROOT = Path(__file__).parent

# (variante, arteria, frac, nombre)
# frac = donde cae el trombo a lo largo del segmento (0 proximal, 1 distal)
#
# La anatomia ya NO va fija: cada caso elige donante (pid=None) y usa una seed unica (su
# indice en main), asi que el arbol arterial cambia de un caso a otro. Se mantiene el
# barrido clinico -- variantes de CoW x sitios de oclusion cubriendo circulacion anterior,
# posterior y las oclusiones asintomaticas -- pero ya no como pares sobre el mismo arbol.
REJILLA = [
    # --- circulacion anterior: ICA en el sifon segun la variante ---------------
    ("CoW completo",             "R-ICA-C6-C7", 0.90, "01_ICA-der_CoW-completo"),
    ("Acom ausente",             "R-ICA-C6-C7", 0.90, "02_ICA-der_SIN-Acom"),
    ("Acom ausente",             "L-ICA-C6-C7", 0.90, "03_ICA-izq_SIN-Acom"),
    # --- ICA con el trombo PROXIMAL al Pcom: el Pcom la rellena hacia atras
    ("CoW completo",             "R-ICA-C6-C7", 0.25, "04_ICA-der_proximal-al-Pcom"),
    ("Pcom bilateral ausente",   "R-ICA-C6-C7", 0.25, "05_ICA-der_proximal_SIN-Pcom"),
    # --- M1 vs M2: el M1 se lleva las perforantes profundas, el M2 las respeta
    ("CoW completo",             "R-M1",        0.30, "06_M1-der_proximal"),
    ("CoW completo",             "R-M2",        0.50, "07_M2-der"),
    ("CoW completo",             "L-M1",        0.30, "08_M1-izq_proximal"),
    ("A1 hipoplasica izquierda", "L-M1",        0.40, "09_M1-izq_A1-izq-hipoplasica"),
    # --- ACA
    ("CoW completo",             "R-A1A2",      0.50, "10_A1A2-der"),
    ("A1 derecha ausente",       "L-A1A2",      0.25, "11_A1A2-izq_SIN-A1-der"),
    # --- circulacion posterior
    ("CoW completo",             "BA",          0.60, "12_basilar_CoW-completo"),
    ("Pcom bilateral ausente",   "BA",          0.60, "13_basilar_SIN-Pcom"),
    ("CoW completo",             "R-P1P2",      0.50, "14_P1P2-der"),
    ("PCA fetal derecha",        "R-P1P2",      0.50, "15_P1P2-der_PCA-fetal"),
    ("CoW completo",             "R-VA",        0.50, "16_vertebral-der"),
    # --- la oclusion asintomatica: carotida cervical con el CoW competente
    ("CoW completo",             "R-ICA-C1-C5", 0.50, "17_carotida-cervical"),
    # --- contrafactual sano: sin ocluir nada
    ("CoW completo",             None,          None, "18_sano"),
]

COLS = ["caso", "variante", "arteria", "frac", "anatomia", "familia_cow",
        "apagados", "ramas_rescatadas", "relleno_retrogrado",
        "territorios", "volumen_infarto_ml",
        "HU_vaso_permeable", "HU_vaso_apagado", "HU_trombo",
        "HU_infarto", "HU_contralateral"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="dataset_casos")
    ap.add_argument("--fuerza", type=float, default=1.5,
                    help="cuanto se ve la hipodensidad (baja a ~0.8 para un infarto mas precoz)")
    a = ap.parse_args()

    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    filas, fallos = [], []

    for k, (var, art, frac, name) in enumerate(REJILLA, 1):
        print(f"\n[{k}/{len(REJILLA)}] {name}", flush=True)
        try:
            # seed = indice del caso: unica por caso, asi el donante elegido y la
            # deformacion no se repiten. pid=None -> anatomia libre dentro de la familia.
            info = caso.generar(variante=var, arteria=art, frac=frac, seed=k,
                                fuerza=a.fuerza, out=out / name, quiet=True, pid=None)
        except Exception:
            fallos.append((name, traceback.format_exc(limit=1).strip().split("\n")[-1]))
            print(f"   FALLO: {fallos[-1][1]}", flush=True)
            continue

        terr = ", ".join(info["territorios"]) or "ninguno"
        print(f"   {info['anatomia']} | apaga: {', '.join(info['apagados']) or 'nada'}"
              f"\n   -> infarto: {terr}  ({info['volumen_infarto_ml']} ml)", flush=True)
        filas.append({
            "caso": name, "variante": var, "arteria": art or "", "frac": frac or "",
            "anatomia": info["anatomia"], "familia_cow": info["familia_cow"],
            "apagados": " ".join(info["apagados"]),
            "ramas_rescatadas": " ".join(info["ramas_rescatadas"]),
            "relleno_retrogrado": info["relleno_retrogrado"],
            "territorios": terr, "volumen_infarto_ml": info["volumen_infarto_ml"],
            **{c: info[c] for c in COLS if c.startswith("HU_")},
        })

    with open(out / "manifest.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=COLS)
        w.writeheader()
        w.writerows(filas)

    print(f"\n{len(filas)}/{len(REJILLA)} casos en {out.resolve()}")
    print(f"  manifest.csv  (verdad de terreno)")
    con = sum(1 for r in filas if r["volumen_infarto_ml"])
    print(f"  {con} con infarto, {len(filas) - con} sin infarto")
    for n, e in fallos:
        print(f"  FALLO {n}: {e}")


if __name__ == "__main__":
    main()
