"""Genera un dataset de CTA sinteticas: N casos con variante y oclusion al azar.

    python batch.py --n 50 --out dataset
    python batch.py --n 20 --variant "Acom ausente"      # fija la variante
    python batch.py --n 20 --artery R-M1                 # fija el sitio

Cada caso sale con un paciente distinto (donante aleatorio + reflejo L-R +
deformacion elastica) y se guarda con su etiqueta de vasos y un manifest.csv que
dice, para cada archivo, que variante, que arteria y que territorios se apagaron.
"""
import sys
import csv
import json
import argparse
import traceback
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent))
import common as C
import cow_edit as E
import hybrid as H

ROOT = Path(__file__).parent


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=20)
    ap.add_argument("--out", default="dataset")
    ap.add_argument("--variant", default=None, help="si no, aleatoria")
    ap.add_argument("--artery", default=None, help="si no, aleatoria")
    ap.add_argument("--p_sano", type=float, default=0.15,
                    help="fraccion de casos sin oclusion (controles)")
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()

    out = ROOT / a.out
    out.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(a.seed)
    variants = list(E.VARIANTS)

    rows, ok, fail = [], 0, 0
    for i in range(a.n):
        seed = int(rng.integers(1e9))
        var = a.variant or variants[rng.integers(len(variants))]
        if a.artery:
            art = a.artery
        elif rng.random() < a.p_sano:
            art = None                       # control sano
        else:
            art = C.OCCLUDABLE[rng.integers(len(C.OCCLUDABLE))]
        frac = float(rng.uniform(0.3, 0.95)) if art else None
        try:
            hu, ves, thr, info = H.generate_hybrid(var, art, frac, seed=seed,
                                                   out=None)
            # mover a la carpeta del dataset
            src = Path(info["salida"])
            dst = out / src.name
            src.replace(dst)
            sv = Path(str(src).replace(".nii.gz", "_vasos.nii.gz"))
            if sv.exists():
                sv.replace(out / sv.name)
            rows.append({
                "archivo": dst.name,
                "donante": info["donante"],
                "semilla": seed,
                "variante": var,
                "arteria_ocluida": art or "",
                "frac": round(frac, 2) if frac else "",
                "territorios_apagados": ";".join(info["oclusion"].get("apagados", [])),
                "relleno_retrogrado": info["oclusion"].get("distal_retrogrado", ""),
                "HU_vaso_permeable": info["HU_vaso_permeable"],
                "HU_vaso_apagado": info["HU_vaso_apagado"],
            })
            ok += 1
            print(f"[{i+1}/{a.n}] OK {dst.name}", flush=True)
        except Exception as ex:
            fail += 1
            print(f"[{i+1}/{a.n}] FALLO ({var} / {art}): {ex}", flush=True)

    if rows:
        with open(out / "manifest.csv", "w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=list(rows[0]))
            w.writeheader()
            w.writerows(rows)
    print(f"\n{ok} generados, {fail} fallidos -> {out}")
    print(f"manifest: {out / 'manifest.csv'}")


if __name__ == "__main__":
    main()
