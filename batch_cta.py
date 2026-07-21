"""Lote de CTA (sin NCCT) sobre el pool GRANDE de donantes: 183 anatomias, no 8.

`batch_casos.py` da CTA + NCCT del mismo paciente, y por eso esta atado al banco de
`bank_sem` (8 donantes usables): es la unica fuente con las dos imagenes del mismo
enfermo. Si el NCCT no hace falta, ese limite desaparece — `sem/topaneu` (58) y
`sem/topcow` (125) traen arbol y CTA real, y son 183 cerebros distintos.

Lo que NO sale de aqui: territorio infartado ni volumen. El puente
`apagados -> territorio` necesita el atlas MNI alineado voxel a voxel con el NCCT del
donante, que solo existe para el banco. En `caso.json` van a `null` — que es distinto
de 0.0 ml (eso significaria "no infarta nada", y aqui es "no se ha calculado").

    python batch_cta.py                          # la rejilla de abajo -> dataset_cta/
    python batch_cta.py --out mis_cta --n 6

Despues, las laminas:

    python arbol3d.py --dir dataset_cta --paso 1
    python mip_caso.py --dir dataset_cta
"""
import csv
import sys
import json
import argparse
import traceback
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent))
import common as C
import cow_edit as E
import hybrid as H
import caso as K          # otro_paciente / ESCALA: la deformacion que NO rompe el grafo
import deface as D        # quita la cara y las coordenadas de mesa del donante real

ROOT = Path(__file__).parent

# (variante, arteria, frac, seed, nombre)
#
# La semilla elige el donante dentro del pool de 183, asi que cada caso cae en un
# cerebro distinto a proposito: es justo lo que este lote demuestra y lo que
# `batch_casos.py` no puede hacer. Para comparar dos casos en el MISMO arbol hay que
# fijar `--pid`, como alli.
REJILLA = [
    ("CoW completo",             "R-ICA-C6-C7", 0.90,  1, "01_ICA-der_CoW-completo"),
    ("Acom ausente",             "R-ICA-C6-C7", 0.90,  2, "02_ICA-der_SIN-Acom"),
    ("Pcom bilateral ausente",   "R-ICA-C6-C7", 0.25,  3, "03_ICA-der_proximal_SIN-Pcom"),
    ("CoW completo",             "R-M1",        0.30,  4, "04_M1-der_proximal"),
    ("CoW completo",             "R-M2",        0.50,  5, "05_M2-der"),
    ("CoW completo",             "L-M1",        0.30,  6, "06_M1-izq_proximal"),
    ("A1 hipoplasica izquierda", "L-M1",        0.40,  7, "07_M1-izq_A1-izq-hipoplasica"),
    ("CoW completo",             "L-A1A2",      0.30,  8, "08_A1A2-izq"),
    ("CoW completo",             "BA",          0.60,  9, "09_basilar_CoW-completo"),
    ("Pcom bilateral ausente",   "BA",          0.60, 10, "10_basilar_SIN-Pcom"),
    ("PCA fetal derecha",        "R-P1P2",      0.50, 11, "11_P1P2-der_PCA-fetal"),
    ("CoW completo",             "R-VA",        0.50, 12, "12_vertebral-der"),
    ("CoW completo",             "R-ICA-C1-C5", 0.50, 13, "13_carotida-cervical"),
    ("CoW completo",             None,          None, 14, "14_sano"),
]

COLS = ["caso", "variante", "arteria", "frac", "donante", "dataset", "espejo", "deform",
        "comunicantes_donante", "apagados", "relleno_retrogrado",
        "HU_vaso_permeable", "HU_vaso_apagado", "HU_trombo"]


COMUNICANTES = ("Acom", "R-Pcom", "L-Pcom")


def comunicantes(ves):
    """Que comunicantes trae ETIQUETADAS el donante. Son las unicas aristas
    bidireccionales del grafo: las que rescatan territorios."""
    return [n for n in COMUNICANTES if int((ves == E.V[n]).sum()) > 0]


def donantes(sets=("topaneu", "topcow")):
    """Los 183: (pid, dataset, ruta del sem)."""
    out = []
    for ds in sets:
        out += [(p.stem, ds, p) for p in sorted((ROOT / "sem" / ds).glob("*.npz"))]
    return out


def candidatos(variante, arteria, seed, pid=None, sets=("topaneu", "topcow")):
    """Donantes que admiten la variante y tienen la arteria, en orden aleatorio.

    Se devuelve la lista entera, no el primero: `occludable` filtra por volumen de vaso
    pero no por longitud, y `occlude` ademas exige una geodesica minima. Si el primero
    no vale hay que pasar al siguiente, no abortar el caso (mismo motivo que en
    `caso.candidatos`).

    OJO con "CoW completo": es la variante PEDIDA, no una descripcion del donante.
    `apply_variant` no borra nada, pero tampoco garantiza que el poligono este entero, y
    en este pool solo lo esta en 55 de 183 (30%). Sin ese filtro salia esto: la carotida
    cervical de `topcow_ct_049` -- que no tiene NINGUN Pcom etiquetado -- apagaba el
    hemisferio derecho entero, cuando la cervical con un poligono competente es la
    oclusion asintomatica clasica y no debe apagar nada. El grafo estaba bien; mentia la
    etiqueta. Si el donante no tiene Pcom, el caso ES un "Pcom ausente" se llame como se
    llame, asi que para "CoW completo" se exigen las tres comunicantes.
    """
    cand = donantes(sets)
    if pid is not None:
        cand = [c for c in cand if c[0] == pid] or cand
    rng = np.random.default_rng(seed)
    out, sin_com = [], 0
    for i in rng.permutation(len(cand)):
        p_id, ds, p = cand[i]
        d = np.load(p)
        ves = d["ves"]
        if variante not in E.available_variants(ves):
            continue
        if arteria and arteria not in E.occludable(ves):
            continue
        if variante == "CoW completo" and len(comunicantes(ves)) < len(COMUNICANTES):
            sin_com += 1
            continue
        out.append((p_id, ds, d))
    if not out:
        raise ValueError(f"ningun donante compatible con '{variante}' + {arteria}"
                         + (f" ({sin_com} descartados por polígono incompleto)" if sin_com else ""))
    return out


def generar(variante="CoW completo", arteria=None, frac=None, seed=0, out=None,
            pid=None, variar=True, quiet=False, sets=("topaneu", "topcow"), tope=12):
    """Un caso: CTA con el arbol editado + los mapas. Devuelve el dict de caso.json.

    `tope` limita cuantos donantes se prueban antes de rendirse. Con 183 en el pool,
    recorrerlos todos cuando la arteria no es ocluible en ninguno cuesta minutos por
    caso y no aporta: si no ha salido en 12, no va a salir.
    """
    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(seed)
    if arteria and frac is None:
        frac = float(rng.uniform(0.35, 0.9))

    errs, hallado = [], None
    for pid_, ds, d in candidatos(variante, arteria, seed, pid, sets)[:tope]:
        ves_d = d["ves"]
        img = np.load(ROOT / "prep" / ds / f"{pid_}.npz")["img"].astype(np.float32)
        hu_d = C.to_hu(img)
        # `K.otro_paciente` deforma el arbol por su SDF (no por vecino mas proximo, que
        # es lo que hace `hybrid.randomize_anatomy` y le parte los vasos finos: el grafo
        # necesita conectividad y longitud, no solo pintar).
        for alpha in (K.ESCALA if variar else (0.0,)):
            hu0, ves0, espejo, _ = K.otro_paciente(hu_d, ves_d, seed, pid_, alpha)
            ves_v, cambios = E.apply_variant(ves0, variante)
            if not arteria:
                hallado = (pid_, ds, d, hu0, ves0, espejo, alpha, ves_v, cambios,
                           ves_v, None, {})
                break
            try:
                ves2, thr, oinfo = E.occlude(ves_v, arteria, frac=frac, rng=rng)
            except ValueError as e:
                errs.append(f"{pid_} (alpha {alpha}): {e}")
                continue
            hallado = (pid_, ds, d, hu0, ves0, espejo, alpha, ves_v, cambios,
                       ves2, thr, oinfo)
            break
        if hallado:
            break
    if not hallado:
        raise ValueError(f"ningun donante puede ocluir {arteria}: " + "; ".join(errs[:4]))

    (pid_, ds, d, hu, ves0, espejo, alpha, ves_v, cambios, ves2, thr, oinfo) = hallado
    if not quiet:
        print(f"   {pid_} ({ds}{', espejo' if espejo else ''}, deform {alpha})", flush=True)

    hu2 = H.render(hu, ves0, ves2, thr, rng=rng)

    # De-identificacion. El sustrato sigue siendo un donante REAL (espejado + deformado),
    # con su cara y su craneo: se quita la cara -- todo lo anterior al cerebro, sin tocar
    # ningun vaso -- y se re-centra la affine para borrar la posicion de mesa del escaner.
    # `aff` pasa a ser la canonica-al-origen y se aplica a TODAS las salidas (comparten
    # espacio, no deben descuadrarse). La mascara de lo borrado se guarda aparte.
    hu2, cara = D.deface_head(hu2, ves2)
    aff = D.deid_affine()

    # Igual que en `caso.py`: `apagados` incluye segmentos que ya nacian huerfanos del
    # pseudo-etiquetado (un 3rd-A3 sin su 3rd-A2 nunca fue alcanzable). Solo cuenta lo
    # que pasa de perfundido a NO perfundido.
    base = E.reachable(ves_v)
    apagados = [s for s in oinfo.get("apagados", []) if s in base]
    huerfanos = [s for s in oinfo.get("apagados", []) if s not in base]

    C.save_nifti(out / "cta_oclusion.nii.gz", hu2.astype(np.float32), aff)
    C.save_nifti(out / "vasos.nii.gz", ves2.astype(np.int16), aff)
    C.save_nifti(out / "vasos_sano.nii.gz", ves_v.astype(np.int16), aff)
    C.save_nifti(out / "trombo.nii.gz",
                 (thr if thr is not None else np.zeros(ves2.shape, bool)).astype(np.uint8), aff)
    # la region de cara borrada, para que la de-identificacion sea auditable/reversible
    C.save_nifti(out / "deface_mask.nii.gz", cara.astype(np.uint8), aff)

    perm = ves2 > 0
    lost = (ves_v > 0) & (ves2 == 0)
    info = {
        "donante": pid_, "dataset": ds, "espejo": espejo, "deform": alpha,
        "variante": variante, "arteria": arteria, "frac": frac, "seed": seed,
        "defaced": True, "deface_voxeles": int(cara.sum()), "affine_recentrada": True,
        "cambios_variante": cambios,
        # lo que el donante trae DE VERDAD, no lo que se pidio: en este pool el
        # pseudo-etiquetado se deja comunicantes, y sin ellas el grafo no rescata nada
        "comunicantes_donante": comunicantes(ves0),
        "apagados": apagados, "huerfanos_pseudolabel": huerfanos,
        "ramas_rescatadas": oinfo.get("rescatadas", []),
        "relleno_retrogrado": oinfo.get("distal_retrogrado"),
        # sin NCCT no hay atlas MNI, asi que no hay territorio. null != 0.0 ml.
        "territorios": None, "volumen_infarto_ml": None,
        "HU_vaso_permeable": round(float(hu2[perm].mean()), 1) if perm.any() else None,
        "HU_vaso_apagado": round(float(hu2[lost].mean()), 1) if lost.any() else None,
        "HU_trombo": round(float(hu2[thr].mean()), 1) if thr is not None and thr.any() else None,
    }
    (out / "caso.json").write_text(json.dumps(info, indent=1, ensure_ascii=False),
                                   encoding="utf-8")
    return info


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="dataset_cta")
    ap.add_argument("--n", type=int, default=None, help="solo los n primeros de la rejilla")
    ap.add_argument("--no-variar", action="store_true")
    a = ap.parse_args()

    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    rej = REJILLA[:a.n] if a.n else REJILLA
    filas, fallos = [], []

    for k, (var, art, frac, seed, name) in enumerate(rej, 1):
        print(f"\n[{k}/{len(rej)}] {name}", flush=True)
        try:
            info = generar(variante=var, arteria=art, frac=frac, seed=seed,
                           out=out / name, variar=not a.no_variar)
        except Exception:
            fallos.append((name, traceback.format_exc(limit=1).strip().split("\n")[-1]))
            print(f"   FALLO: {fallos[-1][1]}", flush=True)
            continue
        print(f"   apaga: {', '.join(info['apagados']) or 'nada'}", flush=True)
        filas.append({
            "caso": name, "variante": var, "arteria": art or "", "frac": frac or "",
            "donante": info["donante"], "dataset": info["dataset"],
            "espejo": info["espejo"], "deform": info["deform"],
            "comunicantes_donante": " ".join(info["comunicantes_donante"]),
            "apagados": " ".join(info["apagados"]),
            "relleno_retrogrado": info["relleno_retrogrado"],
            **{c: info[c] for c in COLS if c.startswith("HU_")},
        })

    with open(out / "manifest.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=COLS)
        w.writeheader()
        w.writerows(filas)

    n_don = len({r["donante"] for r in filas})
    print(f"\n{len(filas)}/{len(rej)} casos en {out.resolve()}  —  {n_don} donantes distintos")
    for n, e in fallos:
        print(f"  FALLO {n}: {e}")


if __name__ == "__main__":
    main()
