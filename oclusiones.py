"""Precalcula RECETAS de oclusion para aumentar el entrenamiento del generador.

El generador no habia visto casi ninguna combinacion variante x oclusion. Aqui se
le fabrican los pares que le faltan (mapa ocluido -> CTA ocluida), que es justo lo
que `hybrid.py` sabe hacer.

Una oclusion, para este generador, es que el vaso DEJA DE VERSE distalmente. NO se
pinta ningun coagulo: no queremos el signo hiperdenso, solo la ausencia de
contraste a partir del punto ocluido. Por eso la receta guarda unicamente que
segmentos se apagan.

## Por que RECETAS y no volumenes

Generar y guardar los volumenes editados costaria ~50 MB por caso y por
combinacion. Pero la edicion es determinista dada la receta, y la receta es
minuscula: la lista de segmentos que se apagan (<=36 ids). El volumen se
reconstruye en el parche, no en el slab entero:

    ves2 = ves.copy(); ves2[isin(ves, apagados)] = 0
    img2 = hybrid.render(hu, ves, ves2, None)    # sobre el PARCHE, sin coagulo

`cow_edit.occlude` cuesta ~1-2 s por receta (geodesica sobre el vaso) y no se
puede pagar por parche de entrenamiento; por eso se precalcula una vez.

## Solo CPU

Corre en paralelo con el entrenamiento en GPU sin estorbarle, pero comparte CPU
con la carga de datos: usar pocos workers si hay un entrenamiento en marcha.

    python oclusiones.py --workers 3 --por-donante 4
"""
import json
import argparse
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor

import numpy as np

import common as C
import cow_edit as E

ROOT = Path(__file__).parent
OUT = ROOT / "oclus"

# Las oclusiones que importan clinicamente (las mismas que ofrece la UI), con el
# rango de `frac` util: 0.35-0.95 cubre desde proximal hasta el T carotideo.
FRAC = (0.35, 0.95)


def _ves_de_sem(sem):
    """Recupera el mapa de vasos a partir del semantico (ids 1..36)."""
    ves = np.zeros(sem.shape, np.uint8)
    m = (sem >= C.VESSEL0) & (sem <= C.VESSEL0 + C.N_VESSEL - 1)
    ves[m] = sem[m] - C.VESSEL0 + 1
    return ves


def _receta(ves, variante, arteria, frac, rng):
    """Aplica variante + oclusion y devuelve solo lo minimo para reconstruirla.

    Una oclusion es que el vaso DEJA DE VERSE distalmente: la receta guarda que
    segmentos se apagan y nada mas. NO se marca ningun coagulo -- el generador no
    debe pintar el signo hiperdenso, solo la ausencia de contraste distal.
    """
    ves2, _ = E.apply_variant(ves, variante)
    _, _, info = E.occlude(ves2, arteria, frac=frac, rng=rng)
    apagados = [C.VID[n] for n in info["apagados"] if n in C.VID]
    return {"variante": variante, "arteria": arteria, "frac": round(float(frac), 3),
            "apagados": apagados,
            "n_apagados_vox": int(np.isin(ves, apagados).sum())}


def _procesar(args):
    f, k_por_donante, seed = args
    try:
        d = np.load(f)
        ves = _ves_de_sem(d["sem"])
        rng = np.random.default_rng(seed)

        vars_ = [v for v in E.VARIANTS if v in E.available_variants(ves)]
        arts = [a for a in C.OCCLUDABLE if a in E.occludable(ves)]
        if not arts:
            return f.stem, {"error": "ningun vaso ocluible"}

        recetas = []
        intentos = 0
        while len(recetas) < k_por_donante and intentos < k_por_donante * 4:
            intentos += 1
            v = vars_[rng.integers(len(vars_))] if vars_ else "CoW completo"
            a = arts[rng.integers(len(arts))]
            fr = float(rng.uniform(*FRAC))
            try:
                r = _receta(ves, v, a, fr, rng)
            except Exception:                       # noqa: BLE001 (vaso corto, etc.)
                continue
            if not r["apagados"]:                   # oclusion que no apaga nada: inutil
                continue
            recetas.append(r)

        if not recetas:
            return f.stem, {"error": "ninguna receta valida"}

        OUT.mkdir(exist_ok=True)
        np.savez_compressed(OUT / f"{f.stem}.npz", meta=json.dumps(recetas))
        return f.stem, {"n": len(recetas),
                        "apagados": [len(r["apagados"]) for r in recetas]}
    except Exception as e:                          # noqa: BLE001
        return f.stem, {"error": f"{type(e).__name__}: {e}"}


def cargar(stem):
    """Recetas de un donante -> lista de dicts. [] si no hay.

    Tolera los .npz antiguos que ademas guardaban indices de trombo (`thr{i}`):
    solo se lee `meta` y se ignora el resto.
    """
    p = OUT / f"{stem}.npz"
    if not p.exists():
        return []
    return json.loads(str(np.load(p)["meta"]))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--workers", type=int, default=3)
    ap.add_argument("--por-donante", dest="k", type=int, default=4)
    ap.add_argument("--ds", nargs="*", default=["topaneu", "topcow", "isles"])
    a = ap.parse_args()

    tareas = [(f, a.k, i) for i, f in enumerate(
        sorted(p for ds in a.ds for p in (ROOT / "sem" / ds).glob("*.npz")))]
    print(f"{len(tareas)} donantes x {a.k} recetas, {a.workers} workers", flush=True)

    rep, malos = {}, 0
    with ProcessPoolExecutor(a.workers) as ex:
        for k, (name, r) in enumerate(ex.map(_procesar, tareas), 1):
            rep[name] = r
            if "error" in r:
                malos += 1
                print(f"  [{k}/{len(tareas)}] {name}  {r['error']}", flush=True)
            elif k % 20 == 0 or k == len(tareas):
                print(f"  [{k}/{len(tareas)}] {name}  {r}", flush=True)

    (OUT / "report.json").write_text(json.dumps(rep, indent=1))
    ok = [r for r in rep.values() if "error" not in r]
    tot = sum(r["n"] for r in ok)
    print(f"\n{len(ok)} donantes, {tot} recetas, {malos} sin receta -> {OUT}")


if __name__ == "__main__":
    main()
