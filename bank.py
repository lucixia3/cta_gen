"""Prepara el banco de anatomias de lesion2ncct para el grafo de cta_gen.

El banco de `lesion2ncct_portable` es la UNICA fuente que trae, del mismo cerebro,
un CTA (1 mm, nativo) y un NCCT (MNI 182x218x182) alineado voxel a voxel con el
atlas de territorios. Por eso es la anatomia que usa `caso.py`: sin el no hay
forma de que el CTA con la oclusion y el NCCT con el infarto sean el mismo paciente.

Lo que hay que arreglar antes de poder usarlo:

El banco solo trae las 15 clases de TopCoW (en la practica, 1-12 y a veces la 15).
No estan ni las ICA cervicales ni las vertebrales -- que son precisamente las
FUENTES del grafo de flujo de `cow_edit` -- ni el arbol distal (M2, M3, A3, P3P4).
Sobre esas etiquetas crudas, `reachable()` arranca con el conjunto de fuentes vacio
y declara apagado todo el arbol, y ademas no habria M2/M3 que apagar al ocluir un
M1, que es justo lo que distingue a este generador.

La solucion ya estaba escrita para TopCoW en `pseudolabel.py`: remuestrear al slab
canonico de 0.6 mm y completar a 36 clases con el segmentador. El GT humano (1-15)
manda; el segmentador solo aporta las clases que el GT no cubre (16-36).

Salida: bank_sem/<pid>.npz con img [-1,1], ves 0..36 y el affine canonico.
Se cachea: solo hay 9 anatomias y esto necesita GPU.

Uso:  python bank.py [--force]
"""
import sys
import csv
import json
import argparse
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).parent))
import common as C
import pseudolabel as P

ROOT = Path(__file__).parent
L2N = Path(r"C:\Desactivar_Respaldo\lesion2ncct_portable\lesion2ncct_bundle")
BANK = L2N / "anatomy_bank"
OUT = ROOT / "bank_sem"


def pids():
    """[(pid, cow_id), ...] del manifest del banco."""
    with open(BANK / "manifest.csv", encoding="utf-8") as f:
        return [(r["pid"], int(r["cow_id"])) for r in csv.DictReader(f)]


# Las 6 clases lateralizadas de TopCoW (las unicas R/L entre las 15 del GT humano):
# ICA-C6-C7, M1, A1A2, P1P2 (occluibles) + Pcom, A3 (colaterales/distales). Cubrirlas
# TODAS permite corregir el GT clase a clase sin dejar ninguna etiqueta lateralizada sin
# verificar -- las clases 16-36 las aporta luego el segmentador, con la R/L ya buena.
CHEQUEO_LR = ("ICA-C6-C7", "M1", "A1A2", "P1P2", "Pcom", "A3")
MARGEN = 2.0                        # un lado necesita ganar por 2x para no abstenerse
MIN_LADO = 100                     # voxeles minimos para que UN lado emita voto


def votos_lr(gt, pred):
    """Voto del segmentador por clase Y por lado: {clase: {'R': (a, b), 'L': (a, b)}}.

    Para los voxeles que el GT llama "R-<n>", `a` = cuantos el segmentador tambien ve R,
    `b` = cuantos ve L (y analogo para "L-<n>"). Se separa por lado a proposito: el voto
    AGREGADO no distingue un espejo bilateral limpio (ambos lados cambiados, que SE puede
    corregir con un swap) de una contaminacion de un solo lado (que no). El juez es el
    propio segmentador: se entreno con el GT de TopAneu, cuya convencion R/L es la buena.
    """
    votos = {}
    for n in CHEQUEO_LR:
        rid, lid = C.VID.get(f"R-{n}"), C.VID.get(f"L-{n}")
        if rid is None or lid is None:
            continue
        d = {}
        for lado, sid, oid in (("R", rid, lid), ("L", lid, rid)):
            m = gt == sid
            d[lado] = (int((pred[m] == sid).sum()), int((pred[m] == oid).sum())) \
                if m.sum() >= 50 else (0, 0)
        votos[n] = d
    return votos


def _lado(a, b):
    """El voto de UN lado -> 'agree' | 'mirror' | 'abstain'."""
    if a + b < MIN_LADO:
        return "abstain"
    if b >= MARGEN * max(a, 1):
        return "mirror"
    if a >= MARGEN * max(b, 1):
        return "agree"
    return "abstain"


def decision_lr(votos):
    """Por clase -> 'keep' | 'swap' | 'incoherente' | 'abstain', a partir de los DOS lados.

    'keep'        ambos lados con voto coinciden en 'agree': bien lateralizada.
    'swap'        ambos coinciden en 'mirror': espejo bilateral LIMPIO -> se puede voltear.
    'incoherente' un lado 'agree' y el otro 'mirror': un swap arreglaria uno y rompería el
                  otro. No hay forma segura -> el donante se descarta.
    'abstain'     sin evidencia en ninguno de los dos lados: no se toca.
    """
    dec = {}
    for n, d in votos.items():
        lados = {_lado(*d.get("R", (0, 0))), _lado(*d.get("L", (0, 0)))} - {"abstain"}
        if not lados:
            dec[n] = "abstain"
        elif lados == {"agree"}:
            dec[n] = "keep"
        elif lados == {"mirror"}:
            dec[n] = "swap"
        else:
            dec[n] = "incoherente"
    return dec


def swap_clases(gt, clases):
    """Intercambia R-<n> <-> L-<n> SOLO para `clases` (relabel; no mueve voxeles)."""
    out = gt.copy()
    for n in clases:
        rid, lid = C.VID.get(f"R-{n}"), C.VID.get(f"L-{n}")
        if rid is None or lid is None:
            continue
        out[gt == rid] = lid
        out[gt == lid] = rid
    return out


def lateralidad(gt, pred):
    """Corrige la R/L del GT clase a clase, con verificacion por lado.

    -> (estado, gt_corregido, votos, swaps)  con estado en:
        "ok"          la R/L ya era buena en todas las clases con voto.
        "espejo"      TODAS las clases con voto estaban espejadas: espejo global limpio.
        "corregido"   SOLO algunas estaban espejadas (defecto por clase): se enderezan
                      esas y se deja el resto. Es el caso que antes se descartaba entero.
        "incoherente" alguna clase tiene un lado bien y el otro al reves: no es un espejo
                      limpio, no hay forma de garantizar el lado. Se descarta.

    Un tercio del banco trae el poligono mal lateralizado. En `pat_cow1_02` es un espejo
    global (todas las clases). Pero en `pat_cow0_02` solo la ICA y el M1 estan espejados y
    la P1P2 no: un LR_SWAP global arreglaria unas y rompería otras, por eso antes se
    descartaba ENTERO. La clave es que ese defecto es coherente POR CLASE -- cada clase
    esta espejada en sus DOS lados o en ninguno -- asi que se puede enderezar clase a
    clase. Solo es irreparable si una clase mezcla los lados (uno bien, otro espejado),
    que es lo que detecta `decision_lr` mirando cada lado por separado.

    Las clases que se abstienen (pocos voxeles o voto ambiguo, tipico de la ICA y la Acom
    pegadas a la linea media, Dice 0.71) se dejan como estan: es el mismo riesgo residual
    que ya aceptaban los donantes "ok"/"espejo", no uno nuevo.
    """
    votos = votos_lr(gt, pred)
    dec = decision_lr(votos)
    if any(d == "incoherente" for d in dec.values()):
        return "incoherente", gt, votos, [n for n, d in dec.items() if d == "incoherente"]

    swaps = [n for n, d in dec.items() if d == "swap"]
    if not swaps:
        return "ok", gt, votos, []
    n_con_voto = sum(1 for d in dec.values() if d != "abstain")
    estado = "espejo" if len(swaps) == n_con_voto else "corregido"
    return estado, swap_clases(gt, swaps), votos, swaps


def prep_one(pid, net, patch):
    """CTA+seg del banco -> slab canonico con las 36 clases completadas."""
    import nibabel as nib
    # canonico RAS: `cow_edit` asume x = derecha (el reflejo L-R y los nombres
    # lateralizados dependen de ello)
    cim = nib.as_closest_canonical(nib.load(BANK / f"{pid}_cta.nii.gz"))
    sim = nib.as_closest_canonical(nib.load(BANK / f"{pid}_seg.nii.gz"))
    hu = np.asanyarray(cim.dataobj).astype(np.float32)
    ves = np.asanyarray(sim.dataobj).astype(np.uint8)
    ves = np.where(ves <= 15, ves, 0).astype(np.uint8)   # 15 clases TopCoW == 15 primeras de TopAneu

    taff = C.target_affine(C.cow_center(ves, sim.affine))
    img = C.to_unit(C.resample(hu, cim.affine, taff, order=1, cval=-1024.0))
    gt = C.resample(ves, sim.affine, taff, order=0, cval=0).astype(np.uint8)

    pred = P.predict(net, img.astype(np.float32), patch)
    # Endereza la R/L clase a clase (espejo global, defecto por clase, o descarte) y
    # verifica el resultado con el mismo segmentador. `gt` ya sale corregido.
    estado, gt, votos, swaps = lateralidad(gt, pred)

    full = P.fuse(gt, pred)
    return img.astype(np.float16), full, taff.astype(np.float32), estado, votos, swaps


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--force", action="store_true", help="rehace los ya cacheados")
    a = ap.parse_args()

    OUT.mkdir(exist_ok=True)
    todo = [(p, c) for p, c in pids() if a.force or not (OUT / f"{p}.npz").exists()]
    if not todo:
        print(f"banco ya preparado ({len(pids())} anatomias en {OUT})")
        return

    if not torch.cuda.is_available():
        print("AVISO: sin CUDA. El segmentador ira por CPU y tardara mucho.", flush=True)
    net, patch = P.load_seg()

    import cow_edit as E
    # Se fusiona con el reporte existente: si solo se reprocesan algunos donantes (sin
    # --force), las entradas de los ya cacheados no deben perderse.
    rep_path = OUT / "report.json"
    report = json.loads(rep_path.read_text(encoding="utf-8")) if rep_path.exists() else {}
    for k, (pid, cow) in enumerate(todo, 1):
        img, ves, aff, estado, votos, swaps = prep_one(pid, net, patch)
        v = " ".join(f"{n}[R {d['R'][0]}/{d['R'][1]} L {d['L'][0]}/{d['L'][1]}]"
                     for n, d in votos.items())

        if estado == "incoherente":
            # Ni la correccion por clase lo salva: tras enderezar lo que el segmentador veia
            # al reves, aun ve alguna clase espejada. Se descarta, que es preferible a un
            # caso con la oclusion en un hemisferio y el infarto en el otro -- y ademas
            # ninguna imagen lo delataria.
            (OUT / f"{pid}.npz").unlink(missing_ok=True)
            report[pid] = {"cow_id": cow, "estado": estado, "votos": votos,
                           "swaps": swaps, "usado": False}
            print(f"[{k}/{len(todo)}] {pid} (CoW {cow}): DESCARTADO -- R/L incoherente pese "
                  f"a corregir {swaps} (voto tras correccion: {v})", flush=True)
            continue

        np.savez_compressed(OUT / f"{pid}.npz", img=img, ves=ves, affine=aff, cow_id=cow)
        fuentes = [s for s in E.SOURCES if E.present(ves, s)]
        distal = [n for n in ("R-M2", "L-M2", "R-M3", "L-M3") if E.present(ves, n)]
        report[pid] = {"cow_id": cow, "estado": estado, "votos": votos, "swaps": swaps,
                       "usado": True,
                       "n_clases": len(sorted(set(np.unique(ves)) - {0})),
                       "fuentes": fuentes, "distal": distal}
        nota = {"espejo": "  -> espejo global, enderezado",
                "corregido": f"  -> ENDEREZADO por clase: {swaps}"}.get(estado, "")
        print(f"[{k}/{len(todo)}] {pid} (CoW {cow}): fuentes {len(fuentes)}/4 | "
              f"distal {len(distal)} | R/L {estado}" + nota, flush=True)

    (OUT / "report.json").write_text(json.dumps(report, indent=1, ensure_ascii=False))
    esp = [p for p, r in report.items() if r["estado"] == "espejo"]
    cor = [p for p, r in report.items() if r["estado"] == "corregido"]
    mal = [p for p, r in report.items() if r["estado"] == "incoherente"]
    ok = [p for p, r in report.items() if r.get("usado")]
    print(f"\n{len(ok)}/{len(report)} donantes utilizables.")
    if esp:
        print(f"  {len(esp)} con espejo global, enderezados: {esp}")
    if cor:
        print(f"  {len(cor)} con defecto de lateralidad POR CLASE, enderezados y "
              f"re-verificados (antes se descartaban): {cor}")
    if mal:
        print(f"  {len(mal)} DESCARTADOS: ni la correccion por clase resuelve la R/L: {mal}")
    fam = {}
    for p in ok:
        fam[report[p]["cow_id"]] = fam.get(report[p]["cow_id"], 0) + 1
    print(f"  anatomias por familia de CoW: {dict(sorted(fam.items()))}")


if __name__ == "__main__":
    main()
