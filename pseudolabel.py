"""Completa las etiquetas de TopCoW e ISLES a 36 clases y deriva la oclusion.

TopCoW e ISLES solo traen las 15 clases del CoW. El segmentador (entrenado con
el GT de TopAneu) anade el resto del arbol (M2, M3, VA, PICA, SCA, P3P4...).

Fusion: donde el GT humano dice algo, manda el GT; el segmentador solo rellena
las clases que el GT no cubre (16-36). Nunca pisa una etiqueta anotada.

En ISLES (CTA con ictus) el arbol sale truncado del lado ocluido -- porque no hay
contraste que segmentar. Eso hace que el par (etiqueta, imagen) siga siendo
consistente, y ademas permite LEER el sitio de oclusion comparando cada
territorio con su contralateral.

Salida: sem/<ds>/<cid>.npz con el mapa semantico de 41 clases listo para el
generador.  Uso: python pseudolabel.py [--ds topcow,isles,topaneu]
"""
import sys
import json
import argparse
from pathlib import Path

import numpy as np
import torch
from scipy import ndimage as ndi
from monai.networks.nets import UNet
from monai.inferers import sliding_window_inference

sys.path.insert(0, str(Path(__file__).parent))
import common as C
import cow_edit as E

ROOT = Path(__file__).parent
DEV = "cuda"

# cadenas proximal -> distal por territorio, para localizar la oclusion
CHAINS = {
    "R": [["R-ICA-C1-C5", "R-ICA-C6-C7", "R-M1", "R-M2", "R-M3"],
          ["R-ICA-C6-C7", "R-A1A2", "R-A3"],
          ["R-VA", "R-P1P2", "R-P3P4"]],
    "L": [["L-ICA-C1-C5", "L-ICA-C6-C7", "L-M1", "L-M2", "L-M3"],
          ["L-ICA-C6-C7", "L-A1A2", "L-A3"],
          ["L-VA", "L-P1P2", "L-P3P4"]],
}


def load_seg():
    ck = torch.load(ROOT / "ckpt" / "seg_best.pt", map_location=DEV, weights_only=False)
    net = UNet(spatial_dims=3, in_channels=1, out_channels=ck["ncls"],
               channels=(32, 64, 128, 256, 320), strides=(2, 2, 2, 2),
               num_res_units=2).to(DEV)
    net.load_state_dict(ck["model"])
    net.eval()
    print(f"segmentador: Dice val {ck['dice']:.4f} (epoca {ck['epoch']+1})")
    return net, ck["patch"]


@torch.no_grad()
def predict(net, img, patch):
    x = torch.from_numpy(img[None, None].astype(np.float32)).to(DEV)
    with torch.autocast("cuda", dtype=torch.bfloat16):
        lo = sliding_window_inference(x, (patch,) * 3, 2, net, overlap=0.5)
    return lo.argmax(1)[0].cpu().numpy().astype(np.uint8)


def fuse(gt, pred):
    """GT humano (1-15) manda; el segmentador solo aporta las clases 16-36."""
    out = gt.copy()
    add = (gt == 0) & (pred > 15)
    out[add] = pred[add]
    return out


def detect_occlusion(ves):
    """Localiza la arteria ocluida por asimetria con el lado contralateral.

    Devuelve (arteria|None, mascara_trombo, info). La arteria ocluida es el
    ultimo segmento PERMEABLE de una cadena cuyo territorio distal esta apagado.
    """
    vol = {n: int((ves == v).sum()) for n, v in E.V.items()}
    info = {"ratios": {}}
    best = None

    for side, chains in CHAINS.items():
        other = "L" if side == "R" else "R"
        for chain in chains:
            for i, seg in enumerate(chain):
                if i == 0:
                    continue
                mirror = other + seg[1:]
                a, b = vol.get(seg, 0), vol.get(mirror, 0)
                if b < 100:
                    continue                    # sin referencia contralateral fiable
                r = a / b
                info["ratios"][seg] = round(r, 2)
                if r < 0.40:                    # territorio claramente apagado
                    prox = chain[i - 1]
                    if vol.get(prox, 0) < 100:
                        continue                # el proximal tampoco esta: sigue subiendo
                    score = 1.0 - r
                    if best is None or score > best[0]:
                        best = (score, prox, seg)
                    break                       # el primer eslabon roto de la cadena

    if best is None:
        return None, np.zeros(ves.shape, bool), {**info, "detected": False}

    _, prox, lost = best
    # el trombo esta al final del ultimo segmento con contraste
    m = ves == E.V[prox]
    gd = E.geodesic(m, E._seed_of(ves, prox))
    valid = gd >= 0
    dmax = int(gd[valid].max())
    thr = valid & (gd > dmax - max(2, int(4.0 / C.SPACING)))
    return prox, thr, {**info, "detected": True, "arteria": prox,
                       "territorio_apagado": lost, "len_vox": dmax}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ds", default="topcow,isles,topaneu")
    a = ap.parse_args()
    dss = a.ds.split(",")

    net = patch = None
    if {"topcow", "isles"} & set(dss):
        net, patch = load_seg()

    report = {}
    for ds in dss:
        (ROOT / "sem" / ds).mkdir(parents=True, exist_ok=True)
        files = sorted((ROOT / "prep" / ds).glob("*.npz"))
        for k, f in enumerate(files, 1):
            d = np.load(f)
            img = d["img"].astype(np.float32)
            hu = C.to_hu(img)
            ane = d["ane"]
            thr = None
            occ = {}

            if ds == "topaneu":
                ves = d["ves"]                  # GT completo de 36 clases
            else:
                ves = fuse(d["ves"], predict(net, img, patch))
                if ds == "isles":
                    art, thr, occ = detect_occlusion(ves)
                    occ["arteria"] = art

            sem = C.semantic_map(hu, ves, ane, thr)
            np.savez_compressed(ROOT / "sem" / ds / f.name, sem=sem, ves=ves,
                                affine=d["affine"], src=ds)
            report[f.stem] = {"ds": ds, "vessel_vox": int((ves > 0).sum()),
                              "n_clases": int(len(np.unique(ves)) - 1), **occ}
            msg = f"[{ds} {k}/{len(files)}] {f.stem}: {int((ves>0).sum())} vox, " \
                  f"{int(len(np.unique(ves))-1)} clases"
            if ds == "isles":
                msg += f" | oclusion: {occ.get('arteria') or 'NO DETECTADA'}"
            print(msg, flush=True)

    (ROOT / "sem" / "report.json").write_text(json.dumps(report, indent=1))
    isl = [v for v in report.values() if v["ds"] == "isles"]
    if isl:
        det = sum(1 for v in isl if v.get("arteria"))
        print(f"\nISLES: oclusion localizada en {det}/{len(isl)} casos")
        from collections import Counter
        print(" ", Counter(v.get("arteria") for v in isl).most_common())


if __name__ == "__main__":
    main()
