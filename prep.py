"""Remuestrea TopAneu / TopCoW / ISLES al slab canonico de 0.6 mm.

Salida: prep/<dataset>/<case>.npz con
    img  float16 [-1,1]  (HU ventaneada)
    ves  uint8   0..36   (etiquetas de vaso en el espacio TopAneu)
    ane  uint8   0/1     (aneurisma, solo TopAneu)
    src  str             dataset de origen
Uso:  python prep.py [topaneu|topcow|isles|all] [--workers N]
"""
import sys
import json
import argparse
import traceback
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor, as_completed

import numpy as np

sys.path.insert(0, str(Path(__file__).parent))
import common as C

DL = Path(r"C:\Users\LBorrego\Downloads")
OUT = Path(__file__).parent / "prep"

TOPANEU = DL / "topaneu_deployment" / "topaneu_deployment"
TOPCOW = DL / "Dataset008_Willis" / "Dataset008_Willis"
ISLES = DL / "TopCoW_reference" / "CTA_ISLES2024_TUM" / "CTA_ISLES2024_TUM"


def cases(dataset):
    """[(case_id, img_path, vessel_path, aneurysm_path|None), ...]"""
    if dataset == "topaneu":   # solo CT: el generador es de CTA
        out = []
        for f in sorted((TOPANEU / "images").glob("topaneu_center*_ct_*_0000.nii.gz")):
            cid = f.name[:-12]
            out.append((cid, f, TOPANEU / "vessel_masks" / f"{cid}.nii.gz",
                        TOPANEU / "location_masks" / f"{cid}.nii.gz"))
        return out
    if dataset == "topcow":
        return [(f.name[:-12], f, TOPCOW / "labelsTr" / f"{f.name[:-12]}.nii.gz", None)
                for f in sorted((TOPCOW / "imagesTr").glob("topcow_ct_*_0000.nii.gz"))]
    if dataset == "isles":
        out = []
        for f in sorted((ISLES / "imagesTr").glob("sub-stroke_*_ct.nii.gz")):
            cid = f.name[:-7]
            out.append((cid, f, ISLES / "cow_seg_labelsTr" / f"{cid}_full_LPS_Mask.nii.gz", None))
        return out
    raise ValueError(dataset)


def prep_one(args):
    dataset, cid, img_p, ves_p, ane_p = args
    dst = OUT / dataset / f"{cid}.npz"
    if dst.exists():
        return cid, "skip", {}
    try:
        hu, aff = C.load_nifti(img_p)
        hu = hu.astype(np.float32)
        ves, vaff = C.load_nifti(ves_p)
        ves = ves.astype(np.uint8)

        if dataset != "topaneu":   # 15 clases TopCoW -> ids TopAneu (identidad)
            ves = np.where(ves <= 15, ves, 0).astype(np.uint8)

        # el grid se ancla en el CoW, que existe en los tres datasets
        center = C.cow_center(ves, vaff)
        taff = C.target_affine(center)

        img = C.to_unit(C.resample(hu, aff, taff, order=1, cval=-1024.0))
        vg = C.resample(ves, vaff, taff, order=0, cval=0).astype(np.uint8)

        ane = np.zeros(C.SHAPE, np.uint8)
        if ane_p is not None and Path(ane_p).exists():
            a, aaff = C.load_nifti(ane_p)
            ane = (C.resample(a.astype(np.uint8), aaff, taff, 0, 0) > 0).astype(np.uint8)

        dst.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(dst, img=img.astype(np.float16), ves=vg, ane=ane,
                            affine=taff.astype(np.float32), src=dataset)
        n = int((vg > 0).sum())
        return cid, "ok", {"vessel_vox": n, "n_labels": int(len(np.unique(vg)) - 1),
                           "aneurysm_vox": int(ane.sum())}
    except Exception:
        return cid, "FAIL: " + traceback.format_exc(limit=2).replace("\n", " | "), {}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("dataset", choices=["topaneu", "topcow", "isles", "all"])
    ap.add_argument("--workers", type=int, default=6)
    a = ap.parse_args()

    dss = ["topaneu", "topcow", "isles"] if a.dataset == "all" else [a.dataset]
    jobs = [(d, *c) for d in dss for c in cases(d)]
    print(f"{len(jobs)} casos, {a.workers} workers", flush=True)

    stats, nok = [], 0
    with ProcessPoolExecutor(max_workers=a.workers) as ex:
        futs = [ex.submit(prep_one, j) for j in jobs]
        for i, f in enumerate(as_completed(futs), 1):
            cid, status, st = f.result()
            if status == "ok":
                nok += 1
                stats.append(st)
            print(f"[{i}/{len(jobs)}] {cid}: {status} {st}", flush=True)

    if stats:
        v = np.array([s["vessel_vox"] for s in stats])
        print(f"\nOK {nok}/{len(jobs)}. vasos/caso: mediana {np.median(v):.0f} "
              f"min {v.min()} max {v.max()}", flush=True)
        bad = [s for s in stats if s["vessel_vox"] < 3000]
        if bad:
            print(f"AVISO: {len(bad)} casos con <3000 vox de vaso (posible mal centrado)")


if __name__ == "__main__":
    main()
