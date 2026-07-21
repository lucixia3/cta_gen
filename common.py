"""Espacio canonico compartido por el segmentador y el generador.

Grid: 0.6 mm isotropico, 256 x 256 x 192 (RAS), centrado en el poligono de Willis.
Cubre VA/BA -> CoW -> M1/M2/M3, A1-A3, P1-P4: todo el territorio de las LVO.
"""
import json
import numpy as np
import nibabel as nib
from scipy import ndimage as ndi

SPACING = 0.6
SHAPE = (256, 256, 192)              # x (L-R), y (P-A), z (I-S)
# Offset respecto al centroide del CoW, medido sobre la extension real de los
# vasos en TopAneu (x 124 mm, y 111 mm, z 103 mm de mediana): el arbol queda
# centrado en el slab con y desplazado en posterior (VA/PICA/P3P4).
CENTER_OFFSET = np.array([0.0, -11.0, 0.0])

HU_WINDOW = (-100.0, 900.0)          # aire ... hueso; el contraste vive en 150-450

# --- espacio de etiquetas semanticas -----------------------------------------
# 0 aire | 1 tejido blando/parenquima | 2 hueso | 3..38 vasos (TopAneu 1..36)
# 39 aneurisma | 40 trombo
AIR, SOFT, BONE, VESSEL0 = 0, 1, 2, 3
ANEURYSM, THROMBUS = 39, 40
N_CLASSES = 41

N_VESSEL = 36
VESSEL_NAMES = ["BA", "R-P1P2", "L-P1P2", "R-ICA-C6-C7", "R-M1", "L-ICA-C6-C7",
                "L-M1", "R-Pcom", "L-Pcom", "Acom", "R-A1A2", "L-A1A2", "R-A3",
                "L-A3", "3rd-A2", "3rd-A3", "R-M2", "R-M3", "L-M2", "L-M3",
                "R-P3P4", "L-P3P4", "R-VA", "L-VA", "R-SCA", "L-SCA", "R-AICA",
                "L-AICA", "R-PICA", "L-PICA", "R-AChA", "L-AChA", "R-OA",
                "L-OA", "R-ICA-C1-C5", "L-ICA-C1-C5"]
VID = {n: i + 1 for i, n in enumerate(VESSEL_NAMES)}   # nombre -> id TopAneu 1..36

# TopCoW/ISLES usan 15 clases que coinciden EXACTAMENTE con las 15 primeras de
# TopAneu (BA, R/L-PCA=P1P2, R/L-ICA=ICA-C6-C7, R/L-MCA=M1, R/L-Pcom, Acom,
# R/L-ACA=A1A2, R/L-A3, 3rd-A2), asi que la conversion es la identidad.
TOPCOW_TO_TOPANEU = {i: i for i in range(1, 16)}

# El CoW propiamente dicho: presente en los tres datasets -> ancla de centrado.
COW_CORE = [VID["BA"], VID["R-P1P2"], VID["L-P1P2"], VID["R-ICA-C6-C7"],
            VID["L-ICA-C6-C7"], VID["R-M1"], VID["L-M1"], VID["R-A1A2"],
            VID["L-A1A2"]]

# Arterias ocluibles en una LVO (las que ofrece la UI).
OCCLUDABLE = ["BA", "R-ICA-C6-C7", "L-ICA-C6-C7", "R-M1", "L-M1", "R-M2",
              "L-M2", "R-A1A2", "L-A1A2", "R-P1P2", "L-P1P2",
              "R-ICA-C1-C5", "L-ICA-C1-C5", "R-VA", "L-VA"]


def _lr_swap_table():
    """Permutacion de ids de vaso al reflejar en L-R (R-M1 <-> L-M1, ...).

    Sin par: BA, Acom, 3rd-A2, 3rd-A3 (estructuras de linea media).
    """
    t = np.arange(N_VESSEL + 1)
    for name, vid in VID.items():
        if name.startswith("R-"):
            t[vid] = VID["L-" + name[2:]]
        elif name.startswith("L-"):
            t[vid] = VID["R-" + name[2:]]
    return t


LR_SWAP = _lr_swap_table()


def flip_lr(img, ves):
    """Refleja el volumen en L-R e intercambia las etiquetas lateralizadas."""
    return img[::-1].copy(), LR_SWAP[ves[::-1]].astype(np.uint8)


def target_affine(center_world):
    """Affine RAS del grid canonico centrado en `center_world` (3,)."""
    aff = np.eye(4)
    aff[0, 0] = aff[1, 1] = aff[2, 2] = SPACING
    aff[:3, 3] = center_world - SPACING * (np.array(SHAPE) - 1) / 2.0
    return aff


def cow_center(lab, affine):
    """Centro del grid: centroide del CoW en coordenadas mundo + offset."""
    m = np.isin(lab, COW_CORE)
    if m.sum() < 50:
        m = lab > 0
    idx = np.array(np.nonzero(m)).mean(axis=1)
    world = affine[:3, :3] @ idx + affine[:3, 3]
    return world + CENTER_OFFSET


def resample(vol, src_affine, dst_affine, order, cval):
    """Remuestrea `vol` al grid destino. order=1 imagen, order=0 etiquetas."""
    M = np.linalg.inv(src_affine) @ dst_affine
    src_sp = np.linalg.norm(src_affine[:3, :3], axis=0)
    if order > 0:
        # antialiasing al bajar de resolucion (0.43 -> 0.6 mm)
        sig = np.maximum(0.0, 0.35 * (SPACING / src_sp - 1.0))
        if sig.max() > 1e-3:
            vol = ndi.gaussian_filter(vol.astype(np.float32), sig)
    return ndi.affine_transform(vol, M[:3, :3], offset=M[:3, 3],
                                output_shape=SHAPE, order=order, cval=cval,
                                mode="constant", prefilter=False)


def to_unit(hu):
    """HU -> [-1, 1]."""
    lo, hi = HU_WINDOW
    return (np.clip(hu, lo, hi) - lo) / (hi - lo) * 2.0 - 1.0


def to_hu(x):
    lo, hi = HU_WINDOW
    return (x + 1.0) / 2.0 * (hi - lo) + lo


# Estadisticas globales del dataset en el espacio [-1,1] (209 casos).
# La difusion asume x0 ~ N(0,1). En [-1,1] los datos tienen std 0.50, y con la
# senal asi de comprimida el modelo aprende la solucion trivial eps ~= x_t a t
# alto -> x0_pred ~= 0 y el muestreo devuelve un volumen PLANO. Estandarizar (y
# usar v-prediction) es lo que lo arregla.
NORM_MU, NORM_SD = -0.6592, 0.4960


def normalize(x):
    """[-1,1] -> N(0,1) aprox. (el espacio en el que vive la difusion)."""
    return (x - NORM_MU) / NORM_SD


def denormalize(z):
    return z * NORM_SD + NORM_MU


def semantic_map(hu, vessels, aneurysm=None, thrombus=None):
    """Construye el mapa semantico de 41 clases a partir de HU + vasos."""
    lab = np.full(hu.shape, SOFT, np.uint8)
    lab[hu < -300] = AIR
    lab[hu > 500] = BONE
    v = vessels > 0
    lab[v] = VESSEL0 + vessels[v].astype(np.uint8) - 1
    if aneurysm is not None:
        lab[aneurysm > 0] = ANEURYSM
    if thrombus is not None:
        lab[thrombus > 0] = THROMBUS
    return lab


def load_nifti(path):
    im = nib.load(str(path))
    return np.asarray(im.dataobj), im.affine


def save_nifti(path, arr, affine):
    nib.save(nib.Nifti1Image(np.asarray(arr), affine), str(path))
