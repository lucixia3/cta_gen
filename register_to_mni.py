"""Registro de un NCCT arbitrario al espacio MNI del atlas (182x218x182).

Es la pieza que faltaba para ANADIR donantes/fondos al banco: un NCCT de control (p.ej.
SinoCT, leido como normal) llega en su espacio nativo; para que el infarto sintetico caiga
en el territorio correcto tiene que estar en el MISMO grid que el atlas de territorios. Se
registra (afin, Mattes MI, multi-resolucion) a un template CT-MNI y se remuestrea al grid
del atlas.

El template `atlas/mni_ncct_template.nii.gz` es el promedio de los NCCT del banco (todos ya
alineados al atlas), asi que registrar contra el deja el nuevo NCCT en espacio del atlas.

Uso:
    python register_to_mni.py <archivo|carpeta> --out <carpeta>     # registra -> {stem}_ncct.nii.gz
    python register_to_mni.py --validate                            # round-trip sobre el banco
"""
import sys
import argparse
from pathlib import Path

import numpy as np
import SimpleITK as sitk

L2N = Path(r"C:\Desactivar_Respaldo\lesion2ncct_portable\lesion2ncct_bundle")
TEMPLATE = L2N / "atlas" / "mni_ncct_template.nii.gz"
BANK = L2N / "anatomy_bank"
AIR = -1000.0


def _register(fixed, moving):
    """Afin Mattes-MI multi-resolucion. Devuelve la transformada moving->fixed."""
    ini = sitk.CenteredTransformInitializer(
        fixed, moving, sitk.AffineTransform(3),
        sitk.CenteredTransformInitializerFilter.MOMENTS)
    R = sitk.ImageRegistrationMethod()
    R.SetMetricAsMattesMutualInformation(numberOfHistogramBins=50)
    R.SetMetricSamplingStrategy(R.RANDOM)
    R.SetMetricSamplingPercentage(0.2, seed=42)
    R.SetInterpolator(sitk.sitkLinear)
    R.SetOptimizerAsGradientDescent(learningRate=1.0, numberOfIterations=300,
                                    convergenceMinimumValue=1e-6, convergenceWindowSize=10)
    R.SetOptimizerScalesFromPhysicalShift()
    R.SetShrinkFactorsPerLevel([4, 2, 1])
    R.SetSmoothingSigmasPerLevel([2, 1, 0])
    R.SetInitialTransform(ini, inPlace=False)
    return R.Execute(fixed, moving)


def register_to_mni(moving_path, template_path=TEMPLATE):
    """NCCT (cualquier espacio) -> mismo NCCT remuestreado al grid MNI del atlas (sitk image)."""
    fixed = sitk.Cast(sitk.ReadImage(str(template_path)), sitk.sitkFloat32)
    moving = sitk.Cast(sitk.ReadImage(str(moving_path)), sitk.sitkFloat32)
    T = _register(fixed, moving)
    return sitk.Resample(moving, fixed, T, sitk.sitkLinear, AIR, sitk.sitkFloat32)


def _brain(a):
    from scipy import ndimage as ndi
    b = ndi.binary_fill_holes((a > 5) & (a < 78))
    lab, n = ndi.label(b)
    return b if n == 0 else lab == (1 + int(np.argmax(np.bincount(lab.ravel())[1:])))


def validate():
    """Round-trip: perturba cada NCCT del banco con una afin conocida y lo registra de vuelta;
    mide el Dice de la mascara de cerebro contra el original (alto = el registro converge)."""
    import glob
    from scipy import ndimage as ndi
    fixed = sitk.Cast(sitk.ReadImage(str(TEMPLATE)), sitk.sitkFloat32)
    rng = np.random.default_rng(0)
    dices = []
    for f in sorted(glob.glob(str(BANK / "*_ncct.nii.gz"))):
        orig = sitk.Cast(sitk.ReadImage(f), sitk.sitkFloat32)
        # perturbacion rigida conocida: rota ~10 grados y desplaza ~8 mm
        pert = sitk.Euler3DTransform()
        pert.SetCenter(orig.TransformContinuousIndexToPhysicalPoint(
            [(s - 1) / 2 for s in orig.GetSize()]))
        pert.SetRotation(*(rng.uniform(-0.17, 0.17, 3)))
        pert.SetTranslation(tuple(rng.uniform(-8, 8, 3)))
        moved = sitk.Resample(orig, orig, pert, sitk.sitkLinear, AIR, sitk.sitkFloat32)
        # registrar de vuelta al template
        T = _register(fixed, moved)
        back = sitk.Resample(moved, fixed, T, sitk.sitkLinear, AIR, sitk.sitkFloat32)
        ob = _brain(sitk.GetArrayFromImage(sitk.Resample(orig, fixed, sitk.Transform(),
                    sitk.sitkLinear, AIR, sitk.sitkFloat32)))
        bb = _brain(sitk.GetArrayFromImage(back))
        d = 2 * (ob & bb).sum() / (ob.sum() + bb.sum() + 1e-6)
        dices.append(d)
        print(f"  {Path(f).stem:28s} recuperacion Dice {d:.3f}", flush=True)
    print(f"\nmedia {np.mean(dices):.3f} | min {np.min(dices):.3f}  "
          f"(>0.9 = el registro recupera la alineacion MNI)")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("target", nargs="?", help="NCCT .nii.gz o carpeta")
    ap.add_argument("--out", help="carpeta de salida (NCCT en MNI 182^3)")
    ap.add_argument("--validate", action="store_true", help="round-trip sobre el banco")
    a = ap.parse_args()

    if a.validate:
        validate(); return
    if not a.target or not a.out:
        ap.error("da <target> y --out, o usa --validate")

    p = Path(a.target)
    files = [p] if p.is_file() else (sorted(p.glob("*.nii.gz")))
    out = Path(a.out); out.mkdir(parents=True, exist_ok=True)
    for f in files:
        reg = register_to_mni(f)
        dst = out / f"{f.stem.replace('.nii','')}_ncct.nii.gz"
        sitk.WriteImage(reg, str(dst))
        print(f"  {f.name} -> {dst.name}", flush=True)


if __name__ == "__main__":
    main()
