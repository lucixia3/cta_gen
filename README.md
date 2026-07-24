# gen_cta — synthetic stroke CTA (Circle of Willis variant × occlusion site)

A single **3D SPADE diffusion model** that paints a contrast CT angiography (CTA)
from a semantic map. The **Circle of Willis (CoW) variant** and the **occlusion
site** are *not* separate model inputs — they are edits of the semantic map that
enters through SPADE. One model therefore covers any variant × occlusion
combination, including ones absent from the training data.

```
donor --> [CoW variant] --> [occlusion by graph reachability] --> edited vessel tree --> SPADE diffusion --> CTA
```

Pretrained weights: **[`lborrego/gen_cta`](https://huggingface.co/lborrego/gen_cta)**.

---

## Quick start

```bash
pip install -r requirements.txt
```

Download the weights into `ckpt/` (the filename `generate.py` loads by default):

```python
from huggingface_hub import hf_hub_download
hf_hub_download("lborrego/gen_cta", "gen_best.pt", local_dir="ckpt")  # -> ckpt/gen_best.pt, 166 MB
```

Generate one case — from Python:

```python
from generate import generate

generate(
    variant="CoW completo",   # run `python generate.py --list` for the options
    artery="L-M1",            # occlusion site: the vessel disappears distal to it
    frac=0.6,                 # occlusion position along the segment (0 proximal .. 1 distal)
    guidance=1.3,             # vessel intensity (1.0 -> ~66% of real, 1.3 -> matches real)
    denoise=0.5,              # light parenchyma de-graining (protects vessels/bone)
    out="cta.nii.gz",
)
```

…or from the CLI:

```bash
python generate.py --variant "CoW completo" --artery L-M1 --frac 0.6 --out cta.nii.gz
python generate.py --list      # available CoW variants and occludable arteries
```

A CUDA GPU is required. Each call writes the CTA (`cta.nii.gz`) and its semantic
map (`cta_sem.nii.gz`).

---

## How it works

### The semantic map is the interface

The model is conditioned on a **multi-hot map of 45 channels**: 41 semantic classes
(air, soft tissue, bone, 36 labelled arteries, aneurysm) plus 4 tissue channels
(brain / extracranial / CSF). Editing that map — not the network — is how variant
and occlusion are applied, so the same weights generalise to combinations never
seen in training.

### Variant and occlusion are graph edits (`cow_edit.py`)

The vessel tree is a graph. A **variant** rewires it (e.g. absent Acom, fetal PCA).
An **occlusion** turns off every segment distal to the occluded artery by *graph
reachability* — the vessel simply stops being visible past the occluded point. This
is reachability, **not** a perfusion simulation: there are no flow rates or
resistances, and no pial collaterals. An occlusion is rendered as the **absence of
distal contrast**, not as a hyperdense clot.

### Vessel intensity is set at sampling, not training (`guidance`)

Diffusion regresses thin structures toward the mean, so vessels (1–2 voxels wide)
render at ~66 % of their real intensity. Raising the training loss weight on vessels
does not fix it. What does is **amplifying the vessel condition during sampling**
(classifier-free-guidance style, using the map with vessels removed as the
unconditional):

| `guidance` | vessel contrast vs. real |
|-----------:|:-------------------------|
| 1.0 | ~66 % (pale) |
| **1.3** | **~100 % (matches real)** |
| ≥2.0 | over-bright, grainy background |

It is free — no retraining — and lives entirely in `model.sample`.

### Tissue channels must be passed at sampling

The 4 tissue channels (brain / CSF / extracranial) are learned SPADE inputs. If they
are omitted at sampling, brain and CSF collapse to the same grey (~24 HU). `generate.py`
always passes them; `--denoise 0.5` then removes the fine single-voxel speckle the
diffusion leaves in the parenchyma, matching real CT noise while protecting
vessels and bone.

---

## Training (`train_gen.py`)

Patch-based (96³), v-prediction, DDPM noising / DDIM sampling. The occlusion
augmentation (`oclusiones.py`) precomputes edit *recipes* per donor and materialises
them per patch, so the model sees variant × occlusion pairs that do not exist in the
raw data.

```bash
python oclusiones.py --workers 4 --por-donante 6      # once, precompute recipes
python train_gen.py --resume --epochs 800 --bs 3 --patch 96
```

Two things that cost real time to learn, documented in the code:

- **Do not exceed VRAM.** At 96³, `--bs 3` peaks at ~37 GB. On Windows, exceeding the
  card's memory does **not** raise OOM — the driver overflows to system RAM over PCIe,
  running ~68× slower and eventually crashing the machine (`HYPERVISOR_ERROR` under
  VBS/HVCI). A guard aborts if a step would overflow; `vigilar.ps1` auto-resumes the
  run after a crash.
- **Validation loss does not tell you if the vessels are there.** Watch the
  vessel–parenchyma contrast in HU (PNGs written to `mon/`), not the MSE.

---

## Data

209 de-identified CTA volumes (TopAneu, TopCoW, ISLES). **No patient data is
distributed** — this repository is code and pretrained weights only. See the
`.gitignore` (whitelist strategy: everything is ignored except source files).

## Files

| file | role |
|------|------|
| `generate.py` | inference API + CLI (variant × occlusion → CTA) |
| `model.py` | the SPADE diffusion network, schedulers, guided sampler |
| `train_gen.py` | training loop + occlusion augmentation |
| `oclusiones.py` | precomputed occlusion recipes |
| `cow_edit.py` | CoW variants and occlusion as vessel-graph edits |
| `hybrid.py` | renders an edited vessel tree onto a donor CTA (training pairs) |
| `tejidos.py` | derives the brain / CSF / extracranial tissue channels |
| `common.py` | shared constants, HU/normalisation, I/O |
| `vigilar.ps1` | training supervisor: auto-resumes after a machine crash |

## Limitations

- Occlusion is **graph reachability, not perfusion**: a filiform Pcom rescues a
  territory as well as a robust one, and **pial collaterals are not modelled** (an M1
  occlusion turns off M2/M3 entirely).
- Vessel intensity and parenchyma texture are controlled at sampling (`guidance`,
  `denoise`), not learned — tune them per use case.
- A CUDA GPU is required for both sampling and training.
