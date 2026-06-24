# Generalization harness — does the Etsy detector survive unseen generators?

A single self-contained Colab notebook, **`generalization_eval.ipynb`**, that runs
the **frozen** Etsy detector over generators it never trained on (GANs, diffusion,
DALL·E, Midjourney, …) and reports **per-generator** metrics. The finding is
*where the detector holds up and where it collapses* — the single 0.941 F1 hides
this. Nothing is retrained.

## Why this isn't a one-line "load checkpoint"

Your detector is **not** a single network, and the repo saves **no artifacts**
(`.gitignore` excludes every checkpoint/pickle/csv). The model is a multi-stage
ensemble trained in-memory in `notebooks/Etsy_final.ipynb`:

| Signal | Used by |
|--------|---------|
| Intermediate-CLIP (resblocks 18/20/22/23 CLS) → XGBoost | `single` **and** `ensemble` |
| CLIP ViT-L/14 final embedding → XGBoost | `ensemble` |
| EfficientNet-B0 fine-tuned + 5-way TTA | `ensemble` |
| FFT radial power spectrum → scaler → XGBoost | `ensemble` |

So the notebook's Step 1 **serializes** the trained pieces from the training
notebook (they don't exist yet); the rest reloads them frozen and evaluates.

## Layered modes

- **`single`** — Intermediate-CLIP + XGBoost only (strongest single model, 0.923
  val F1). One artifact, deterministic. **Run this first** — already a publishable
  per-generator story.
- **`ensemble`** — full 4-model Optuna blend (0.941) with tuned weights/threshold.

## How to use

1. Open **`generalization_eval.ipynb`** in Colab.
2. **Step 1 (once):** copy that cell into the **end** of `Etsy_final.ipynb` and run
   it there — it serializes artifacts to Drive (`/content/drive/MyDrive/etsy_artifacts`).
3. **Step 2:** set the config cell (`ARTIFACTS_DIR`, `DATA_DIR`, `MODE`, …).
4. **Step 4/5:** lay out `DATA_DIR` as one folder per source — `real/` plus one
   folder per generator (helpers for GenImage / Synthbuster are included).
5. **Step 6–7:** run; it prints the table (worst-AUROC first) and writes
   `reports/generalization_<mode>.csv / .md / .json`.

```
DATA_DIR/
├── real/         authentic photos      (label 0)
├── sd15/         Stable Diffusion 1.5  (label 1)
├── sdxl/
├── dalle3/
└── midjourney/
```

## Datasets (Step 5 in the notebook)

- **GenImage** — Midjourney, SD1.4/1.5, Wukong, VQDM, ADM, GLIDE, BigGAN; real =
  ImageNet. arxiv.org/abs/2306.08571 (mirrors on HF Hub).
- **Synthbuster** — DALL·E 2/3, Firefly, Midjourney v5, SD1.3/1.4/2/XL, GLIDE;
  real = RAISE-1k. arxiv.org/abs/2304.06184 (Zenodo).
- **ForenSynths / UFD** — pure-GAN suite.

Repo IDs / Zenodo records drift, so they're notebook arguments — verify against
the canonical pages above.

## Gotchas that decide whether the numbers mean anything

1. **Preprocessing matches training by construction** — the extractor cell is
   copied from the notebook (CLIP→224, FFT→256, ImageNet norm). Don't add resizes.
2. **Per generator, never the average** — the notebook refuses to collapse rows;
   AUROC-sorted output puts the collapse at the top.
3. **Match real content** to your training domain (Etsy-style product photos),
   else you measure content shift, not generator shift.
4. **Compression confound** — if `real/` is PNG and a generator folder is JPEG,
   you may measure compression, not generation. Normalize across classes.
5. **Threshold may not transfer** — `single` uses 0.5, `ensemble` the tuned
   threshold; AP/AUROC are threshold-free, trust those.
