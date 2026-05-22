# Etsy AI Image Detection — Analysis

Binary image-classification project that distinguishes **authentic photographs** from
**AI-generated images** (diffusion / GAN output) for an Etsy-style marketplace dataset.

The work explores multiple feature representations and model families, then combines
them into Optuna-optimized weighted ensembles. The best validation **F1 score is ≈ 0.941**.

---

## Project contents

| File | Description |
|------|-------------|
| `Etsy_final.ipynb` | Main notebook — full pipeline (data prep → models → ensembles → submission). |
| `Etsy_AI_Image_Detection_Analysis.pdf` | Exported report of the analysis. |
| `Comaprison_model.png` | Bar chart comparing F1 of every model and ensemble. |
| `Confusion_matrix.png` | Confusion matrix of the best ensemble on the validation set. |
| `t-SNE_plot.png` | t-SNE projection of intermediate-CLIP features (class separability). |
| `fft_comparison.png` | Frequency-domain (FFT) comparison of authentic vs AI images. |
| `ELA_plot.png` | Error Level Analysis visualization of authentic vs AI images. |
| `model_confidence_analysis.png` | Prediction-confidence distribution of the ensemble. |

---

## Problem & dataset

- **Task:** binary classification — `0 = authentic`, `1 = ai_generated`.
- **Metric:** F1 score.
- **Data** (after dropping rows with missing image files):
  - Train: **4,340** images — 2,240 AI-generated / 2,100 authentic.
  - Test: **1,860** images.
  - Stratified 80/20 split → **3,472 train / 868 validation**.

The notebook reads `train (1).csv` and `test (1).csv` (columns: `image_id`,
`ground_truth`) and loads images from a Google Drive folder `images_final_sample`.

---

## Approach

The core idea: AI-generated images leave **two complementary kinds of fingerprints** —
high-level semantic anomalies and low-level texture/frequency artifacts. The pipeline
extracts both and fuses them.

### Individual models

| Model | Signal captured | Val F1 |
|-------|-----------------|:------:|
| FFT features + XGBoost | Frequency-domain artifacts | 0.603 |
| ELA features + XGBoost | JPEG re-compression artifacts | 0.652 |
| DINOv2 (ViT-L/14) + XGBoost | Self-supervised visual features | 0.790 |
| EfficientNet-B3 (fine-tuned) | Low-level texture | 0.791 |
| DINOv2 + MLP head | Self-supervised features | 0.835 |
| EfficientNet-B0 (fine-tuned) | Low-level texture | 0.840 |
| EfficientNet-B0 + TTA | Texture + test-time augmentation | 0.854 |
| CLIP (ViT-L/14) final layer + XGBoost | High-level semantics | 0.848 |
| **Intermediate CLIP (layers 18/20/22/23) + XGBoost** | **Mid-level semantics** | **0.923** |

The **intermediate-CLIP** representation (a 4,096-dim concatenation of hidden-layer
activations) is the strongest single model by a wide margin.

### Enhancements

- **Test-Time Augmentation (TTA)** — 5 transforms averaged at inference.
- **Pseudo-labeling** — high-confidence test predictions (≥0.95 / ≤0.05) fed back as
  extra training data, including an iterative multi-round variant.
- **Mega feature fusion** — CLIP + intermediate-CLIP + DINOv2 + FFT concatenated into a
  6,018-dim vector for a single XGBoost model.
- **5-fold cross-validation** — CLIP+XGBoost, mean F1 **0.872 ± 0.014** (stable).

### Ensembles

Model probabilities are blended with **Optuna**-tuned weights and a tuned decision
threshold:

| Ensemble | Val F1 |
|----------|:------:|
| 4-model Optuna ensemble | **0.941** |
| 5-model Optuna ensemble | 0.940 |
| 7-model "Ultimate" ensemble | 0.940 |
| 6-model "Ultimate V2" ensemble | 0.933 |

Each ensemble run writes a `submission*.csv` with `image_id` + `prediction`.

---

## Tech stack

- **Python 3**, originally run on **Google Colab** with a CUDA GPU.
- PyTorch / torchvision — EfficientNet fine-tuning.
- `open_clip` — CLIP ViT-L/14 (`laion2b_s32b_b82k`).
- `torch.hub` — DINOv2 ViT-L/14.
- XGBoost — classifier on extracted features.
- scikit-learn — splits, metrics, scaling, t-SNE.
- Optuna — ensemble weight / threshold optimization.
- OpenCV, Pillow, NumPy, pandas, matplotlib, tqdm.
- `pytorch-grad-cam` — model interpretability.

---

## How to run

1. **Open the notebook** in Google Colab (recommended — it expects a GPU and mounts
   Google Drive) or a local Jupyter environment.

2. **Install dependencies** (the first cell handles this on Colab):

   ```bash
   pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
   pip install transformers open-clip-torch xgboost scikit-learn pandas tqdm pillow matplotlib
   pip install optuna pytorch-grad-cam opencv-python
   ```

3. **Provide the data** and update the paths in the first code cell:
   - `TRAIN_CSV` / `TEST_CSV` — the train/test CSV files.
   - `IMAGES_DIR` — folder containing the image files referenced by `image_id`.

4. **Run the cells top to bottom.** Feature extraction and fine-tuning are GPU-heavy;
   later ensemble cells reuse variables from earlier cells, so run in order.

5. **Outputs:** submission CSVs and the analysis plots are saved to the working
   directory.

---

## Key takeaways

- **Intermediate transformer layers beat the final layer** — mid-level CLIP activations
  carry the most discriminative signal for AI-detection (0.923 vs 0.848 F1).
- **Frequency/compression cues are weak alone** (FFT 0.60, ELA 0.65) but still add a
  small lift inside the ensemble.
- **Ensembling semantic + texture models** is what pushes performance from ~0.92 to
  ~0.94 — the signal types are genuinely complementary.
- Aggressive pseudo-labeling shows **diminishing/negative returns** after the first
  round, so it is applied conservatively.
