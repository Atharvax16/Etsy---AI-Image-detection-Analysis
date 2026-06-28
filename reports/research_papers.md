# AI-Generated Image Detection — A Deep-Dive Research Map

A sequential, annotated reading list to understand the **whole** field of AI-generated
image detection — not just the pieces already in this repo, but the foundations beneath
them and the frontiers beyond them. Built around the Etsy detector
(`notebooks/Etsy_final.ipynb`: intermediate-CLIP + XGBoost, fused with CLIP-final,
DINOv2, EfficientNet, FFT/ELA via Optuna ensembles → 0.941 F1) and its cross-generator
generalization harness (`generalization/generalization_eval.ipynb`).

Every entry follows the same shape:

- **Idea** — the one-line contribution.
- **Why it matters** — its place in the field.
- **Connects to your work** — the explicit tie to this repo (only where real).
- **Try** — a concrete experiment it suggests for your detector (where there is one).

arXiv IDs are web-verified (June 2026). Read top-to-bottom for a guided tour, or jump to
the section that matches what you're building.

---

## How to read this map (the 5 mental models)

The whole field is really five recurring ideas in tension. Hold these and every paper
slots into place:

1. **Low-level vs. high-level signal.** Fakes betray themselves with *texture/frequency
   artifacts* (low level — §2, §3) AND *semantic implausibility* (high level — §5). Your
   pipeline already fuses both; that's the central bet of the field.
2. **Train-a-classifier vs. measure-a-distance.** Either learn real-vs-fake directly
   (§4), or use a *frozen* feature space / reconstruction process and measure how far an
   image sits from "real" (§5, §6). The frozen-feature camp generalizes better — your
   intermediate-CLIP model lives here.
3. **Passive detection vs. active provenance.** Detect after the fact (everything above),
   or *mark* images at generation time so origin is verifiable (§10). These are rival
   answers to the same societal problem.
4. **Generalization is the whole game.** A detector that scores 0.99 on its training
   generator and 0.55 on the next one is worthless. §4, §7, and §12 are all about this.
5. **It's an arms race.** Every artifact gets fixed by the next generator; every detector
   gets evaded (§9). "Solved on this benchmark" never means "solved."

---

## §1 — Orientation: surveys & the threat model

Start here to get the lay of the land before diving into methods.

**[1] Methods and Trends in Detecting AI-Generated Images: A Comprehensive Review** —
Mahara & Rishe, 2025 ([arXiv:2502.15176](https://arxiv.org/abs/2502.15176))
- **Idea:** Up-to-date taxonomy of synthetic-image detection — spatial, frequency,
  foundation-model, training-free, and reasoning/LLM-based approaches.
- **Why it matters:** The single best modern map; covers diffusion-era methods that older
  deepfake surveys miss.
- **Connects to your work:** Read it to locate where each of your models (FFT, ELA, CLIP,
  EfficientNet) sits in the broader taxonomy.

**[2] Are GAN Generated Images Easy to Detect? A Critical Analysis of the
State-of-the-Art** — Gragnaniello, Cozzolino, Marra, Poggi, Verdoliva, 2021
([arXiv:2104.02617](https://arxiv.org/abs/2104.02617))
- **Idea:** Dissects *why* the best GAN detectors of the era worked — augmentation, no
  resizing, avoiding down-sampling that destroys the fingerprint.
- **Why it matters:** The "key ingredients" paper; its lessons (don't resize, augment
  hard) still govern how you should preprocess.
- **Connects to your work:** Validates the harness gotcha "don't add resizes" — preprocessing
  must match training by construction.

**[3] The Unwinnable Arms Race of AI Image Detection** — 2025
([arXiv:2509.21135](https://arxiv.org/abs/2509.21135))
- **Idea:** Argues passive detection is structurally chasing a moving target as generators
  improve.
- **Why it matters:** The honest framing of the whole enterprise — read it to stay
  humble about any single F1 number.
- **Connects to your work:** This is exactly why your generalization harness (not the
  0.941) is the real story.

**[4] Is AI-Generated Image Detection a Solved Problem?** — 2025
([arXiv:2505.12335](https://arxiv.org/abs/2505.12335))
- **Idea:** Stress-tests "near-perfect" detectors under realistic in-the-wild conditions
  and finds large drops.
- **Why it matters:** A reality check on benchmark optimism.
- **Try:** Use its evaluation protocol as a template for an honest "in-the-wild" slice of
  your own eval.

---

## §2 — Frequency & spectral artifacts (connects to your FFT model)

Why generated images look wrong in the Fourier/DCT domain — the basis of your FFT feature.

**[5] Watch Your Up-Convolution: CNN-Based Generative DNNs Fail to Reproduce Spectral
Distributions** — Durall, Keuper & Keuper, CVPR 2020
([arXiv:2003.01826](https://arxiv.org/abs/2003.01826))
- **Idea:** Transposed/up-convolution leaves systematic spectral distortions; generators
  can't match the high-frequency power spectrum of real images.
- **Why it matters:** The mechanistic *root cause* behind frequency-based detection — not
  just "there's a difference" but *why*.
- **Connects to your work:** This is the theory under your FFT radial-power-spectrum
  feature. Understanding it tells you what your FFT model is actually keying on.

**[6] Leveraging Frequency Analysis for Deep Fake Image Recognition** — Frank et al.,
ICML 2020 ([arXiv:2003.08685](https://arxiv.org/abs/2003.08685))
- **Idea:** A DCT representation exposes grid-like GAN artifacts that are consistent
  across architectures, datasets, and resolutions.
- **Why it matters:** Showed frequency features generalize across generators surprisingly
  well — and are cheap.
- **Try:** Add a **DCT** representation alongside your FFT radial spectrum; DCT often
  separates JPEG-grid artifacts from generation artifacts more cleanly.

**[7] On the Detection of Synthetic Images Generated by Diffusion Models** — Corvi,
Cozzolino, Zingarini, Poggi, Nagano, Verdoliva, ICASSP 2023
([arXiv:2211.00680](https://arxiv.org/abs/2211.00680))
- **Idea:** Diffusion images carry *different* spectral fingerprints than GANs; some
  (Stable Diffusion) are strong, others (DALL·E 2, ADM) weak.
- **Why it matters:** Warns that the GAN-era frequency intuition only partially transfers
  to diffusion — directly relevant to your diffusion-heavy data.
- **Connects to your work:** Explains why your FFT model alone is weak (0.60 F1) on a
  diffusion-mixed set — the artifact is fainter than in the GAN era.

---

## §3 — Forensics & low-level fingerprints (connects to your ELA model)

Noise residuals, compression traces, and per-generator fingerprints — the forensics
lineage your ELA feature descends from.

**[8] Do GANs Leave Artificial Fingerprints?** — Marra, Gragnaniello, Verdoliva, Poggi,
2019 ([arXiv:1812.11842](https://arxiv.org/abs/1812.11842))
- **Idea:** Each GAN leaves a stable, model-specific noise fingerprint, analogous to a
  camera's PRNU sensor pattern.
- **Why it matters:** Founded the "generation leaves a signature" view — basis for both
  detection and attribution.
- **Connects to your work:** ELA is a coarse cousin of residual-fingerprinting; this is
  the rigorous version of the same intuition.

**[9] Attributing Fake Images to GANs: Learning and Analyzing GAN Fingerprints** — Yu,
Davis, Fritz, ICCV 2019 ([arXiv:1811.08180](https://arxiv.org/abs/1811.08180))
- **Idea:** Learns fingerprints that not only detect but *attribute* an image to its
  source GAN — even across minor training differences.
- **Why it matters:** Reframes the problem from binary detection to multi-class
  attribution (which model made this?).
- **Try:** Turn your generalization harness into an **attribution** experiment — can your
  intermediate-CLIP features cluster images by generator (t-SNE you already use)?

**[10] Error Level Analysis (ELA) — classical JPEG forensics**
- **Idea:** Re-compress an image at known quality; uniformly-edited/synthetic regions
  show different recompression error than authentic camera regions.
- **Why it matters:** A pre-deep-learning forensic staple; cheap and interpretable.
- **Connects to your work:** This *is* your ELA feature (0.65 F1). Note its weakness:
  it mostly detects *recompression boundaries*, which AI images may or may not have —
  hence the modest score. Read [37] (JPEG bias) before trusting it.

---

## §4 — Learned CNN detectors & the generalization question

The seminal line that defined "do detectors generalize?" — the question your harness asks.

**[11] CNN-Generated Images Are Surprisingly Easy to Spot... For Now** — Wang, Wang,
Zhang, Owens, Efros, CVPR 2020 ([arXiv:1912.11035](https://arxiv.org/abs/1912.11035))
- **Idea:** A single ResNet trained on *one* generator (ProGAN) with aggressive
  augmentation (blur + JPEG) generalizes to 11 unseen CNN generators.
- **Why it matters:** The foundational generalization paper — and "...for now" foreshadowed
  the arms race. Its augmentation recipe is still a baseline.
- **Connects to your work:** The intellectual ancestor of your generalization harness.
- **Try:** Replicate their **blur+JPEG augmentation** during your EfficientNet fine-tuning
  and re-run the harness — it's the cheapest known generalization booster.

**[12] Rethinking the Up-Sampling Operations in CNN-Based Generative Networks for
Generalizable Deepfake Detection (NPR)** — Tan et al., CVPR 2024
([arXiv:2312.10461](https://arxiv.org/abs/2312.10461))
- **Idea:** "Neighboring Pixel Relationships" — a tiny local feature capturing
  up-sampling artifacts — generalizes from ProGAN training to 28 GAN *and* diffusion
  generators.
- **Why it matters:** Shows a hand-crafted local artifact can rival heavy models on
  generalization, at near-zero cost.
- **Try:** Add **NPR** as a cheap ensemble member; it's complementary to your semantic
  CLIP signal and trivial to compute.

**[13] Rich and Poor Texture Contrast: A Simple yet Effective Approach for AI-Generated
Image Detection** — 2024 ([arXiv:2311.12397](https://arxiv.org/abs/2311.12397))
- **Idea:** Contrast between texture-rich and texture-poor patches exposes generation
  artifacts robustly.
- **Why it matters:** Another strong, cheap, generalizable hand-crafted signal.
- **Try:** A candidate texture feature to concatenate into your "mega fusion" vector.

---

## §5 — Foundation-model features (the heart of your method)

Why frozen CLIP/DINO features detect fakes, and why *intermediate* layers win — the core
of your 0.923 single model.

**[14] Towards Universal Fake Image Detectors that Generalize Across Generative Models** —
Ojha, Li, Lee, CVPR 2023 ([arXiv:2302.10174](https://arxiv.org/abs/2302.10174))
- **Idea:** Don't *train* a real-vs-fake net (it overfits its generator). Instead do
  nearest-neighbor / linear probing in a **frozen CLIP** feature space → big
  generalization gains to unseen diffusion/autoregressive models.
- **Why it matters:** The paper that legitimized "frozen foundation features + light
  classifier" — the exact recipe of your strongest model.
- **Connects to your work:** This is the theoretical backbone of CLIP-features → XGBoost.
  Your XGBoost is the "light classifier"; CLIP is the frozen space.
- **Try:** Add their **linear-probe** baseline as a sanity check against your XGBoost head.

**[15] Raising the Bar of AI-Generated Image Detection with CLIP** — Cozzolino, Poggi,
Corvi, Nießner, Verdoliva, CVPR 2024 Workshops
([arXiv:2312.00195](https://arxiv.org/abs/2312.00195))
- **Idea:** A *handful* of examples from a single generator + CLIP features yields a
  detector robust to DALL·E-3, Midjourney v5, Firefly. Big domain-specific datasets are
  neither necessary nor helpful.
- **Why it matters:** Directly challenges the "more data = better" reflex; argues for
  few-shot CLIP probing.
- **Connects to your work:** Suggests your 4,340-image training set may be *overkill* for
  the CLIP branch — and possibly hurting cross-generator transfer.
- **Try:** Train a CLIP head on a tiny single-generator subset and run the harness — does
  it generalize *better* than your full-data model?

**[16] DE-FAKE: Detection and Attribution of Fake Images Generated by Text-to-Image
Diffusion Models** — Sha, Li, Yu, Zhang, ACM CCS 2023
([arXiv:2210.06998](https://arxiv.org/abs/2210.06998))
- **Idea:** Combines image **and prompt** features (via CLIP) for detection + attribution;
  shows the caption carries detectable signal.
- **Why it matters:** Opens the multimodal angle — text conditioning leaves traces.
- **Try:** For Etsy listings you often have a **title/description**. Fuse CLIP *text*
  embeddings of the listing with your image features — a signal you currently ignore.

**[17] CLIP: Learning Transferable Visual Models From Natural Language Supervision** —
Radford et al., 2021 ([arXiv:2103.00020](https://arxiv.org/abs/2103.00020))
- **Idea:** Contrastive image-text pretraining yields a general-purpose visual feature
  space.
- **Why it matters:** The backbone every §5 detector (including yours) stands on.
- **Connects to your work:** Understanding CLIP's training explains *why* intermediate
  layers (18/20/22/23) carry more discriminative mid-level signal than the final,
  text-aligned layer — your single biggest empirical finding (0.923 vs 0.848).

**[18] DINOv2: Learning Robust Visual Features Without Supervision** — Oquab et al., 2023
([arXiv:2304.07193](https://arxiv.org/abs/2304.07193))
- **Idea:** Self-supervised ViT features that transfer without fine-tuning.
- **Why it matters:** The main *alternative* frozen backbone to CLIP for detection.
- **Connects to your work:** Your DINOv2 branch (0.79–0.835 F1). It underperforms CLIP
  here — interesting, since DINO has no text bias.
- **Try:** You used DINOv2's final features. Apply your **intermediate-layer** trick to
  DINOv2 too — the same insight that won for CLIP may lift DINO above 0.84.

---

## §6 — Diffusion-specific detection (a new angle you haven't tried)

Reconstruction/inversion methods exploit the *generative process itself* — no fingerprint
classifier needed. This is the most promising untried direction for your pipeline.

**[19] DIRE: Diffusion Reconstruction Error for Diffusion-Generated Image Detection** —
Wang, Bao, Zhou et al., ICCV 2023 ([arXiv:2303.09295](https://arxiv.org/abs/2303.09295))
- **Idea:** Run an image through a pretrained diffusion model's invert-then-reconstruct
  loop. *Generated* images reconstruct with low error; *real* images don't. The error map
  is the feature.
- **Why it matters:** A fundamentally different, generalizable signal that needs no
  per-generator training labels.
- **Try:** Compute **DIRE** features on your data and add them as an ensemble member —
  they're orthogonal to CLIP semantics and could lift the diffusion-heavy slice.

**[20] AEROBLADE: Training-Free Detection of Latent Diffusion Images Using Autoencoder
Reconstruction Error** — Ricker, Lukovnikov, Fischer, CVPR 2024
([arXiv:2401.17879](https://arxiv.org/abs/2401.17879))
- **Idea:** Latent-diffusion images pass cleanly through the model's *own VAE* (low
  reconstruction error); real images don't. **Zero training.**
- **Why it matters:** Cheaper than DIRE (one VAE pass, no full inversion) and nearly
  matches trained detectors. Also localizes inpainted regions.
- **Try:** The single highest-ROI experiment here — a **training-free** AEROBLADE score
  you can compute today and blend into your Optuna ensemble. Great robustness baseline.

**[21] Diffusion Noise Feature (DNF): Accurate and Fast Generated Image Detection** —
2024 ([arXiv:2312.02625](https://arxiv.org/abs/2312.02625))
- **Idea:** Uses the estimated noise from the diffusion inversion (not pixel
  reconstruction error) as the discriminative feature.
- **Why it matters:** Faster, often more accurate variant of the reconstruction family.
- **Try:** If DIRE is too slow, DNF is the speed-optimized alternative to benchmark.

**[22] Recent Advances on Generalizable Diffusion-Generated Image Detection** — 2025
([arXiv:2502.19716](https://arxiv.org/abs/2502.19716))
- **Idea:** Focused survey of the diffusion-detection subfield.
- **Why it matters:** Your one-stop catch-up on everything in §6 beyond these anchors.

---

## §7 — Benchmarks & cross-generator generalization (connects to your harness)

The datasets that *define* generalization — what your harness evaluates on.

**[23] GenImage: A Million-Scale Benchmark for Detecting AI-Generated Images** — 2023
([arXiv:2306.08571](https://arxiv.org/abs/2306.08571))
- **Idea:** ~1M real/fake pairs across 8 generators (Midjourney, SD1.4/1.5, Wukong, VQDM,
  ADM, GLIDE, BigGAN); real = ImageNet.
- **Why it matters:** The standard cross-generator benchmark.
- **Connects to your work:** A primary data source for your harness (cited in
  `generalization/README.md`).

**[24] Synthbuster: Towards Detection of Diffusion Model Generated Images** — 2023
([arXiv:2304.06184](https://arxiv.org/abs/2304.06184))
- **Idea:** RAISE-1k real images + matched DALL·E 2/3, Firefly, Midjourney, SD variants,
  GLIDE — content-controlled so you measure *generator* shift, not content shift.
- **Why it matters:** Cleaner generalization signal than GenImage for the content-confound
  reason.
- **Connects to your work:** Your harness's second data source; its content-control design
  is exactly your gotcha #3 ("match real content").

**[25] Community Forensics: Using Thousands of Generators to Train Fake Image Detectors** —
Park & Owens, CVPR 2025 ([arXiv:2411.04125](https://arxiv.org/abs/2411.04125))
- **Idea:** 2.7M images from **4,803** generators; detection accuracy rises with generator
  *diversity* in training.
- **Why it matters:** Reframes generalization as a *data-diversity* problem — more
  generators beats cleverer models.
- **Try:** If your harness shows collapse on unseen generators, the fix this paper implies
  is "train on more *kinds* of fakes," not "tune the ensemble harder."

**[26] Aligned Datasets Improve Detection of Latent Diffusion-Generated Images** — 2024
([arXiv:2410.11835](https://arxiv.org/abs/2410.11835))
- **Idea:** When real and fake sets are *aligned* in content, detectors learn generation
  artifacts instead of content shortcuts.
- **Why it matters:** Methodological hygiene — directly about not measuring the wrong
  thing.
- **Connects to your work:** Reinforces harness gotchas #3/#4; consider re-pairing your
  Etsy train set so real/fake share content.

---

## §8 — Ensembling, calibration & semi-supervision (your systems layer)

The "make it actually work" tricks in your pipeline, with the canonical references.

**[27] XGBoost: A Scalable Tree Boosting System** — Chen & Guestrin, KDD 2016
([arXiv:1603.02754](https://arxiv.org/abs/1603.02754))
- **Idea:** Regularized gradient-boosted trees, the workhorse classifier on top of your
  extracted features.
- **Connects to your work:** Your classification head on CLIP/DINO/FFT features.

**[28] Optuna: A Next-Generation Hyperparameter Optimization Framework** — Akiba et al.,
KDD 2019 ([arXiv:1907.10902](https://arxiv.org/abs/1907.10902))
- **Idea:** Define-by-run Bayesian/TPE hyperparameter search.
- **Connects to your work:** Tunes your ensemble weights + decision threshold (0.923 →
  0.941). Worth understanding TPE so you know what it's actually optimizing — and that
  tuning the threshold on validation risks overfitting it (harness gotcha #5).

**[29] EfficientNet: Rethinking Model Scaling for CNNs** — Tan & Le, ICML 2019
([arXiv:1905.11946](https://arxiv.org/abs/1905.11946))
- **Idea:** Compound scaling of depth/width/resolution.
- **Connects to your work:** Your fine-tuned EfficientNet-B0/B3 texture branch + TTA.

> **On TTA, pseudo-labeling, and calibration:** these are general semi-supervised /
> inference techniques rather than detection-specific papers. Key principle to internalize:
> pseudo-labeling amplifies a detector's *existing biases* — if your model is overconfident
> on one generator, self-training entrenches that. This is exactly the diminishing/negative
> return you observed after round 1. Before trusting ensemble probabilities, check
> **calibration** (reliability diagrams / ECE) — your `model_confidence_analysis.png` is the
> right instinct; formalize it.

---

## §9 — Robustness & adversarial attacks (the field's biggest weakness)

Every detector above can be broken. This is essential, sobering reading — and a new angle
for you to stress-test against.

**[30] Evading Deepfake-Image Detectors with White- and Black-Box Attacks** — Carlini &
Farid, CVPR 2020 Workshops ([arXiv:2004.00622](https://arxiv.org/abs/2004.00622))
- **Idea:** Imperceptible adversarial perturbations flip state-of-the-art detectors, in
  both white-box and black-box settings.
- **Why it matters:** The wake-up call — detectors that over-rely on high-frequency cues
  are trivially evadable.
- **Connects to your work:** Your FFT/ELA branches are precisely the high-frequency
  detectors this attack targets.
- **Try:** Run a simple **FGSM/PGD** attack on your ensemble to measure how fragile 0.941
  really is. Then try adversarial training as a defense.

**[31] Robustness of AI-Image Detectors: Fundamental Limits and Practical Attacks** —
Saberi et al., ICLR 2024 ([arXiv:2310.00076](https://arxiv.org/abs/2310.00076))
- **Idea:** Proves a fundamental evasion/spoofing trade-off; "diffusion purification"
  attacks remove detector signal, and real images can be *spoofed* as fake.
- **Why it matters:** Theoretical limits, not just one attack — bounds what *any* detector
  can promise.
- **Try:** Apply a **diffusion-purification** pass (light img2img) to your fakes and watch
  the F1 fall — a realistic "laundering" threat for a marketplace.

**[32] AI-Generated Image Detectors Overrely on Global Artifacts: Evidence from Inpainting
Exchange** — 2026 ([arXiv:2602.00192](https://arxiv.org/abs/2602.00192))
- **Idea:** Swapping inpainted regions shows detectors lean on global, not local, cues —
  a brittleness diagnostic.
- **Why it matters:** Tells you *what* your detector is really keying on (and how a
  partially-edited Etsy photo could fool it).
- **Connects to your work:** A real marketplace case — sellers AI-edit *part* of a real
  photo. Test your detector on partial manipulations, not just fully-synthetic images.

---

## §10 — Provenance & watermarking (the other paradigm)

Instead of detecting after the fact, *mark at generation*. A different answer to the same
problem — worth understanding even if you stay on the detection side.

**[33] The Stable Signature: Rooting Watermarks in Latent Diffusion Models** — Fernandez,
Couairon, Jégou, Douze, Furon, ICCV 2023
([arXiv:2303.15435](https://arxiv.org/abs/2303.15435))
- **Idea:** Fine-tune the LDM decoder so every generated image carries an invisible,
  extractable signature — origin verifiable at <10⁻⁶ false-positive rate.
- **Why it matters:** Proactive provenance from cooperating model owners.
- **Connects to your work:** The complement to your passive detector — works only if the
  generator cooperates, which adversarial Etsy sellers won't.

**[34] Tree-Ring Watermarks: Fingerprints for Diffusion Images that Are Invisible and
Robust** — Wen, Kirchenbauer, Geiping, Goldstein, 2023
([arXiv:2305.20030](https://arxiv.org/abs/2305.20030))
- **Idea:** Embed a Fourier-space pattern into the *initial noise*; detect by inverting
  the diffusion process. Robust to crops, flips, rotations.
- **Why it matters:** A cleverer, more robust watermark than post-hoc pixel marks.

**[35] Invisible Image Watermarks Are Provably Removable Using Generative AI** — 2023
([arXiv:2306.01953](https://arxiv.org/abs/2306.01953))
- **Idea:** Regeneration attacks can strip "invisible" watermarks.
- **Why it matters:** The rebuttal to §10 — watermarking has the same arms-race problem as
  detection. Read [33]/[34] and this together.

> **Industry provenance (not arXiv, but read these):** the **C2PA** standard
> (c2pa.org — cryptographically signed content credentials, backed by Adobe/Microsoft/etc.)
> and **Google DeepMind SynthID** (watermarking + detection for Imagen/Veo outputs). These
> are the deployed, real-world face of §10 and where marketplace policy is heading — relevant
> if Etsy ever mandates provenance metadata.

---

## §11 — Interpretability: what the artifacts actually are (connects to your Grad-CAM)

Don't just classify — understand *where* and *why*.

**[36] Grad-CAM: Visual Explanations from Deep Networks via Gradient-Based Localization** —
Selvaraju et al., ICCV 2017 ([arXiv:1610.02391](https://arxiv.org/abs/1610.02391))
- **Idea:** Use gradients into the last conv layer to produce a heatmap of what drove a
  prediction.
- **Why it matters:** The standard interpretability tool; tells you which image regions
  scream "fake."
- **Connects to your work:** Your `pytorch-grad-cam` analysis. Combine it with [32]'s
  finding: if your Grad-CAM lights up globally rather than on a specific artifact, you've
  confirmed the "global over-reliance" brittleness on your own model.

---

## §12 — Dataset bias, frontiers & a punch-list of things to try

The methodological traps, and a synthesized to-do list mapping all of the above onto
*your* pipeline.

**[37] Fake or JPEG? Revealing Common Biases in Generated Image Detection Datasets** —
Grommelt, Weiss, Pfreundt, Keuper, ECCV Workshops 2024
([arXiv:2403.17608](https://arxiv.org/abs/2403.17608))
- **Idea:** GenImage and friends are biased by **JPEG compression and image size** —
  detectors learn the compression, not the generation. Removing the bias changes
  cross-generator results by 11+ points.
- **Why it matters:** The most important methodological warning for *your* harness.
- **Connects to your work:** This is harness gotcha #4 ("compression confound") made
  rigorous. If your `real/` and generator folders differ in JPEG quality, your numbers may
  be measuring compression. **Read before trusting any harness result.**

**[38] A Bias-Free Training Paradigm for More General AI-Generated Image Detection** — 2024
([arXiv:2412.17671](https://arxiv.org/abs/2412.17671))
- **Idea:** Construct training data so content/compression are controlled, forcing the
  model onto true generation artifacts.
- **Why it matters:** The constructive fix to [37]'s problem.
- **Try:** Re-pair your Etsy train set (same content, same JPEG quality across real/fake)
  and re-measure — a likely large generalization gain.

### Your punch-list (ideas drawn from the whole map)

Ranked roughly by return-on-effort for *this* repo:

1. **Add a training-free reconstruction score** — AEROBLADE [20] (cheap) or DIRE [19] /
   DNF [21] — as a new Optuna ensemble member. Orthogonal to CLIP semantics; likely lifts
   the diffusion slice. *(§6)*
2. **Audit the compression confound** [37][38] — verify real/fake share JPEG quality and
   size before trusting the harness; re-pair if not. Possibly your single biggest hidden
   bug. *(§12)*
3. **Stress-test robustness** [30][31][32] — FGSM/PGD + diffusion-purification + partial
   inpainting. Report 0.941 *and* its post-attack value; the gap is the real risk story
   for a marketplace. *(§9)*
4. **Apply the intermediate-layer trick to DINOv2** [18] — the insight that won for CLIP
   may close DINO's gap and add a stronger, decorrelated ensemble member. *(§5)*
5. **Add the listing text** [16] — fuse CLIP *text* embeddings of Etsy titles/descriptions;
   a free modality you currently ignore. *(§5)*
6. **Cheap hand-crafted artifact features** — NPR [12], rich/poor texture [13], DCT [6] —
   each trivial to compute and known to generalize from ProGAN to diffusion. *(§2, §4)*
7. **Turn the harness into attribution** [9] — cluster/t-SNE features by generator; "which
   model made this" is more useful to Etsy than a single bit. *(§3)*
8. **Replicate the blur+JPEG augmentation** [11] during fine-tuning — the cheapest known
   generalization booster. *(§4)*
9. **Calibration, not just accuracy** [§8 note] — reliability diagrams / ECE before any
   further pseudo-labeling; you already started this with the confidence plot. *(§8)*
10. **Read the arms-race papers** [3][4][35] for the honest framing: there is no permanent
    win, only a maintained detector. Plan for retraining as generators evolve. *(§1, §10)*

---

*Citations web-verified June 2026. arXiv IDs link to abstracts. C2PA and SynthID are
industry standards/tools rather than papers — see their official sites. This map is a
living document; append new generators and detectors as the field moves.*
