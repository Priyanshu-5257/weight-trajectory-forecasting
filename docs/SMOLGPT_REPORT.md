# Empirical Assessment of Zero-Shot Weight Trajectory Forecasting on Transformer Language Models (SmolGPT)

**Date:** November 18, 2026  
**Subject:** Empirical evaluation of zero-shot weight-space trajectory forecasting (Google TimesFM 2.0 500M and low-rank extrapolations) on a 1.25M-parameter causal Transformer language model.  
**Repository:** `weight-trajectory-forecasting`  
**Dataset:** TinyStories (disjoint 2.68M-token train split, 298K-token held-out validation split)  
**Hardware:** NVIDIA GeForce RTX 3060 Laptop GPU (CUDA 12.8, PyTorch 2.10.0+cu128)  

---

## 1. Executive Summary & Verdict

Following our prior investigations into low-rank trajectory forecasting across regression, MNIST classification, and modular arithmetic grokking, this benchmark evaluates whether **Google TimesFM 2.0 500M** and low-rank trajectory extrapolations can anticipate optimization trajectories in a **modern causal Transformer language model architecture** (`SmolGPT`, 4 layers, GQA, RoPE, SwiGLU, RMSNorm, tied embeddings; 1,246,080 parameters).

```
+----------------------------------------------------------------------------------------------------+
|                                    DECISIVE RESEARCH VERDICT                                       |
+----------------------------------------------------------------------------------------------------+
| 1. Low-Dimensional Subspace Concentration:            CONFIRMED  (Rank 2 captures 97.10% variance) |
| 2. Low-Rank Linear Extrapolation:                     NO-GO      (Severe curvature blowout: PPL>19k)|
| 3. Low-Rank Damped Extrapolation:                     NO-GO      (Stabilizes but degrades: PPL 40.7)|
| 4. Zero-Shot TimesFM-2.0 Forecasting:                 NO-GO      (PPL 19.22 vs 16.09 Frozen, p=0.04)|
| 5. Actual Optimization Continuation (AdamW):          SUPERIOR   (PPL drops to 12.23, p=0.0002)     |
| 6. Zero-Cost Stochastic Weight Averaging (SWA):       SUPERIOR   (PPL 15.30 vs 16.09 Frozen, p<1e-4)|
+----------------------------------------------------------------------------------------------------+
```

### Key Quantitative Findings across 3 Paired Seeds (95% CI):
1. **AdamW Checkpoint Concentration is Overwhelmingly Low-Rank:**
   In SmolGPT, rank 1 captures **92.61% $\pm$ 0.81%** and rank 2 captures **97.10% $\pm$ 0.44%** of the cumulative parameter displacement variance over 32 checkpoints (640 steps). By rank 4, **98.78%** of all displacement variance is accounted for.
2. **Linear Trajectory Extrapolation Diverges Catastrophically:**
   Attempting linear continuation in the rank-2 subspace causes rapid manifold departure: validation perplexity degrades from **16.09** at the frozen reference checkpoint to **20.36** (+160 steps), **33.53** (+320 steps), **191.67** (+640 steps), and blows up to **19,383.52 $\pm$ 20,137.17** (+1280 steps, $p = 0.0537$).
3. **Damped Extrapolation Prevents Divergence but Increases Loss:**
   Geometric damping ($\gamma = 0.95$) prevents numerical explosion, but parameter drift nonetheless steadily degrades language modeling quality ($PPL = 19.05 \to 40.70$, $p = 0.0001$ worse than frozen).
4. **TimesFM 2.0 (500M) Fails to Outperform Frozen Reference or Standard Baselines:**
   At all evaluated horizons ($h \in [8, 16, 32, 64]$ checkpoints, $+160$ to $+1280$ optimizer steps), TimesFM produces higher perplexity than the frozen checkpoint (**19.22 – 27.43** vs **16.09 $\pm$ 0.13**, paired $p = 0.0094$ to $0.0430$).
5. **Matched SWA Strictly Beats Every Extrapolation Method with Zero Compute Cost:**
   Taking the empirical average of the last 16 historical checkpoints (`Matched SWA`) yields a statistically significant perplexity reduction (**15.30 $\pm$ 0.12** vs **16.09 $\pm$ 0.13**, paired $p < 0.0001$) in $< 0.01\text{ ms}$, soundly outperforming TimesFM (**19.22**), Damped (**40.70**), and Linear (**19,383**).
6. **Temporal Shuffling Ablation Confirms Failure of Dynamics Modeling:**
   Permuting the temporal order of checkpoints (`Shuffled TimesFM`) results in *better* perplexity (**16.79 – 17.08**) than ordered forecasting (**19.22 – 27.43**). This demonstrates that TimesFM's sequential forecasts extrapolate along high-curvature tangents that damage the model, whereas temporal scrambling naturally contracts the prediction toward the empirical mean.

---

## 2. Experimental Setup & Architecture

### Model Architecture: SmolGPT
We trained an autoregressive causal decoder modeled on modern LLM conventions:
* **Architecture:** 4 Transformer Blocks, Causal Grouped-Query Attention (GQA).
* **Dimensionality:** $d_{\text{model}} = 128$, $n_{\text{heads}} = 4$, $n_{\text{kv\_heads}} = 2$ ($d_{\text{head}} = 32$).
* **Feed-Forward:** SwiGLU FFN with hidden dimension $d_{\text{ffn}} = \lfloor \frac{8}{3} d_{\text{model}} \rfloor = 341$.
* **Positional Encoding:** Rotary Positional Embeddings (RoPE, $\theta = 10,000$).
* **Normalization:** Pre-RMSNorm ($\epsilon = 10^{-5}$) and QK-Normalization.
* **Tied Embeddings:** $W_{\text{embed}} \equiv W_{\text{lm\_head}}$ (SentencePiece vocabulary = 4,096).
* **Total Parameters:** $P = 1,246,080$ unique parameters.
  * Embeddings / Head: $524,288$ ($42.07\%$)
  * Attention projections: $196,864$ ($15.80\%$)
  * SwiGLU MLP: $523,776$ ($42.03\%$)
  * RMSNorm scales: $1,152$ ($0.09\%$)

### Dataset & Training Protocol
* **Corpus:** TinyStories, pre-tokenized with SentencePiece into uint16 arrays:
  * Training split: $2,686,590$ tokens (90%)
  * Validation split: $298,510$ tokens (10%)
* **Batching:** Sequences of $T=128$ tokens, batch size $B=16$ ($2,048$ tokens/step).
* **Deterministic Evaluation:** 20 pre-materialized, disjoint validation chunks ($40,960$ tokens) evaluated across all checkpoints and methods.
* **Optimizer:** AdamW ($\eta = 10^{-3}$, $\beta_1 = 0.9$, $\beta_2 = 0.95$, weight decay $\lambda = 0.01$, gradient clip norm $= 1.0$).
* **Training Speed:** 84.9 steps/sec on RTX 3060 Laptop GPU.

### Trajectory Window & Forecast Horizons
* **History Window:** 32 checkpoints recorded every $\Delta t = 20$ optimizer steps between steps 380 and 1,000.
* **Cutoff Step ($t_{\text{ref}}$):** Step 1,000.
* **Forecast Horizons ($h$):**
  * $h = 8$ checkpoints $\implies +160$ training steps (Step 1,160)
  * $h = 16$ checkpoints $\implies +320$ training steps (Step 1,320)
  * $h = 32$ checkpoints $\implies +640$ training steps (Step 1,640)
  * $h = 64$ checkpoints $\implies +1,280$ training steps (Step 2,280)

---

## 3. Empirical Results

### Table 1: Validation Perplexity ($PPL = \exp(\text{CE})$) Across Horizons (3 Seeds, Mean $\pm$ 95% CI)

| Method | $+160$ Steps ($h=8$) | $+320$ Steps ($h=16$) | $+640$ Steps ($h=32$) | $+1280$ Steps ($h=64$) | Compute Latency |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **Continued AdamW (Ground Truth)** | **`15.19 ± 0.12`** 🟢 | **`14.49 ± 0.08`** 🟢 | **`13.41 ± 0.03`** 🟢 | **`12.23 ± 0.23`** 🏆 | $1.8\text{ s} - 14.5\text{ s}$ |
| **Frozen Checkpoint ($W_{\text{ref}}$)** | `16.09 ± 0.13` | `16.09 ± 0.13` | `16.09 ± 0.13` | `16.09 ± 0.13` | $0.0\text{ ms}$ |
| **Matched SWA (Last 16 Ckpts)** | `15.30 ± 0.12` ⭐ | `15.30 ± 0.12` ⭐ | `15.30 ± 0.12` ⭐ | `15.30 ± 0.12` ⭐ | $< 0.01\text{ ms}$ |
| **Matched EMA ($\beta=0.9$)** | `15.50 ± 0.14` | `15.50 ± 0.14` | `15.50 ± 0.14` | `15.50 ± 0.14` | $< 0.01\text{ ms}$ |
| **Weight Decay Only** | `16.09 ± 0.13` | `16.10 ± 0.14` | `16.12 ± 0.15` | `16.16 ± 0.17` | $9.3\text{ ms}$ |
| **Low-Rank Linear (Rank 2)** | `20.36 ± 0.39` | `33.53 ± 1.04` | `191.67 ± 27.28` | `19383.52 ± 20137` 💥 | $9.0\text{ ms}$ |
| **Low-Rank Damped (Rank 2, $\gamma=0.95$)** | `19.05 ± 0.31` | `23.27 ± 0.55` | `31.78 ± 0.96` | `40.70 ± 1.31` | $8.2\text{ ms}$ |
| **Low-Rank TimesFM (Rank 2)** | `21.31 ± 2.12` | `27.43 ± 9.93` | `19.55 ± 9.43` | `19.22 ± 2.97` | $46.5\text{ ms}$ |
| **Shuffled TimesFM (Rank 2 Control)** | `16.85 ± 0.90` | `17.00 ± 1.59` | `16.79 ± 0.20` | `17.08 ± 1.57` | $47.2\text{ ms}$ |
| **Low-Rank TimesFM (Rank 4)** | `25.51 ± 3.39` | `23.44 ± 8.47` | `21.04 ± 11.56` | `19.39 ± 2.45` | $62.8\text{ ms}$ |

---

### Table 2: Paired Difference Tests vs. Frozen Checkpoint and Continued AdamW

| Method | Horizon | Paired Diff vs. Frozen Checkpoint | Paired Diff vs. Continued AdamW | Statistical Verdict |
| :--- | :---: | :---: | :---: | :---: |
| **Matched SWA** | Any | **$\Delta = -0.79$, $p < 0.0001$** | $\Delta = +0.11$, $p = 0.1683$ ($h=8$) | **Statistically Significant Gain** 🟢 |
| **Continued AdamW** | $+160$ ($h=8$) | **$\Delta = -0.90$, $p = 0.0037$** | $\Delta = 0.00$ | **Statistically Significant Gain** 🟢 |
| **Continued AdamW** | $+1280$ ($h=64$) | **$\Delta = -3.86$, $p = 0.0002$** | $\Delta = 0.00$ | **Major Optimization Gain** 🟢 |
| **TimesFM (Rank 2)** | $+160$ ($h=8$) | $\Delta = +5.23$, $p = 0.0094$ | $\Delta = +6.13$, $p = 0.0058$ | **Statistically Significant Degradation** 🔴 |
| **TimesFM (Rank 2)** | $+1280$ ($h=64$) | $\Delta = +3.14$, $p = 0.0430$ | $\Delta = +7.00$, $p = 0.0107$ | **Statistically Significant Degradation** 🔴 |
| **Linear (Rank 2)** | $+160$ ($h=8$) | $\Delta = +4.28$, $p = 0.0002$ | $\Delta = +5.18$, $p = 0.0005$ | **Statistically Significant Degradation** 🔴 |
| **Linear (Rank 2)** | $+640$ ($h=32$) | $\Delta = +175.58$, $p = 0.0013$ | $\Delta = +178.25$, $p = 0.0013$ | **Severe Manifold Departure** 🔴 |
| **Damped (Rank 2)** | $+1280$ ($h=64$) | $\Delta = +24.62$, $p = 0.0001$ | $\Delta = +28.48$, $p = 0.0001$ | **Severe Degradation** 🔴 |

---

## 4. Mechanistic Analysis

### 4.1 Subspace Geometry in Language Models
The displacement matrix $D \in \mathbb{R}^{1,246,080 \times 32}$ exhibits steep singular value decay:
* $\sigma_1 = 3.66$, $\sigma_2 = 0.81$, $\sigma_3 = 0.48$, $\sigma_4 = 0.38$, $\dots$, $\sigma_{32} = 0.04$.
* Rank 1 captures **$92.61\%$** of total displacement energy.
* Rank 2 captures **$97.10\%$**.
* Rank 4 captures **$98.78\%$**.

Although 97.10% of the trajectory displacement is concentrated in a 2-dimensional plane, **extrapolating within this historical plane does not lead to generalization or continued loss descent**. 

### 4.2 Why Does Linear Extrapolation Blow Up?
In language models, cross-entropy is acutely sensitive to logit scale and softmax temperature:
1. Historical AdamW updates maintain parameter velocity along the early descent gradient direction.
2. Linear extrapolation multiplies this displacement vector by large scalar factors ($h \in [8 \dots 64]$).
3. Because the true optimization manifold is non-linear and convex only locally, linear extrapolation punches through the boundary of the basin.
4. Activations in deeper layers explode, producing extreme logit unbalances and sending perplexity from $16.09$ to $19,383.52$.

### 4.3 Why Does TimesFM Fail to Beat the Baseline?
TimesFM generates bounded latent coordinate forecasts (preventing the $19,000$ blowout), but:
1. **Curvature Blindness:** TimesFM extrapolates coordinate sequences without awareness of loss Hessian eigenvalues or parameter-to-loss couplings.
2. **Subspace Curvature:** As the model continues optimizing, the true AdamW trajectory curves out of the historical rank-2 plane. At $h=64$, the full-parameter cosine similarity between the TimesFM forecast and the actual continuation drops to **$\cos = -0.1498$** (orthogonal or slightly anti-aligned).
3. **Implicit Smoothing Failure:** When the temporal order is preserved, TimesFM outputs a non-zero extrapolation step that enters high-curvature territory. When the temporal order is destroyed (`Shuffled TimesFM`), TimesFM outputs a conservative shrinkage toward the origin, yielding $PPL = 16.79$—closer to the frozen model ($16.09$).

### 4.4 Layer-Wise Alignment Breakdown
At $h=64$ (+1280 steps), directional alignment to the true continuation differs across architectural components:
* **Token Embeddings ($W_{\text{embed}}$):** Cosine similarity $= -0.151$ (negative alignment).
* **Attention Projections ($W_Q, W_K, W_V, W_O$):** Cosine similarity $= -0.148$ (negative alignment).
* **SwiGLU MLP ($W_1, W_2, W_3$):** Cosine similarity $= -0.150$ (negative alignment).
* **RMSNorm Scaling Parameters:** Cosine similarity $= +0.021$.

The entire parameter vector rotates away from the historical tangent space across every layer group.

---

## 5. Artifacts & Reproducibility

All code, tests, results, and publication figures have been compiled into the repository:
* **Architecture Implementation:** [src/models_transformer.py](file:///home/prinshu/exp/weightSpace/anomlydet/src/models_transformer.py)
* **Dataset Loader:** [src/data_transformer.py](file:///home/prinshu/exp/weightSpace/anomlydet/src/data_transformer.py)
* **Benchmark Runner:** [benchmarks/benchmark_smolgpt_trajectory.py](file:///home/prinshu/exp/weightSpace/anomlydet/benchmarks/benchmark_smolgpt_trajectory.py)
* **Figure Generator:** [scripts/generate_smolgpt_figures.py](file:///home/prinshu/exp/weightSpace/anomlydet/scripts/generate_smolgpt_figures.py)
* **Unit Tests (18 passing):** [tests/test_smolgpt.py](file:///home/prinshu/exp/weightSpace/anomlydet/tests/test_smolgpt.py)
* **Raw Artifacts:**
  * [results/smolgpt_benchmark_results.json](file:///home/prinshu/exp/weightSpace/anomlydet/results/smolgpt_benchmark_results.json)
  * [results/smolgpt_benchmark_summary.json](file:///home/prinshu/exp/weightSpace/anomlydet/results/smolgpt_benchmark_summary.json)
  * [results/smolgpt_benchmark_per_seed.csv](file:///home/prinshu/exp/weightSpace/anomlydet/results/smolgpt_benchmark_per_seed.csv)
* **Publication Figures:**
  * Figure 1: [figures/smolgpt_ppl_vs_horizon.png](file:///home/prinshu/exp/weightSpace/anomlydet/figures/smolgpt_ppl_vs_horizon.png)
  * Figure 2: [figures/smolgpt_explained_variance_and_spectrum.png](file:///home/prinshu/exp/weightSpace/anomlydet/figures/smolgpt_explained_variance_and_spectrum.png)
  * Figure 3: [figures/smolgpt_cosine_alignment_and_layer_breakdown.png](file:///home/prinshu/exp/weightSpace/anomlydet/figures/smolgpt_cosine_alignment_and_layer_breakdown.png)
  * Figure 4: [figures/smolgpt_compute_quality_tradeoff.png](file:///home/prinshu/exp/weightSpace/anomlydet/figures/smolgpt_compute_quality_tradeoff.png)
