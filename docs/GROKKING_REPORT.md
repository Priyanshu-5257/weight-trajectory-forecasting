# Grokking & Algorithmic Phase Transitions: Low-Rank Weight Trajectory Forecasting Benchmark (v2.0)

**A Multi-Seed Empirical Investigation on Modular Arithmetic Transformers**  
*Reproducing Nanda et al. (ICLR 2023) and Power et al. (OpenAI 2022) Trajectory Dynamics*

---

## 🎯 Executive Summary & Core Findings (v2.0)

This benchmark extends the `weight-trajectory-forecasting` framework to **algorithmic phase transitions (grokking)**. In standard tasks (MNIST, California Housing), optimization plateaus represent stationary stochastic oscillations where forecasting acts as weight averaging. In **grokking**, the pre-generalization regime is fundamentally different: training loss is identically zero ($\approx 0.0$), but network parameters undergo a **continuous, directed drift** driven by weight decay toward Fourier circuits.

In this updated **v2.0 benchmark**, we resolved all methodological limitations and bugs identified during review:
1. **TimesFM Horizon Indexing:** Corrected the indexing into the 128-step prediction vector (`mean_predictions[:, horizon - 1]`), ensuring genuine horizon sensitivity.
2. **Pre-Grokking Intervention Anchor:** Relocated the intervention point to **Step 1,500** (pre-grokking floor where non-symmetric test accuracy is $0.67\% \pm 0.66\%$, strictly at the random chance floor $1/97 \approx 1.03\%$).
3. **Commutative Memorization Disentanglement:** Disentangled all test pairs from strictly non-symmetric test pairs (pairs where the reverse $(b, a)$ was not seen during training).
4. **Full-Parameter Cosine Alignment:** Evaluated cosine similarity and relative $L_2$ error in full parameter space $\mathbb{R}^{227,200}$, resolving the 1D latent coordinate artifact.
5. **Mechanistic Controls:** Evaluated an analytical **Weight-Decay-Only control baseline** $\hat{W}^{\text{decay}} = (1 - \eta \lambda)^{h \cdot \Delta t} W_{\text{ref}}$ alongside SWA, EMA, and Fourier circuit saliency ($2$D-DFT power concentration on token embeddings).

### Key Scientific Conclusions (v2.0)

| Dimension | Empirical Finding | Status |
|---|---|---|
| **1. Subspace Collapse** | SVD on 32 checkpoint displacements reveals that **97.47% ± 1.89%** of displacement variance is captured by **Rank 1**, and **99.69% ± 0.47%** by **Rank 2**. The pre-grokking displacement trajectory is effectively a 2D manifold. | **Confirmed 🟢** |
| **2. Radial Shrinkage Dominance** | The dominant low-rank direction is primarily radial $L_2$ shrinkage driven by weight decay ($\lambda=1.0$). The pre-grokking displacement does not yet point in the direction of the rotating trigonometric circuits. | **Mechanistic Insight 🟢** |
| **3. Non-Symmetric Floor** | At Step 1,500, non-symmetric test accuracy is strictly **0.67% ± 0.66%** (chance = 1.03%). The 48.75% accuracy reported on all pairs is entirely due to early commutativity memorization $(b, a)$. | **Falsification Control 🟢** |
| **4. Long-Horizon Bifurcation** | Continued AdamW executes a genuine phase transition: non-symmetric test accuracy rises from **0.67%** at cutoff to **83.14% ± 45.12%** (reaching 99.9%–100% on individual seeds) at $+2500$ steps ($p = 0.0069$), accompanied by a jump in Fourier saliency from **19.5% to 58.0%**. | **True Grokking 🟢** |
| **5. Forecasting Barrier** | Neither Low-Rank Linear (0.78% ± 0.35%), Low-Rank Damped (1.97% ± 2.31%), Low-Rank TimesFM (0.10% ± 0.08%), nor Weight-Decay (0.98% ± 0.17%) can bridge the phase transition. Forecasting pre-grokking trajectories cannot anticipate the orthogonal trigonometric circuit rotations that produce generalization. | **Decisive Negative Boundary 🔴** |
| **6. TimesFM Dynamic Evolution** | With the corrected horizon indexing, TimesFM dynamically varies across horizons ($1.00\%$ at $h=8$, peaking at $1.43\%$ at $h=32$, and decaying to $0.10\%$ at $h=100$), confirming that the indexing bug is resolved. | **Verified 🟢** |

---

## 🔬 Experimental Methodology & Reproduction Fidelity

### 1. Mathematical Task & Architecture
Following **Nanda et al. (ICLR 2023)** and **Power et al. (OpenAI 2022)**:
* **Task:** Modular addition $(a, b) \mapsto (a + b) \pmod{97}$.
* **Dataset Split:** Exactly 50% training pairs ($4,704$ pairs) and 50% test pairs ($4,705$ pairs), strictly disjoint ($Train \cap Test = \emptyset$).
  - *Non-Symmetric Test Subset:* $\approx 2,350$ pairs where neither $(a, b)$ nor $(b, a)$ appeared in training.
* **Architecture:** 1-Layer Transformer decoder:
  - Sequence length: 3 (`[a, b, =]`).
  - $d_{\text{model}} = 128$, $n_{\text{heads}} = 4$ ($d_{\text{head}} = 32$).
  - MLP hidden dimension $d_{\text{mlp}} = 512$ with ReLU activation.
  - Normalization: No LayerNorm.
  - Parameter count: $227,200$ parameters.
* **Optimization:** AdamW ($\eta = 10^{-3}$, $\beta_1 = 0.9, \beta_2 = 0.98$, $\lambda = 1.0$), **full-batch gradient descent** (zero mini-batch stochastic noise).

### 2. Multi-Seed Protocol
* **Seeds:** 5 independent paired runs (`seeds = [42, 43, 44, 45, 46]`).
* **Cutoff Intervention Point:** Step 1,500 (Pre-grokking floor, train acc = 100%, non-sym test acc = 0.67%).
* **Context Window:** 32 checkpoints spaced at $\Delta t = 25$ steps (Steps 725 to 1,500).
* **Evaluation Horizons:** $h \in \{8, 16, 32, 50, 64, 80, 100\}$ corresponding to $+200, +400, +800, +1250, +1600, +2000, +2500$ steps beyond cutoff (training through step 4,000).

---

## 📊 Comprehensive Results Table (5 Seeds, 95% CI)

### Strict Non-Symmetric Generalization Accuracy (%)

| Method / Baseline | Horizon $h=8$ (+200s) | Horizon $h=16$ (+400s) | Horizon $h=32$ (+800s) | Horizon $h=50$ (+1250s) | Horizon $h=64$ (+1600s) | Horizon $h=80$ (+2000s) | Horizon $h=100$ (+2500s) |
|---|---|---|---|---|---|---|---|
| **Frozen Checkpoint (Step 1500)** | 0.67% ± 0.66% | 0.67% ± 0.66% | 0.67% ± 0.66% | 0.67% ± 0.66% | 0.67% ± 0.66% | 0.67% ± 0.66% | 0.67% ± 0.66% |
| **Matched SWA** | 0.25% ± 0.13% | 0.25% ± 0.13% | 0.25% ± 0.13% | 0.25% ± 0.13% | 0.25% ± 0.13% | 0.25% ± 0.13% | 0.25% ± 0.13% |
| **EMA ($\alpha=0.1$)** | 0.44% ± 0.26% | 0.44% ± 0.26% | 0.44% ± 0.26% | 0.44% ± 0.26% | 0.44% ± 0.26% | 0.44% ± 0.26% | 0.44% ± 0.26% |
| **Weight Decay Only Control** | 0.56% ± 0.65% | 0.43% ± 0.45% | 0.46% ± 0.38% | 0.80% ± 0.47% | 0.65% ± 0.41% | 0.98% ± 0.17% | 0.98% ± 0.17% |
| **Continued AdamW (Ground Truth)** | 1.00% ± 0.98% | 1.58% ± 1.67% | 5.17% ± 6.68% | 21.21% ± 26.76% | 46.95% ± 47.68% | 69.23% ± 52.99% | **83.14% ± 45.12%** |
| **Low-Rank Linear ($r=2$)** | 0.97% ± 1.00% | 1.29% ± 1.37% | 1.86% ± 1.95% | 2.26% ± 1.83% | 1.81% ± 1.25% | 1.15% ± 0.56% | 0.78% ± 0.35% |
| **Low-Rank Damped ($r=2$, $\gamma=0.95$)** | 0.97% ± 1.00% | 1.28% ± 1.37% | 1.69% ± 2.06% | 1.81% ± 2.22% | 1.87% ± 2.27% | 1.93% ± 2.28% | 1.97% ± 2.31% |
| **Low-Rank TimesFM ($r=2$, Fixed)** | 1.00% ± 1.01% | 1.23% ± 1.22% | 1.43% ± 1.29% | 1.26% ± 1.13% | 0.86% ± 0.84% | 0.40% ± 0.47% | 0.10% ± 0.08% |

### Paired Differences vs Frozen Checkpoint ($\Delta = \text{Method} - \text{Frozen}$)

| Method / Horizon | $\Delta$ at $h=8$ (+200s) | $\Delta$ at $h=32$ (+800s) | $\Delta$ at $h=64$ (+1600s) | $\Delta$ at $h=100$ (+2500s) | Paired $t$-test ($h=100$) |
|---|---|---|---|---|---|
| **Continued AdamW** | +0.33% ± 0.41% | +4.50% ± 6.12% | +46.28% ± 47.27% | **+82.47% ± 44.77%** | $t = 5.11$, $p = 0.0069$ 🟢 |
| **Weight Decay Only** | -0.11% ± 0.10% | -0.21% ± 0.53% | -0.02% ± 0.97% | +0.31% ± 0.66% | $t = 1.31$, $p = 0.2592$ |
| **Low-Rank Linear ($r=2$)** | +0.30% ± 0.43% | +1.19% ± 1.36% | +1.14% ± 0.83% | +0.11% ± 0.35% | $t = 0.81$, $p = 0.4619$ |
| **Low-Rank Damped ($r=2$)** | +0.30% ± 0.43% | +1.02% ± 1.46% | +1.20% ± 1.66% | +1.30% ± 1.71% | $t = 2.12$, $p = 0.1017$ |
| **Low-Rank TimesFM ($r=2$)** | +0.33% ± 0.47% | +0.76% ± 0.82% | +0.19% ± 1.08% | -0.57% ± 0.66% | $t = -2.34$, $p = 0.0793$ |

---

## 📈 Visualizations

All publication figures are stored in `figures/`:
1. `figures/grokking_accuracy_vs_horizon.png`: Non-symmetric test accuracy trajectory comparing Continued AdamW, Linear, Damped, TimesFM, and Weight Decay, alongside the commutative memorization artifact panel.
2. `figures/grokking_subspace_explained_variance.png`: SVD explained variance (Rank 1: 97.47%, Rank 2: 99.69%) and radial alignment projection.
3. `figures/grokking_cosine_alignment.png`: Full-parameter cosine similarity ($\in \mathbb{R}^{227,200}$) comparing Linear, Damped, and TimesFM.
4. `figures/grokking_fourier_circuit_saliency.png`: Fourier modular circuit formation (Top-5 DFT power ratio) across optimization.

---

## 🧠 Mechanistic Insights: Why Weight Forecasting Fails in Grokking

1. **Radial Shrinkage vs. Circuit Formation**:
   During the pre-grokking regime (steps 700 to 1,500), the network is in an unregularized memorization state. The primary dynamic occurring in weight space is **radial $L_2$ contraction** ($\hat{W} \approx (1 - \eta \lambda)^t W$) driven by weight decay. The principal components from SVD align heavily with this radial contraction vector ($97.5\%$ variance in Rank 1).
2. **Orthogonality of the Generalization Circuit**:
   As demonstrated by Nanda et al. (2023), generalization in modular arithmetic requires constructing clean **Fourier key-value rotations** $\cos(\frac{2\pi k x}{p}), \sin(\frac{2\pi k x}{p})$ in the token embeddings and attention heads. These rotations are **orthogonal** to the pre-grokking displacement manifold. Because the pre-grokking history contains negligible Fourier circuit energy (saliency is flat at $\sim 19\%$), linear extrapolation, damped extrapolation, and TimesFM cannot anticipate the trigonometric rotation.
3. **The TimesFM Horizon Resolution**:
   With the corrected horizon indexing (`mean_predictions[:, horizon - 1]`), TimesFM outputs vary systematically across horizons. However, because TimesFM is trained on general numerical time series, it expects smooth or mean-reverting continuations, not the sudden phase change of grokking.

---

## 📂 Artifact Links
- Raw 5-Seed Benchmark Metrics: [results/grokking_benchmark_results.json](file:///home/prinshu/exp/weightSpace/anomlydet/results/grokking_benchmark_results.json)
- Statistical Summary (95% CI): [results/grokking_benchmark_summary.json](file:///home/prinshu/exp/weightSpace/anomlydet/results/grokking_benchmark_summary.json)
- Benchmark Runner: [benchmarks/benchmark_grokking_trajectory.py](file:///home/prinshu/exp/weightSpace/anomlydet/benchmarks/benchmark_grokking_trajectory.py)
- Low-Rank Engine: [src/low_rank.py](file:///home/prinshu/exp/weightSpace/anomlydet/src/low_rank.py)
- Plot Generation Script: [scripts/generate_grokking_figures.py](file:///home/prinshu/exp/weightSpace/anomlydet/scripts/generate_grokking_figures.py)
