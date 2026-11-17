# Low-Rank Weight Trajectory Forecasting: Feasibility Benchmark Report (v1.2)

**Status:** Completed & Pre-Registered  
**Date:** November 17, 2026  
**Final Recommendation:** **NO-GO** for Scaling to LoRA Adapters / Large Models  
**Repository:** [github.com/Priyanshu-5257/weight-trajectory-forecasting](https://github.com/Priyanshu-5257/weight-trajectory-forecasting)  
**Artifacts:** [`feasibility_benchmark_results.json`](../results/feasibility_benchmark_results.json), [`feasibility_benchmark_summary.json`](../results/feasibility_benchmark_summary.json), [`feasibility_benchmark_per_seed.csv`](../results/feasibility_benchmark_per_seed.csv)

---

## Executive Summary & Core Verdict

This benchmark evaluated whether zero-shot foundation time-series forecasting (**Google TimesFM 2.0 500M**) over **low-rank weight trajectories** ($D \approx U_r C_r$) can achieve comparable or superior model quality to standard baselines (**SWA, EMA, Continued Training**) at lower end-to-end computational cost.

The evaluation enforced strict experimental hygiene:
- Disjoint 70/15/15 splits on California Housing (StandardScaler fitted strictly on training data).
- 50k/10k/10k splits on MNIST with automated leakage unit tests passing.
- **Hyperparameter selection strictly isolated to validation data** (ranks $r \in \{1, 4, 8, 16\}$, horizons $h \in \{8, 16, 32\}$ selected via $\arg\min \text{Val Loss}$, evaluated once on the untouched test split).
- 10 paired random seeds (`42..51`) with paired 95% confidence intervals and multi-pass timing benchmarks.

```
========================================================================================
 FINAL DECISION: NO-GO
========================================================================================
 1. Quality Failure: TimesFM does not beat online SWA on MNIST (0.1824 vs 0.1810) and
    fails to beat Continued Adam on California Housing (0.2993 vs 0.2801).
 2. Compute Inefficiency: SWA executes in < 0.1 ms with 0 parameters and 0 MB memory.
    TimesFM requires 37-68 ms latency and ~2.0 GB of GPU VRAM for the foundation model.
 3. Mechanism Disproven: Shuffled-history TimesFM matches SWA identically (0.1806 vs 
    0.1810) with cosine similarity 0.9937 to the SWA displacement vector, while showing
    negative correlation (-0.46) to actual future gradient steps.
 4. Recommendation: TimesFM acts as an expensive non-linear shrinkage smoother around
    the trajectory centroid. It does NOT learn forward optimization dynamics in
    plateau regimes. Do not scale to LoRA adapters or LLMs.
========================================================================================
```

---

## 1. Quality: Does Forecasting Outperform Baselines?

### 1.1 Benchmark Results (10 Paired Seeds, 95% Confidence Intervals)

All forecasting hyperparameters were chosen strictly via validation loss per seed, then evaluated on untouched test data.

| Method / Baseline | California Housing Test MSE (Lower is Better) | MNIST Test Loss (Lower is Better) | MNIST Test Accuracy (%) |
| :--- | :---: | :---: | :---: |
| **Frozen Checkpoint (Cutoff Step $T$)** | $0.2916 \pm 0.0078$ | $0.2038 \pm 0.0097$ | $94.01\% \pm 0.28\%$ |
| **Continued Training ($h=16$ steps)** | **$0.2847 \pm 0.0063$** 🟢 | $0.2015 \pm 0.0065$ | $94.08\% \pm 0.24\%$ |
| **Continued Training ($h=32$ steps)** | **$0.2801 \pm 0.0041$** 🏆 | $0.2032 \pm 0.0119$ | $94.13\% \pm 0.31\%$ |
| **Matched EMA ($\alpha = 0.05$)** | $0.2934 \pm 0.0092$ | $0.1832 \pm 0.0041$ | $94.67\% \pm 0.18\%$ |
| **Matched SWA (Online Centroid)** | $0.2981 \pm 0.0085$ | **$0.1810 \pm 0.0042$** 🏆 | **$94.75\% \pm 0.16\%$** 🏆 |
| **Low-Rank Linear (Val-Selected)** | $0.2932 \pm 0.0084$ | $0.2223 \pm 0.0165$ 💥 | $93.45\% \pm 0.42\%$ |
| **Low-Rank Damped (Val-Selected)** | $0.2924 \pm 0.0081$ | $0.2149 \pm 0.0137$ | $93.71\% \pm 0.35\%$ |
| **Low-Rank TimesFM (Val-Selected)** | $0.2993 \pm 0.0088$ | $0.1824 \pm 0.0044$ 🟢 | $94.70\% \pm 0.16\%$ |
| **TimesFM Shuffled-History Control** | $0.2961 \pm 0.0066$ | **$0.1806 \pm 0.0043$** 🏆 | **$94.76\% \pm 0.15\%$** |

### 1.2 Quality Findings
1. **On MNIST Image Classification:**
   - Low-Rank TimesFM (`0.1824 ± 0.0044`) significantly improves over the frozen checkpoint (`0.2038 ± 0.0097`, paired $p < 0.0001$) and Continued SGD (`0.2015 ± 0.0065`).
   - However, **Matched SWA achieves `0.1810 ± 0.0042`**, outperforming TimesFM without running any neural network forward passes.
   - Low-Rank Linear extrapolation severely degrades performance (`0.2223`), proving that raw linear projection cannot stabilize stochastic updates.
2. **On California Housing Tabular Regression:**
   - Continued Adam for 32 steps achieves **`0.2801 ± 0.0041`**, substantially beating TimesFM (`0.2993 ± 0.0088`) and Frozen (`0.2916 ± 0.0078`).
   - Forecasting provides no benefit in regression with adaptive learning rate optimizers.

---

## 2. Efficiency: Is Forecasting Cheaper at Matched Quality?

The previous benchmark showed that low-rank projection achieves an $880\times$ speedup over per-weight TimesFM. However, feasibility requires comparing **end-to-end forecasting cost against standard optimization at matched validation quality**.

### 2.1 Latency Breakdown (Warmed Benchmarks, Median & p95)

| Pipeline Component | Measurement Details | Median Latency | p95 Latency |
| :--- | :--- | :---: | :---: |
| **Checkpoint Storage Overhead** | 64 Checkpoints in memory ($P=3{,}322$) | $830.5\text{ KB total}$ | $< 0.001\text{ ms}$ |
| **SVD Thin Decomposition** | $D \in \mathbb{R}^{3322 \times 64} \to U_r, \Sigma_r, V_r^\top$ | **$7.30\text{ ms}$** | $7.67\text{ ms}$ |
| **Weight Reconstruction** | $W_T + U_r \hat{c}$ (for $r \le 16$) | **$0.016\text{ ms}$** | $0.017\text{ ms}$ |
| **TimesFM Inference ($r=1, h=8$)** | Batch size 1, horizon 8 | **$30.11\text{ ms}$** | $30.46\text{ ms}$ |
| **TimesFM Inference ($r=4, h=16$)** | Batch size 4, horizon 16 | **$60.67\text{ ms}$** | $61.39\text{ ms}$ |
| **TimesFM Inference ($r=16, h=32$)** | Batch size 16, horizon 32 | **$234.47\text{ ms}$** | $235.05\text{ ms}$ |
| **Total TimesFM Pipeline ($r=4, h=16$)** | $\text{SVD} + \text{Inference} + \text{Recon}$ | **$67.99\text{ ms}$** | $69.08\text{ ms}$ |
| **Continued SGD Training ($h=16$ steps)** | Forward + Loss + Backward + Step | **$13.04\text{ ms}$** | $14.12\text{ ms}$ |
| **Continued Adam Training ($h=32$ steps)** | Forward + Loss + Backward + Step | **$31.58\text{ ms}$** | $33.20\text{ ms}$ |
| **Matched SWA (Online Average)** | Running weight accumulator | **$< 0.05\text{ ms}$** | $< 0.05\text{ ms}$ |

### 2.2 Memory Footprint
- **Model Weights in VRAM:** TimesFM 2.0 500M requires **$1{,}966.9\text{ MiB}$ (~1.92 GB)** of GPU memory.
- **Activation Overhead:** Low-rank coordinate batches ($r \le 16$) require $< 1\text{ MB}$ of dynamic activation memory.
- **Baseline Training:** Training the MLP takes only **$85\text{ MiB}$**.

### 2.3 Matched Quality Speedup Evaluation

$$\text{Speedup} = \frac{T_{\text{continued training}}}{T_{\text{entire forecasting pipeline}}}$$

1. **Against SWA:**
   - Matched SWA reaches `0.1810` test loss in **$< 0.05\text{ ms}$**.
   - Low-Rank TimesFM reaches `0.1824` test loss in **$37.4\text{ ms}$ to $68.0\text{ ms}$**.
   - **Relative Speedup:** **$\approx 0.001\times$** (TimesFM is $\approx 750\times$ slower than SWA and consumes 2 GB more VRAM).
2. **Against Continued Adam (California Housing):**
   - Continued Adam reaches `0.2801` MSE in **$31.58\text{ ms}$**.
   - Low-Rank TimesFM reaches `0.2993` MSE in **$68.0\text{ ms}$**.
   - TimesFM is **slower than continued training** ($68\text{ ms}$ vs $32\text{ ms}$) and achieves worse loss.

---

## 3. Mechanism: What Does TimesFM Actually Do?

To test whether TimesFM learns directional extrapolation or merely acts as a smoother, we evaluated three mechanistic probes:

### 3.1 Ordered vs. Shuffled Latent Forecasting
- On MNIST, the **Time-Shuffled Control achieves `0.1806 ± 0.0043`**, which is slightly **better** than the ordered forecast (`0.1824 ± 0.0044`) and matches SWA (`0.1810`).
- On California Housing, Shuffled TimesFM achieves `0.2961 ± 0.0066` vs Ordered `0.2993 ± 0.0088`.
- **Verdict:** Chronological ordering carries no predictive forward value in the optimization plateau. Destroying the time order improves or maintains the forecast.

### 3.2 Alignment to SWA vs. Alignment to Gradient Trajectory

$$\text{CosSim}(\hat{W} - W_T, W_{\text{SWA}} - W_T) \quad \text{vs.} \quad \text{CosSim}(\hat{c}, c_{\text{actual continuation}})$$

| Dataset | Cosine Sim to SWA (Ordered) | Cosine Sim to SWA (Shuffled Control) | Cosine Sim to Actual Continuation | Relative Error to Continuation |
| :--- | :---: | :---: | :---: | :---: |
| **California Housing** | $0.4141 \pm 0.6431$ | **$0.9968 \pm 0.0019$** | **$-0.4622 \pm 0.5462$** | $6.09 \pm 2.47$ |
| **MNIST Classification** | $0.9354 \pm 0.1128$ | **$0.9937 \pm 0.0026$** | **$-0.4629 \pm 0.5948$** | $39.34 \pm 26.31$ |

**Physical Interpretation:**
1. The displacement vector produced by Shuffled TimesFM is **collinear with the SWA vector ($\cos \theta = 0.9937 - 0.9968$)**. When time ordering is permuted, the transformer self-attention heads perform uniform temporal pooling, recovering the empirical coordinate centroid:
   $$\hat{c} \approx \frac{1}{T}\sum_{t=1}^T c_t \implies \hat{W} \approx W_{\text{SWA}}$$
2. The directional alignment to the true continuation trajectory is **negative ($\cos \theta \approx -0.46$)**. TimesFM is not anticipating future gradient steps; it is pulling weights backward into the center of the local Brownian basin.

---

## 4. Scalability: Real Costs of Trajectory Forecasting

| Operation | Cost Scaling | Practical Implication for LoRA / LLMs |
| :--- | :--- | :--- |
| **Checkpoint Storage** | $\mathcal{O}(P \times T)$ | For a 7B LLM with 50M LoRA parameters and $T=64$, storing checkpoints in RAM takes **$12.8\text{ GB}$**. |
| **SVD Thin Decomposition** | $\mathcal{O}(P \times T^2)$ | Computing SVD of $50\text{M} \times 64$ takes **$\sim 1.5 - 2.5\text{ seconds}$** on GPU. |
| **TimesFM Foundation Model VRAM** | Static $2.0\text{ GB}$ | Allocating 2 GB of VRAM for TimesFM competes directly with KV-cache and training batch sizes on edge/single GPUs. |
| **Online SWA Alternative** | $\mathcal{O}(P)$ running sum | Takes **$0\text{ ms}$ latency**, requires no foundation model, and uses only 1 extra parameter buffer. |

---

## 5. Final Decision & Recommendations

### Pre-Registered Criteria Assessment:
1. *Achieves comparable or better validation-selected performance than strongest cheap baseline?* **FAIL.** SWA achieves equal or better test loss on both benchmarks.
2. *Reduces end-to-end compute compared with continued optimization at matched quality?* **FAIL.** SWA is $750\times$ faster; Continued Adam is $2\times$ faster than TimesFM at higher accuracy.
3. *Generalizes across independent seeds without relying on isolated favorable runs?* **PASS on stability, FAIL on utility.** Consistent across all 10 seeds, but consistently redundant to SWA.

### Final Verdict: **NO-GO**
We conclude this research line with a clean, well-characterized negative result:
- **Zero-shot time-series foundation models cannot accelerate neural network optimization in plateau regimes.**
- In stationary quadratic basins, weight dynamics constitute Brownian motion around a local minimum. Foundation models do not discover gradient-free shortcuts along the loss manifold; rather, their attention layers implement an implicit shrinkage operator that approximates Stochastic Weight Averaging at thousands of times higher computational and memory cost.
- **Do not scale this method to LoRA adapters or large language models.** For optimization acceleration and checkpoint stabilization, use online SWA or EMA.
