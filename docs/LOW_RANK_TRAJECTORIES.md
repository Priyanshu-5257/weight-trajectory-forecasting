# Low-Rank Weight Trajectory Forecasting: Intrinsic Subspace Dynamics, Linear Divergence, and 880x Compute Acceleration

## 1. Executive Summary & Hypotheses

Per-weight time-series forecasting using large foundation models (such as Google TimesFM 2.0 500M) suffers from a severe computational bottleneck: forecasting tens or hundreds of thousands of parameter trajectories independently incurs seconds to minutes of GPU inference latency ($45.90$ s for $10{,}000$ weights), dwarfing the cost of actual gradient descent steps ($0.65$ s).

This investigation evaluates **Low-Rank Weight Trajectory Forecasting**:
1. **Low-Dimensional Manifold Hypothesis:** Do neural network optimization updates during plateau regimes lie in an intrinsic low-dimensional subspace ($r \ll P$)?
2. **Latent Extrapolation vs. Foundation Models:** Can simple linear or damped extrapolation of latent coefficients match zero-shot foundation model forecasting?
3. **Implicit Regularization Mechanism:** Why does TimesFM succeed where linear extrapolation fails, and does chronological ordering matter in latent space?
4. **End-to-End Compute Acceleration:** Can low-rank decomposition eliminate the TimesFM inference bottleneck and deliver viable gradient-free optimization acceleration?

---

## 2. Mathematical Framework

### 2.1 Displacement Matrix & Truncated SVD
Let $\{W_1, W_2, \dots, W_T\} \subset \mathbb{R}^P$ be a sequence of $T$ model checkpoints saved at uniform optimizer step intervals. We define the displacement matrix relative to a reference checkpoint $W_{\text{ref}} = W_T$ (the latest frozen checkpoint):

$$\Delta W_t = W_t - W_{\text{ref}} \in \mathbb{R}^P, \quad t \in \{1, \dots, T\}$$

Stacking these column vectors yields the trajectory displacement matrix:

$$D = [\Delta W_1, \Delta W_2, \dots, \Delta W_T] \in \mathbb{R}^{P \times T}$$

Applying thin Singular Value Decomposition (SVD):

$$D = U \Sigma V^\top = \sum_{i=1}^{\min(P, T)} \sigma_i u_i v_i^\top$$

For a chosen subspace rank $r \in \{1, 4, 8, 16\}$:
- $U_r = [u_1, \dots, u_r] \in \mathbb{R}^{P \times r}$: Orthogonal spatial basis vectors in parameter space.
- $C_r = \Sigma_r V_r^\top = U_r^\top D \in \mathbb{R}^{r \times T}$: Low-dimensional latent coefficient trajectories.

### 2.2 Subspace Metrics
1. **Cumulative Explained Variance:**

$$\text{ExpVar}(r) = \frac{\sum_{i=1}^r \sigma_i^2}{\sum_{i=1}^T \sigma_i^2}$$

2. **Relative Reconstruction Error:**

$$\text{RecErr}(r) = \frac{\|D - U_r C_r\|_F}{\|D\|_F} = \sqrt{1 - \text{ExpVar}(r)}$$

### 2.3 Latent Extrapolation & Weight Reconstruction
Given the $r$ latent coordinate series $c_i \in \mathbb{R}^T$ ($i \in \{1, \dots, r\}$), we forecast future coefficients $\hat{c}_i(T + h)$ over horizon $h$ using three paradigms:

1. **Low-Rank Linear Extrapolation:**
   Using the empirical slope $m_i$ estimated over the trajectory context window:
   $$\hat{c}_i(T + h) = c_i(T) + m_i \cdot h$$

2. **Low-Rank Damped Linear Extrapolation:**
   Attenuating high-frequency overshoot using damping factor $\gamma = 0.9$:
   $$\hat{c}_i(T + h) = c_i(T) + m_i \cdot \sum_{k=1}^h \gamma^k$$

3. **Low-Rank TimesFM 2.0 (500M):**
   Evaluating zero-shot autoregressive prediction directly on the $r$ coordinate channels:
   $$\hat{c}_i(T + h) = \text{TimesFM}(c_i(1:T))$$

4. **Low-Rank TimesFM Shuffled Control:**
   Randomly permuting the time steps of $c_i(1:T)$ before feeding into TimesFM to test temporal causal dependency.

Finally, candidate model weights are reconstructed via linear combination:

$$\hat{W}_{T+h} = W_{\text{ref}} + U_r \hat{c}(T+h) = W_T + \sum_{i=1}^r \hat{c}_i(T+h) u_i$$

---

## 3. Empirical Findings (10 Independent Random Seeds)

### 3.1 Subspace Concentration & Reconstruction Error
Evaluated over 10 independent random seeds ($42 \dots 51$) with paired confidence intervals (95% CI):

| Dataset | Subspace Rank ($r$) | Cumulative Explained Variance | Relative Reconstruction Error | SVD Overhead |
| :--- | :---: | :---: | :---: | :---: |
| **California Housing**<br>(MLP, 8 Features, Adam) | Rank 1 | **$93.21\% \pm 1.21\%$** | $25.87\% \pm 2.26\%$ | $1.8 \text{ ms}$ |
| | Rank 4 | **$99.44\% \pm 0.14\%$** | $7.38\% \pm 0.90\%$ | $2.1 \text{ ms}$ |
| | Rank 8 | **$99.97\% \pm 0.01\%$** | $1.73\% \pm 0.16\%$ | $2.2 \text{ ms}$ |
| | Rank 16 | **$100.00\% \pm 0.00\%$** | $0.48\% \pm 0.06\%$ | $2.2 \text{ ms}$ |
| **MNIST Classification**<br>(MLP, 2500 Steps, SGD+Mom) | Rank 1 | **$82.10\% \pm 1.08\%$** | $42.28\% \pm 1.27\%$ | $3.2 \text{ ms}$ |
| | Rank 4 | **$97.90\% \pm 0.18\%$** | $14.45\% \pm 0.62\%$ | $3.5 \text{ ms}$ |
| | Rank 8 | **$99.83\% \pm 0.01\%$** | $4.15\% \pm 0.16\%$ | $3.6 \text{ ms}$ |
| | Rank 16 | **$99.98\% \pm 0.00\%$** | $1.35\% \pm 0.04\%$ | $3.8 \text{ ms}$ |

> **Key Finding 1 (Subspace Concentration):** Over $97.9\%$ of all weight movement in stochastic optimization is confined to an intrinsic subspace of rank $r \le 4$. By rank $r = 8$, over $99.8\%$ of total trajectory variance is captured. Stochastic optimizer trajectories are intensely low-dimensional.

---

### 3.2 Performance vs. Rank Across Extrapolation Paradigms

#### California Housing Tabular Regression (10 Seeds, Held-Out Test MSE)
*Frozen Baseline MSE:* **`0.2916 ± 0.0078`** | *Continued Adam (48 Steps):* **`0.2815 ± 0.0054`** | *Matched SWA:* **`0.2981 ± 0.0085`**

| Rank ($r$) | Low-Rank Linear (Divergent) | Low-Rank Damped ($\gamma=0.9$) | Low-Rank TimesFM (Ordered) | Low-Rank TimesFM (Shuffled Control) |
| :---: | :---: | :---: | :---: | :---: |
| **Rank 1** | $0.3213 \pm 0.0210$ | **$0.2938 \pm 0.0078$** | $0.2995 \pm 0.0072$ | $0.2982 \pm 0.0085$ |
| **Rank 4** | $0.5259 \pm 0.1180$ 💥 | $0.3040 \pm 0.0123$ | $0.3030 \pm 0.0089$ | **$0.2979 \pm 0.0087$** |
| **Rank 8** | $0.5626 \pm 0.1472$ 💥 | $0.3050 \pm 0.0135$ | $0.3025 \pm 0.0088$ | $0.2982 \pm 0.0085$ |
| **Rank 16** | $0.5631 \pm 0.1477$ 💥 | $0.3050 \pm 0.0135$ | $0.3024 \pm 0.0088$ | $0.2982 \pm 0.0085$ |

#### MNIST Image Classification (10 Seeds, Held-Out Test Cross-Entropy Loss)
*Frozen Baseline Loss:* **`0.2038 ± 0.0097`** ($94.01\%$) | *Continued SGD (64 Steps):* **`0.1983 ± 0.0082`** ($94.13\%$) | *Matched SWA:* **`0.1810 ± 0.0042`** ($94.75\%$)

| Rank ($r$) | Low-Rank Linear (Divergent) | Low-Rank Damped ($\gamma=0.9$) | Low-Rank TimesFM (Ordered) | Low-Rank TimesFM (Shuffled Control) |
| :---: | :---: | :---: | :---: | :---: |
| **Rank 1** | $0.6640 \pm 0.2406$ | $0.2280 \pm 0.0186$ | **$0.1880 \pm 0.0050$** 🟢 | **$0.1808 \pm 0.0042$** 🏆 |
| **Rank 4** | $2.7066 \pm 1.1067$ 💥 | $0.2710 \pm 0.0395$ | $0.1932 \pm 0.0063$ | **$0.1810 \pm 0.0040$** 🏆 |
| **Rank 8** | $2.8602 \pm 1.0503$ 💥 | $0.2761 \pm 0.0418$ | $0.1973 \pm 0.0073$ | $0.1814 \pm 0.0040$ |
| **Rank 16** | $2.7887 \pm 0.9927$ 💥 | $0.2748 \pm 0.0408$ | $0.1973 \pm 0.0072$ | $0.1815 \pm 0.0040$ |

---

## 4. Mechanistic Insights & The Linear Divergence Phenomenon

### 4.1 Why Naive Low-Rank Linear Extrapolation Explodes
As seen in the tables above, naive low-rank linear extrapolation suffers a catastrophic failure mode at ranks $r \ge 4$:
- On MNIST, loss explodes from `0.20` to `2.71 - 2.86` ($>13\times$ degradation).
- On California Housing, MSE doubles from `0.29` to `0.56`.

**Mathematical Cause:**
The first 1–2 singular vectors capture slow drift (quasi-deterministic optimization momentum). However, higher singular vectors ($u_3, u_4, \dots, u_r$) capture oscillatory, stochastic mini-batch noise. Because a linear fit estimates an instantaneous gradient across noise fluctuations:

$$\hat{c}_i(T+h) = c_i(T) + m_i \cdot h$$

For noise components where $m_i \neq 0$ purely by sample variance, projecting this displacement back into high-dimensional space ($U_r \hat{c}$) pushes weights drastically orthogonal to the manifold of low loss, destroying calibration and activation norms.

### 4.2 Why TimesFM Remains Stable & The Shuffled History Control
In contrast to linear extrapolation, **TimesFM does not explode**:
- Low-Rank TimesFM achieves **`0.1880 ± 0.0050`** at Rank 1 on MNIST, significantly beating the frozen checkpoint (`0.2038`, $p < 0.001$).
- More importantly, **Low-Rank TimesFM Shuffled Control** achieves **`0.1808 ± 0.0042`**, which **identically matches Matched SWA (`0.1810 ± 0.0042`)**.

**Mechanism Decoded:**
TimesFM’s attention layers and output heads act as a non-linear regularizer / shrinkage operator. When presented with stationary, bounded latent oscillations in optimization plateaus, TimesFM predicts a reversion to the mean rather than extrapolating outward. When time is shuffled, TimesFM reverts fully to estimating the global empirical centroid of the trajectory, which is algebraically identical to Stochastic Weight Averaging (SWA).

---

## 5. Compute, Latency & Memory Profiling

By reducing trajectory forecasting from $P \approx 10{,}000 - 100{,}000$ independent series to $r \le 16$ latent coordinate series, the computational profile changes by orders of magnitude:

| Stage / Method | Number of Series | Wall-Clock GPU Time | Peak VRAM Allocated | Relative Speedup vs Per-Weight |
| :--- | :---: | :---: | :---: | :---: |
| **SVD Decomposition ($D \in \mathbb{R}^{P \times 64}$)** | — | **$2.2 \text{ ms}$** | $< 1 \text{ MB}$ | — |
| **Low-Rank Linear (Rank 4)** | $4$ | **$5.0 \text{ ms}$** | $< 1 \text{ MB}$ | $9{,}180\times$ faster |
| **Low-Rank TimesFM (Rank 1)** | $1$ | **$31.8 \text{ ms}$** | $48 \text{ MB}$ | $1{,}440\times$ faster |
| **Low-Rank TimesFM (Rank 4)** | $4$ | **$52.1 \text{ ms}$** | $52 \text{ MB}$ | **$880\times$ faster** ⚡ |
| **Low-Rank TimesFM (Rank 16)** | $16$ | **$219.0 \text{ ms}$** | $64 \text{ MB}$ | $210\times$ faster |
| **Full Per-Weight TimesFM ($P=10{,}000$)** | $10{,}000$ | **$45{,}900 \text{ ms}$** | $1{,}840 \text{ MB}$ | $1.0\times$ (Baseline) |
| **Continued Training (64 SGD Steps)** | — | **$650.0 \text{ ms}$** | $85 \text{ MB}$ | $70\times$ faster |
| **Matched SWA (Online Average)** | — | **$< 0.1 \text{ ms}$** | $0 \text{ MB}$ | $> 400{,}000\times$ faster |

> **Key Finding 2 (Compute Viability):** Low-Rank TimesFM reduces forecasting latency from **$45.9$ seconds to $52$ milliseconds** (an **$880\times$ speedup**). This renders foundation-model parameter forecasting **$12.5\times$ faster than continuing 64 steps of gradient descent ($52$ ms vs $650$ ms)**.

---

## 6. Decision Rule Evaluation & Final Conclusion

Following the research protocol's pre-registered decision rules:

1. **Does low-rank linear extrapolation match TimesFM?**
   **No.** Naive low-rank linear extrapolation catastrophically diverges ($2.71$ vs $0.19$ loss). Even damped linear extrapolation degrades relative to frozen checkpoints. Zero-shot foundation model forecasting (or non-linear shrinkage) is strictly necessary if performing latent extrapolation.
2. **Does Low-Rank TimesFM beat cheap online baselines (SWA)?**
   **No.** Matched-window SWA achieves `0.1810 ± 0.0042` test loss on MNIST, whereas Low-Rank TimesFM achieves `0.1880 ± 0.0050` (Rank 1) and `0.1932 ± 0.0063` (Rank 4). 
3. **Does the time-series foundation model learn temporal forward dynamics in plateau optimization?**
   **No.** Destroying temporal order via shuffling produces identical or slightly superior performance (`0.1808` vs `0.1810`), demonstrating that the model acts as an implicit variance-reduction filter around the trajectory attractor, not a predictive navigator.
4. **Is Optimization Acceleration Supported?**
   **Conclusion:** While low-rank projection successfully solves the compute latency bottleneck (reducing wall-clock forecasting time from 46 s to 52 ms), the final model quality does not exceed that of zero-parameter, online Stochastic Weight Averaging. Therefore, **using time-series foundation models to accelerate plateau optimization is computationally viable but fundamentally redundant to SWA.** Its valid utility is restricted to offline checkpoint denoising where gradient access is absent.
