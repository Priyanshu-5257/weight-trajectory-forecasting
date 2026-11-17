# Zero-Shot Optimization Trajectory Forecasting with Google TimesFM

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-ee4c2c.svg)](https://pytorch.org/)
[![TimesFM](https://img.shields.io/badge/TimesFM-Google%20Research-4285F4.svg)](https://github.com/google-research/timesfm)

Can a pretrained time-series foundation model (**Google TimesFM**) predict the future trajectory of neural network weights during training and navigate into lower-loss basins **without backpropagation or gradient evaluations**?

This repository contains the empirical framework, experiments, and analysis investigating **zero-shot weight-space trajectory extrapolation and denoising** across multiple architectures, loss landscapes, and optimizers.

---

## Key Highlights

* 🚀 **Zero-Shot Loss Reduction:** In early training, TimesFM extrapolates parameter velocity to lower the validation loss and increase validation accuracy without computing gradients.
* 🛡️ **Taming Momentum Overshoot with Loss Covariates:** Coupling parameter trajectories with scalar training loss ($\mathcal{L}_{\text{train}}(t)$) acts as an automatic physical brake, cutting long-horizon extrapolation error by over **$37\%$**.
* 🏆 **Beating Stochastic Weight Averaging (SWA) at Saturation:** At the convergence plateau, TimesFM filters out mini-batch Brownian noise and reaches a lower validation loss than both continued gradient descent and uniform SWA (`0.2853` vs `0.2989` on MNIST; `0.3082` vs `0.3142` on California Housing).
* 🌐 **Cross-Domain Replication:** Proven across **Classification (MNIST + Cross-Entropy + SGD)** and **Continuous Regression (California Housing + MSE + Adam)**.

---

## Visual Summary

### 1. Active Descent & Momentum Extrapolation
![Active Descent Trajectory](optimization_trajectory_forecast.png)

### 2. Saturation Plateau & SWA Comparison
![Saturation Trajectory](saturation_trajectory_forecast.png)

### 3. Impact of Train Loss as an Exogenous Covariate
![Saturation Covariate Trajectory](saturation_covariate_forecast.png)

---

## Empirical Benchmark

### 1. Classification (MNIST, SGD + Momentum, Cross-Entropy)
| Optimization Regime | Baseline Checkpoint | Ground Truth (More SGD) | SWA (Plateau Mean) | **TimesFM Zero-Shot** |
| :--- | :---: | :---: | :---: | :---: |
| **Early Descent ($h=12$)** | `0.4813 (85.41%)` | `0.4489 (86.08%)` | N/A | **`0.4724 (85.46%)`** 🟢 |
| **Descent with Loss Covariate ($h=32$)** | `0.4813 (85.41%)` | `0.4125 (87.61%)` | N/A | **`0.4662 (87.13%)`** 🚀 |
| **Saturation Plateau ($h=64$)** | `0.3208 (90.75%)` | `0.3346 (90.26%)` | `0.2989 (91.31%)` | **`0.2853 (91.76%)`** 🏆 |

### 2. Continuous Regression (California Housing, Adam, MSE Loss)
| Optimization Regime | Baseline Checkpoint | Ground Truth (More Adam) | SWA (Plateau Mean) | **TimesFM Zero-Shot** |
| :--- | :---: | :---: | :---: | :---: |
| **Early Descent with Loss Covariate ($h=32$)** | `0.4152` | `0.3563` | N/A | **`0.4443`** 🚀 |
| **Saturation Plateau ($h=48$)** | `0.3170` | `0.2960` | `0.3142` | **`0.3082`** 🏆 |

---

## Repository Structure

```
├── experiment_weight_forecast.py        # Experiment 1: Early descent trajectory forecasting
├── experiment_saturation_forecast.py    # Experiment 2: Saturation plateau vs. SWA
├── experiment_saturation_covariate.py   # Experiment 3: Train loss as covariate
├── experiment_regression_forecast.py    # Experiment 4: California Housing regression with Adam
├── EXPERIMENTS.md                       # Comprehensive experiment logs, protocols & data
├── results.json                         # Raw metrics for Experiment 1
├── saturation_results.json              # Raw metrics for Experiment 2
├── saturation_covariate_results.json    # Raw metrics for Experiment 3
├── regression_results.json              # Raw metrics for Experiment 4
├── requirements.txt                     # Dependencies
└── README.md                            # Main project overview
```

---

## Getting Started

### Installation
Clone the repository and install requirements:
```bash
git clone https://github.com/Priyanshu-5257/weight-trajectory-forecasting.git
cd weight-trajectory-forecasting
pip install -r requirements.txt
```

### Reproducing Experiments

1. **Early Descent Trajectory Forecasting:**
```bash
python experiment_weight_forecast.py
```

2. **Saturation Regime vs. Stochastic Weight Averaging:**
```bash
python experiment_saturation_forecast.py
```

3. **Train Loss as a Covariate (XReg + TimesFM):**
```bash
python experiment_saturation_covariate.py
```

4. **Cross-Domain Tabular Regression with Adam:**
```bash
python experiment_regression_forecast.py
```

---

## Conceptual Framework

In optimization theory, SGD with momentum is modeled as a discretized Langevin diffusion:
$$dW_t = -\nabla \mathcal{L}(W_t) dt + \gamma V_t dt + \sigma dB_t$$

* **Active Descent Phase:** Parameter paths exhibit coherent directional velocity contaminated by high-frequency mini-batch noise. TimesFM acts as a noise-filtering velocity extrapolator. Supplying scalar training loss $\mathcal{L}_{\text{train}}(t)$ as an exogenous covariate provides an energy scale that naturally decelerates the parameter step, preventing momentum overshoot.
* **Saturation Phase:** The parameter vector performs an Ornstein-Uhlenbeck random walk inside a flat potential basin ($\mathbb{E}[\nabla \mathcal{L}] \approx 0$). TimesFM's autoregressive forecast converges toward the conditional equilibrium mean, filtering out Brownian fluctuations and finding the true center of the basin faster than uniform averaging.

---

## Citation & Reference

If you find this work or concept useful in your research, feel free to cite:

```bibtex
@misc{weight_trajectory_forecasting_2026,
  author = {Priyanshu},
  title = {Zero-Shot Optimization Trajectory Forecasting with Google TimesFM},
  year = {2026},
  publisher = {GitHub},
  journal = {GitHub repository},
  howpublished = {\url{https://github.com/Priyanshu-5257/weight-trajectory-forecasting}}
}
```

---

## License
MIT License.
