# Mora SP Cup 2026 – Final Denoising Pipeline

## Overview

This submission implements a hybrid low-light image denoising pipeline combining:

- A residual DnCNN denoiser
- Robust Charbonnier-loss fine-tuning
- Classical wavelet denoising
- Signal-dependent noise characterization
- Noisy-image noise-severity estimation
- Severity-aware adaptive fusion

The final pipeline requires only the noisy input image during inference.  
No ground-truth image is used at test time.

---

## Final Pipeline

For each noisy image:

1. The noisy RGB image is passed through a robust residual DnCNN.
2. A classical wavelet denoised estimate is generated using:
   - Symlet-4 (`sym4`)
   - Decomposition level 4
   - BayesShrink thresholding
   - Soft thresholding
3. The wavelet result is also used as an approximate clean image for noise-severity estimation.
4. Signal-dependent residual statistics are calculated using the fitted RGB bias and variance models.
5. A trained regression model predicts the image-level noise severity.
6. The image is classified into Low, Medium, or High noise severity.
7. The Robust DnCNN and Wavelet outputs are fused using severity-dependent weights.

### Adaptive Fusion Weights

| Predicted Severity | Robust DnCNN | Wavelet |
|---|---:|---:|
| Low | 0.85 | 0.15 |
| Medium | 0.90 | 0.10 |
| High | 0.90 | 0.10 |

The final estimate is

$$
\hat{x} = \alpha D(y) + (1-\alpha)W(y)
$$

where:

- $D(y)$ is the Robust DnCNN output
- $W(y)$ is the Wavelet output
- $\alpha$ is selected according to the predicted noise severity

## Files

The `scripts/` directory contains:

```text
scripts/
├── denoise.py
├── dncnn_model.py
├── best_robust_dncnn.pth
├── severity_estimator.joblib
└── README.md

## Model Files

The final inference pipeline requires the following frozen model files:

| File | Location | SHA-256 |
|---|---|---|
| `best_robust_dncnn.pth` | `scripts/best_robust_dncnn.pth` | `FC6A7FAA227FE32EF8C0C1955EBDAC1581422B6364D7DC1F740828E3D73A0E3B` |
| `severity_estimator.joblib` | `scripts/severity_estimator.joblib` | `7628A00D5BB52B8F6DA3EFA82551993C5FA8CC66E1FA247A63D61E6DC5FD79FC` |

These files correspond to the models used to generate the submitted preliminary outputs and must not be modified after the submission deadline.