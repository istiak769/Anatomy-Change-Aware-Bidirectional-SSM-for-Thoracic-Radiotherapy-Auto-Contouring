Anatomy-Change-Aware Bidirectional Selective State-Space Memory for Clinically Deployed Thoracic Radiotherapy Auto-Contouring

[![arXiv](https://img.shields.io/badge/arXiv-XXXX.XXXXX-b31b1b.svg)](https://arxiv.org/abs/XXXX.XXXXX)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 3.9+](https://img.shields.io/badge/python-3.9+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-red.svg)](https://pytorch.org/)

This repository contains the official implementation of **DAMM-Net++** for thoracic radiotherapy auto-contouring.

> **DAMM-Net++: Anatomy-Change-Aware Bidirectional Selective State-Space Memory for Clinically Deployed Thoracic Radiotherapy Auto-Contouring**  
> *Anonymous Author One, Anonymous Author Two*  

> 📄 [Preprint](https://arxiv.org/abs/XXXX.XXXXX)

---

## 📌 Overview

Lung cancer remains the leading cause of cancer-related mortality worldwide, with radiation therapy as a cornerstone of treatment. Accurate delineation of organs-at-risk (OARs) and target volumes on planning CT scans is critical for dose distribution, tumor control, and toxicity risk. However, manual contouring is time-intensive, operator-dependent, and subject to inter-observer variability.

**DAMM-Net++** is a **clinically deployed** 2.5D deep learning architecture that addresses three persistent challenges in thoracic radiotherapy auto-contouring:

- **Inter-slice surface incoherence**
- **Systematic failure on small low-contrast targets**
- **Absence of per-case reliability signals**

### Key Highlights

- ✅ **Clinically Deployed** — Production-grade web application with DICOM RTSTRUCT export integrated at Bangladesh Medical University
- ✅ **Multi-Centre Dataset** — 2,146 patients across 4 Bangladeshi tertiary-care hospitals
- ✅ **State-of-the-Art Performance** — Mean Dice: 0.955, HD95: 3.78 mm
- ✅ **Multicenter Reader Study** — 17 oncologists, 150 cases, 75–80% time reduction
- ✅ **External Validation** — 112 patients, <5% internal-to-external drop
- ✅ **Uncertainty Estimation** — Calibrated per-voxel confidence for clinical triage
- ✅ **Efficient** — 21.7M parameters, 22.4 ms/slice on NVIDIA T4 GPU

---

## 🏗️ Architecture

### Overall Architecture

DAMM-Net++ is a 2.5D sequence-to-sequence segmentation network that processes $T=5$ consecutive axial CT slices. The architecture comprises four principal components:

1. **Shared ConvNeXt Encoder** — Extracts multi-scale feature pyramids
2. **Multi-Scale Inter-Slice Transition Branch** — Models anatomical change between adjacent slices
3. **Bidirectional Selective State-Space Memory (SSM)** — Propagates context in both through-plane directions
4. **Memory-Guided Boundary-Aware Decoder** — Reconstructs full-resolution predictions with boundary refinement

<p align="center">
  <img src="figures/DAMM_Net++_Updated_V2.png" alt="DAMM-Net++ architecture" width="90%">
</p>

### Core Modules

#### Multi-Scale Inter-Slice Transition Branch
Computes feature-space difference $\Delta F_t = F_t^{(4)} - F_{t-1}^{(4)}$ and processes it at three spatial scales (native, half, quarter resolution) with channel and spatial attention.

<p align="center">
  <img src="figures/Detail_module_V2.png" alt="Core modules" width="90%">
</p>

#### Bidirectional Selective State-Space Memory
The Mamba-based SSM generates input-dependent parameters $(\Delta_t, B_t, C_t)$ from the anatomy-change-informed representation $X_t = F_t^{(4)} + \Delta E_t$, enabling selective retention of clinically relevant through-plane context.

#### Memory-Guided Boundary-Aware Decoder
Integrates memory-guided skip attention and learnable edge detection to sharpen predictions near organ boundaries.

#### Uncertainty Estimation
Predicts per-pixel log-variance $\log\hat{\sigma}^2_t$ with confidence map $\hat{c}_t = \sigma(-\log\hat{\sigma}^2_t) = 1/(1+\hat{\sigma}^2_t)$ for clinical quality assurance.

---

## 📊 Dataset

| Feature | Details |
|---------|---------|
| **Total Patients** | 2,146 |
| **Institutions** | 4 (Bangladesh Medical University, Square Hospital Ltd., Labaid Hospital, United Hospital Ltd.) |
| **Anatomical Classes** | 9 (Spinal Cord, Esophagus, Heart, Right Lung, Left Lung, Trachea, Body, GTV, CTV) |
| **Internal Split** | Training: 1,424 (70%), Validation: 305 (15%), Test: 305 (15%) |
| **External Cohort** | 112 patients (United Hospital Ltd.) |
| **Inter-Rater Reliability** | IoU > 0.95 for all structures ($n = 10$ annotators) |

---

## 🧠 Model Performance

### Quantitative Results (Key Metrics)

| Structure | Dice | IoU | HD95 (mm) | ASD (mm) | NSD |
|-----------|------|-----|-----------|----------|-----|
| **Macro-Average** | **0.955** | **0.914** | **3.78** | **2.05** | **0.938** |
| GTV | 0.957 | 0.919 | 3.82 | 1.17 | 0.854 |
| CTV | 0.936 | 0.880 | 2.80 | 1.08 | 0.845 |
| Trachea | 0.943 | 0.892 | 1.28 | 0.53 | 0.995 |
| Esophagus | 0.906 | 0.828 | 6.87 | 5.40 | 0.929 |
| Heart | 0.970 | 0.942 | 5.39 | 3.61 | 0.904 |

### Baseline Comparison

DAMM-Net++ outperforms four baseline architectures (nnUNet, nnMamba, UNETR++, SwinUNETR) across all nine structures with statistically significant improvements ($p < 0.05$, Hedges' $g$ up to 1.46).

### Uncertainty Calibration

| Metric | Value |
|--------|-------|
| Weighted Mean ACE | 0.0095 |
| AUROC (IoU < 0.85 prediction) | 0.960 |
| Calibration Slope | 0.99 |

### Reader Study Results

| Metric | Junior (Unaided) | Junior (AI-Assisted) | Improvement |
|--------|------------------|----------------------|-------------|
| Mean IoU | 0.861 | **0.925** | +0.064 |
| Mean HD95 | 6.07 mm | **3.98 mm** | -2.09 mm |
| Contouring Time | 155 min | **31 min** | **80.1%** ↓ |
| Consultation Rate | 51.6% | **13.5%** | **73.8%** ↓ |
| Confidence Score | 0.571 | **0.839** | +0.268 |

### External Validation

| Cohort | Mean IoU | Relative Gap |
|--------|----------|--------------|
| Internal Test | 0.9322 | — |
| External (UHL, $n=112$) | 0.9076 | **-2.64%** |

---

## 💻 Computational Efficiency

| Model | Params (M) | GFLOPs | Inference (ms/slice) | Peak VRAM (MB) | Training (hrs) |
|-------|------------|--------|----------------------|----------------|----------------|
| **DAMM-Net++ (Ours)** | **21.7** | **72.6** | **22.4** | **3,128** | **172.4** |
| nnUNet | 31.2 | 42.1 | 14.2 | 3,452 | 185.6 |
| nnMamba | 15.55 | 51.6 | 16.3 | 2,612 | 158.2 |
| UNETR++ | 46.7 | 89.4 | 72.5 | 4,896 | 312.8 |
| SwinUNETR | 62.2 | 105.8 | 76.4 | 5,234 | 348.6 |

---

## 🚀 Clinical Deployment

DAMM-Net++ is deployed at Bangladesh Medical University through a **production-grade web application** with:

- 🖥️ Interactive contour editing (Varian Eclipse-inspired three-panel orthogonal display)
- 📤 Native DICOM RTSTRUCT export (compatible with Varian Eclipse, RayStation, Elekta Monaco)
- ⚡ Under 20 seconds inference time for a typical thoracic CT (median 196 slices)
- 🔬 Optional uncertainty heat overlay for quality assurance
- 👨‍⚕️ Clinician-in-the-loop workflow: final contours subject to radiation-oncologist review

<p align="center">
  <img src="figures/Webview.png" alt="Web application interface" width="90%">
</p>

---

## 🧪 Ablation Studies

### Component Contribution

| Configuration | Dice | HD95 (mm) |
|---------------|------|-----------|
| ConvNeXt U-Net (single-slice) | 0.9047 | 7.94 |
| + Inter-slice transition | 0.9128 | 6.71 |
| + Selective SSM (forward) | 0.9251 | 5.33 |
| + Bidirectional + gated fusion | 0.9340 | 4.62 |
| + Memory cross-attention | 0.9402 | 4.21 |
| + MGBA decoder | 0.9481 | 3.94 |
| **+ Uncertainty (Full)** | **0.9545** | **3.78** |

### Controlled Ablation: Inter-Slice Transition Signal

| Model | Dice | $\Delta$ vs SSM only |
|-------|------|---------------------|
| SSM only | 0.9251 | — |
| Randomized + SSM | 0.9254 | +0.0003 (ns) |
| Shuffled + SSM | 0.9282 | +0.0031* |
| **Real + SSM** | **0.9350** | **+0.0099**** |

---

## 🛠️ Requirements

```bash
# Core dependencies
- Python >= 3.9
- PyTorch >= 2.0.0
- torchvision >= 0.15.0
- numpy >= 1.21.0
- scipy >= 1.7.0
- SimpleITK >= 2.3.0
- pydicom >= 2.3.0
- matplotlib >= 3.5.0
- scikit-image >= 0.19.0
