<div align="center">

# 🧠 SWEEP-Net
**Spiking Weakly-supervised EEG Emotion Prototyping Network**

[![PyTorch](https://img.shields.io/badge/PyTorch-%23EE4C2C.svg?style=for-the-badge&logo=PyTorch&logoColor=white)]()
[![Neuromorphic](https://img.shields.io/badge/SNN-Ready-blue)]()
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)]()

*A parameter-efficient, topologically explainable Brain-Computer Interface (BCI) designed to overcome cross-session EEG impedance gaps and anatomical variance.

*[Read the Paper (Coming Soon)] • [View Milestone Notebook](milestones/Project%20Milestone%201.ipynb)

</div>

---

## 🔬 Overview
Current deep learning BCIs struggle with two major flaws: **Session Gaps** (overfitting to daily impedance noise) and **Black-Box Classification** (failing to provide biologically grounded neuro-spatial explainability). Furthermore, dense analog models are highly power-inefficient for wearable deployment. 

**SWEEP-Net** solves this by converting raw 1D EEG into 5D spatiotemporal video tensors and processing them through a custom **Atrous-MobileNet Encoder**. By leveraging **Weakly-Supervised Contrastive Learning (WeakSupCon)** and a **Spiking U-Net Decoder**, the network achieves State-of-the-Art session-invariant representations while physically mapping the topology of human emotion.

---

## 🚀 Key Innovations

*   🗺️ **Density-Adaptive TopoMapping:** Resolves Euclidean spatial distortion by using a t-SNE inspired binary search (Perplexity) to dynamically adjust Gaussian RBF variance ($\sigma$) based on local electrode density.
*   🧬 **Atrous-MobileNet with ODConv & GC-Blocks:** A <1M parameter architecture utilizing Depthwise-Pointwise convolutions to isolate EEG frequency bands. It features Omni-Dimensional Dynamic Convolutions (ODConv) to adapt to skull variance, and Global Context (GC) blocks for long-range fronto-parietal connectivity.
*   ⏱️ **WeakSupCon via SwiGLU-MIL:** Abandons the flawed 1-second label assumption. Uses a Multiple Instance Learning (MIL) SwiGLU attention head to autonomously filter out neutral resting-state windows, aggregating trials into pure affective representations.
*   ⚡ **Spiking Topological Decoder (Phase 2):** Bypasses semantic bottlenecks using high-resolution skip connections to decode 128D embeddings back into biologically accurate 2D Ground Truth Z-Energy Masks, utilizing highly efficient Leaky Integrate-and-Fire (LIF) neurons.

---

## 📊 Phase 1 Results: Contrastive Latent Space

*The UMAP projection below demonstrates the latent space (128D bottleneck) evaluated on a strictly held-out Leave-One-Subject-Out (LOSO) test set.*

<div align="center">
  <img src="milestones/figures/subj1_loso.png" alt="LOSO UMAP Projection" width="70%">
  <p><i><b>Figure 1:</b> LOSO Contrastive Representation. The model successfully segregates Valence extremes (Happy/Sad) into distinct manifolds while maintaining the neurobiologically accurate continuum of High-Arousal/Negative-Valence states (Fear/Disgust).</i></p>
</div>

---

## 🧠 Phase 2: Topological Reconstruction 

Unlike standard classifiers, SWEEP-Net visually outputs the predicted brain activation.

<div align="center">
  <img src="milestones/figures/subj1_gt.png" alt="Ground Truth Masks" width="80%">
  <p><i><b>Figure 2:</b> Ground Truth Z-Energy Masks generated via Class Medoid formulation.</i></p>
</div>

---
