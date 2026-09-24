# Learning Optimal Robust-Pathwise Control of Elliptic Equations with Random Coefficients

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![PyTorch](https://img.shields.io/badge/PyTorch-EE4C2C?style=flat&logo=pytorch&logoColor=white)](https://pytorch.org/)

This repository (`q1sun/RpCtrl_StoPDEs`) contains the official code implementation for the paper **"Learning Optimal Robust-Pathwise Control of Elliptic Equations with Random Coefficients"**. 

## 📖 Abstract

Conventional control approaches to optimization problems under uncertainty entail a practical compromise: deterministic controls offer robustness at the expense of pathwise accuracy while stochastic controls can adapt to individual realizations but suffer from the curse of dimensionality. To achieve both robustness and adaptivity, a neural operator-based approach is proposed in this paper for distributed elliptic optimal control problems with high-dimensional random inputs. 

A data-driven training algorithm is first established to learn the mapping from coefficient realizations directly to their optimal pathwise controls. To bypass the burden of generating training data pairs, a physics-informed training algorithm is then proposed and recast as weakly supervised learning using pseudo label targets produced by standard adjoint-based control updates. Numerical experiments with rough coefficients are conducted to validate the effectiveness and efficiency of our robust-pathwise controls, which outperforms the benchmark robust deterministic control on unseen coefficient realizations owing to the generalization capability and fast inference of trained models.

## 🧠 Neural Operator Architectures

This codebase implements and evaluates several neural network architectures for learning the optimal control mappings:
* **U-Net**
* **DeepONet**
* **Fourier Neural Operator (FNO)**

## 🛠️ Prerequisites & Hardware

The neural network training framework is implemented in **PyTorch**. Scripts are configured and benchmarked for execution on an **NVIDIA GeForce RTX 4090 GPU**. Post-processing scripts (including kernel density estimation and log-scale performance distribution plots) require **MATLAB**.

### Python Dependencies
```bash
pip install torch torchvision
pip install numpy scipy matplotlib
