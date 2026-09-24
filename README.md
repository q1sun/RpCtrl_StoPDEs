# Learning Optimal Robust-Pathwise Control of Elliptic Equations with Random Coefficients

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![PyTorch](https://img.shields.io/badge/PyTorch-EE4C2C?style=flat&logo=pytorch&logoColor=white)](https://pytorch.org/)

This repository (`q1sun/RpCtrl_StoPDEs`) contains our code implementation for the paper **"Learning Optimal Robust-Pathwise Control of Elliptic Equations with Random Coefficients"**. 

## 📖 Abstract

Conventional control approaches to optimization problems under uncertainty entail a practical compromise: deterministic controls offer robustness at the expense of pathwise accuracy while stochastic controls can adapt to individual realizations but suffer from the curse of dimensionality. To achieve both robustness and adaptivity, a neural operator-based approach is proposed in this paper for distributed elliptic optimal control problems with high-dimensional random inputs. 

A data-driven training algorithm is first established to learn the mapping from coefficient realizations directly to their optimal pathwise controls. To bypass the burden of generating training data pairs, a physics-informed training algorithm is then proposed and recast as weakly supervised learning using pseudo label targets produced by standard adjoint-based control updates. Numerical experiments with rough coefficients are conducted to validate the effectiveness and efficiency of our robust-pathwise controls, which outperforms the benchmark robust deterministic control on unseen coefficient realizations owing to the generalization capability and fast inference of trained models.

## 🛠️ Prerequisites 

**MATLAB** is required for dataset generation (Karhunen-Loève Expansion, classical optimal control solvers, et al）. Neural operators are implemented in **PyTorch** and optimized for an **NVIDIA GeForce RTX 4090 GPU Cards**. 

## 🚀 Repository Structure

The codebase is organized into Supervised/Data-Driven and Unsupervised/Physics-Informed Learning directories:

* **Supervised Learning**: Data-driven robust-pathwise control evaluated under correlation lengths of $\ell = 0.2$ and $\ell = 0.08$:
  * `RpCtrl-SL-l2e-1/` 
  * `RpCtrl-SL-l8e-2/` 
* **Unsupervised Learning**: Physics-informed robust-pathwise control evaluated for comparison with data-driven model:
  * `RpCtrl-unSL-l2e-1/`

### Sub-directory Details (e.g., `RpCtrl-SL-l2e-1`)

Each configuration directory contains the complete pipeline from data generation to model testing:

* `Checkpoints/` - Directory to store trained PyTorch model weights.
* `Data/` - Directory containing the generated input coefficient fields and target pathwise optimal controls.
* `Figures/` - Directory for saving output visualization plots.
* `Models/` - PyTorch implementations of the neural operator architectures.
* **Core Pipeline Scripts:**
  * `setup_KLE_normal.m` - MATLAB script to generate training data and target pathwise controls.
  * `train_KLE_normal.py` - PyTorch script for training the neural operator models.
  * `train_KLE_normal.m` - MATLAB script to evaluate saved checkpoints and identify the best-performing model.
  * `test_KLE_normal.m` - MATLAB script to evaluate the best model on the test dataset and compare results against benchmark robust deterministic controls.
* **Execution Wrappers:**
  * `run_setup.sh`, `run_train.sh`, `run_test.sh` - Bash scripts wrapping the pipeline steps.
* **Configuration Files:**
  * `RpCtrl-setup-s1e2.txt`, `RpCtrl-train-s1e2.txt`, `RpCtrl-test-s1e2.txt` - Configuration parameters (e.g., sample sizes, network hyper-parameters).
