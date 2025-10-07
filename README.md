# Robust Triple Tensor Decomposition (TriTD)

This repository contains the implementation and experiments for our paper:

> **Robust Triple Tensor Decomposition for Corrupted and Incomplete Multiway Data**

---

## Overview

Tensors provide a principled language for structuring multiway signals across space, time, and modality, serving as a unifying scaffold for modern analysis.  
They have enabled state-of-the-art advances in **compression, completion, and representation learning** across videos, hyperspectral imaging, and network telemetry ~\[[Kolda & Bader, 2009](https://epubs.siam.org/doi/10.1137/07070111X); [Sedighin et al., 2024](#references); [Thanh et al., 2023](#references); [Wang et al., 2023](#references); [Vijayaraghavan et al., 2020](#references)\].

Building on this foundation, our work introduces a **robust TriTD framework** that extends the *Triple Tensor Decomposition* model ~\[[Qi et al., 2021](#references)\] to handle **sparse corruptions**, **structured missingness**, and **foreground–background separation**.  
We propose a two-constraint **ADMM optimization scheme** with a convex $\ell_1$ residual penalty, leading to strong robustness and efficient convergence.

---

## Key Features

- **Triple Tensor Decomposition (TriTD)** factorization into three coupled 3-way cores  
- **Robust data modeling** via convex $\ell_1$ sparse residual  
- **Two-constraint ADMM solver** with provable convergence (Lemma 1)  
- **Kronecker-free implementation** using reshape–permute construction  
- **Memory-efficient updates** with small $r^2 \!\times\! r^2$ Gram systems  
- **Benchmarked** on:
  - Tensor completion datasets: `sensor`, `taxi`, `network`, `chicago`
  - Video background modeling: `Highway`, `Office`, `PETS2006`, `Sofa`

---

## Algorithmic Highlights

### 1. Model Formulation
min_{A,B,C,O,E}
½‖ X − [[A,B,C]] − O − E ‖_F² + λ‖O‖₁
subject to:
structural constraints on E,
nonnegativity on (A,B,C)
subject to structural constraints on $\EEE$ and nonnegativity on $(\AAA,\BBB,\CCC)$.

### 2. Optimization
Each iteration performs:
1. **Core Updates:** Ridge-regularized least squares on mode-wise unfoldings.  
2. **Sparse Update:** Closed-form soft-thresholding.  
3. **Dual Update:** Lightweight Lagrange multiplier updates.  

---

## Performance Analysis

- **Computational Complexity:**  
  Per iteration cost  
  \[
  \Theta(3n^3 r^2) + \Theta(3n^2 r^3) + \mathcal{O}(r^6),
  \]
  avoiding the $\mathcal{O}(n^3 r^4)$ blow-up from explicit Kronecker builds.

- **Convergence:**  
  The proposed ADMM exhibits monotonic objective descent and convergence to a stationary point under mild conditions.

---
