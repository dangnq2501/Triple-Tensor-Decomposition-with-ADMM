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

### Problem formulation
We estimate low-rank cores \((A,B,C)\) and sparse corruption \(O\) by
```math
\min_{A,B,C,O,E}\; \lambda\|E\|_1 + \tfrac{\alpha_1}{2}\|A\|_F^2 + \tfrac{\alpha_2}{2}\|B\|_F^2 + \tfrac{\alpha_3}{2}\|C\|_F^2
\quad \text{s.t. } X = L + O,\; E = O,\; L=\text{TriTD}(A,B,C).
```
The augmented Lagrangian is optimized by ADMM with soft-thresholding for the sparse copy \(E\) and ridge-regularized least squares for \((A,B,C)\).

### RPAS (Reshape–Permute Acceleration Strategy)
Instead of building Kronecker products to form the mode-wise design matrices $F,G,H$, we compute them *Kronecker-free* via `reshape/permute` and a single GEMM per mode (e.g., $F=\text{RPAS}(B,C)$), reducing the cost from $\mathcal{O}(n^3 r^4)$ to $\mathcal{O}(n^2 r^3)$.

---

## Experiments

### Datasets
- **Tensor completion**: `sensor`, `taxi`, `network`, `chicago`  
- **Video background modeling**: `Highway`, `Office`, `PETS2006`, `Sofa` (CDnet 2014; 300 consecutive frames per sequence)

**Settings (completion):** 10% missing; triple rank \(r=5\); missing treated as corrupted. Metrics: **RRE** and wall-clock **Time (s)**.

### Results — Tensor Completion (10% missing)

| Method                 | sensor RRE | Time(s) | taxi RRE | Time(s) | network RRE | Time(s) | chicago RRE | Time(s) |
|------------------------|:----------:|:-------:|:--------:|:-------:|:-----------:|:-------:|:-----------:|:-------:|
| Sofia                  | 0.341      | 15.95   | 0.584    | 598.24  | 0.963       | 12.01   | 0.352       | 194.36  |
| TRLRF                  | 0.316      | 25.58   | 0.280    | 1799.52 | 0.126       | 41.06   | 0.311       | 1318.22 |
| RC-FCTN                | 0.337      | 2.46    | 0.380    | 128.44  | 1.083       | 5.08    | 0.247       | 29.30   |
| TTNN                   | 0.558      | 4.45    | 0.307    | 340.42  | 0.999       | 7.39    | 0.316       | 264.73  |
| **TriTD-ADMM (Ours)**  | **0.279**  | 2.53    | 0.338    | **53.90** | 0.143     | **1.72** | 0.321     | **20.69** |

*TriTD-ADMM delivers competitive accuracy with state-of-the-art runtime across all four datasets.*

### Results — Video Background Modeling (CDnet2014; 300 f)

**Runtime (s)** — lower is better

| Sequence  | TTNN   | Sofia  | TRLRF   | RC-FCTN | **TriTD-ADMM (Ours)** |
|-----------|:------:|:------:|:-------:|:-------:|:----------------------:|
| Highway   | 201.47 | 370.57 | 1031.97 | 50.64   | **33.68** |
| Sofa      | 225.50 | 419.57 | 1147.48 | 56.92   | **37.05** |
| Office    | 226.36 | 424.15 | 1148.17 | 56.64   | **43.98** |
| PETS2006  | 229.23 | 395.39 | 1215.11 | 92.62   | **35.93** |

Qualitatively, TriTD-ADMM cleanly separates foreground \(O\) from background \(L\) and avoids absorbing moving objects into \(L\) in dynamic scenes (e.g., *Highway*).

---

## Complexity & Convergence

Per iteration, TriTD-ADMM costs \(\mathcal{O}(3 n^3 r^2 + 3 n^2 r^4 + 3 r^6)\) with Kronecker-free designs, and exhibits monotone objective descent with convergence to a stationary point under mild conditions.
