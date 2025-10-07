# Fast and Robust Triple Tensor Decomposition Under Data Corruption (TriTD-ADMM)

Official code for the paper **“Fast and Robust Triple Tensor Decomposition Under Data Corruption.”**  
We propose **TriTD-ADMM**, a fast and robust solver for Triple Tensor Decomposition (TriTD) that models sparse corruptions with an L1 term and accelerates mode-wise updates via a **reshape–permute (Kronecker-free)** design.

---

## Highlights

- **Robust to outliers:** explicit sparse residual with L1 regularization.
- **Simple & fast ADMM:** closed-form soft-thresholding for the sparse term; tiny `r^2 × r^2` ridge systems for core updates.
- **Kronecker-free acceleration (RPAS):** builds designs `F, G, H` using only `reshape/permute + GEMM` (no Khatri–Rao/Kronecker materialization).
- **Strong results** on tensor completion and video background modeling (BMC): competitive accuracy with substantial speedups (up to **63.71×** vs. baselines in our tests).

---

## Method (TL;DR)

Given a 3-way tensor `X`, TriTD factorizes it via three coupled 3-way cores `(A, B, C)` with “triple rank” `r`.  
We solve the **robust** model

$$
\mathcal X \;=\; \operatorname{TriTD}(\mathcal A,\mathcal B,\mathcal C) \;+\; \mathbf O.
$$

---

## Triple Tensor Decomposition (TriTD)

$$
\mathcal X \;=\; \sum_{p=1}^{r}\sum_{q=1}^{r}\sum_{s=1}^{r}
\;\mathcal A_{:,p,q}\ \circ\ \mathcal B_{s,:,q}\ \circ\ \mathcal C_{s,p,:}\,.
$$
