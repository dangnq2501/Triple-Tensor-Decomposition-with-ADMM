# Triple Tensor Decomposition with ADMM and L₁/₂ Regularization

This repository provides a MATLAB implementation of a triple tensor decomposition algorithm using the Alternating Direction Method of Multipliers (ADMM) in the presence of noise. In this implementation, the data fidelity term is enforced by a least squares fit, while the noise is regularized using the \( L_1 \) or \( L_{1/2} \) quasi-norm to promote sparsity.

## Optimization Problem

The algorithm solves the following optimization problem:

$$
\min_{O, A, B, C} \; \lambda \|X - ABC - O\|_F^2 + R(O)
$$


where:
- \( X \) is the observed tensor,
- \( A, B, C \) are the factor tensors,
- \( O \) represents the noise component,
- \( \lambda \) is a parameter that controls the trade-off between data fidelity and sparsity, and
- \( R(O) \) is the regularization term, which can be either:
  - \( \|O\|_1 \) (L1 norm for promoting sparsity), or
  - \( \|O\|_{1/2} \) (L1/2 quasi-norm for enhanced sparsity).

## Features

- **Triple Tensor Decomposition:** Factorizes a tensor \( X \) into three components \( A \), \( B \), and \( C \) using an Alternating Least Squares (ALS) approach.
- **ADMM Framework:** Uses ADMM to alternate between least squares updates for the factor matrices and proximal updates for the noise term.
- **L1/2 Regularization:** Implements a custom proximal operator for the \( L_{1/2} \) quasi-norm to enforce sparsity in the noise.
- **Data Fidelity Weighting:** Incorporates a parameter \( \lambda \) to control the trade-off between data fidelity and sparsity regularization.
