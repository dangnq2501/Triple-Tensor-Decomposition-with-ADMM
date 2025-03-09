# Triple Tensor Decomposition with ADMM and Regularization

This repository provides a MATLAB implementation of a triple tensor decomposition algorithm using the Alternating Direction Method of Multipliers (ADMM) in the presence of noise. In this implementation, the data fidelity term is enforced by a least squares fit, while the noise is regularized using the L1 or L1/2 quasi-norm to promote sparsity.

The algorithm solves the following optimization problem:

\[ \min_{O, A, B, C} \; \lambda\, \|X - A B C - O\|_F^2 + \regularization\ \]

where:
- \(X\) is the observed tensor,
- \(A\), \(B\), and \(C\) are the factor tensors,
- \(O\) represents the noise component, and
- \(\lambda\) is a parameter to increase the data fidelity weight.
- \(regularization\) can be \|O\|_{1} or \|O\|_{1/2}



## Features

- **Triple Tensor Decomposition:** Factorizes a tensor \(X\) into three components \(A\), \(B\), and \(C\) using an ALS (Alternating Least Squares) approach.
- **ADMM Framework:** Uses ADMM to alternate between least squares updates for the factor matrices and proximal updates for the noise term.
- **L1/2 Regularization:** Implements a custom proximal operator for the L1/2 quasi-norm to enforce sparsity in the noise.
- **Data Fidelity Weighting:** Incorporates a parameter \(\lambda\) to control the trade-off between data fidelity and sparsity regularization.

## Repository Structure

