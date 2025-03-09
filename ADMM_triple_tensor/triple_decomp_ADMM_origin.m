function [A, B, C, O, errHist] = triple_decomp_ADMM_origin(X, r, maxIter, tol, rho, lambda, mu, maxInner, tol_inner)
% TRIPLE_DECOMP_ADMM_GLOBAL_FULL solves the problem:
%
%      min_{A,B,C,O} 0.5*||X - A*B*C - O||_F^2 + ||O||_1
%
% where:
%   A ∈ ℝ^(n1 x r x r), B ∈ ℝ^(r x n2 x r), C ∈ ℝ^(r x r x n3)
%   O is the sparse error tensor (same size as X).
%
% Global ADMM is used to split the problem with auxiliary variable Z for O (O = Z),
% and the augmented Lagrangian is:
%
%      L(A,B,C,O,Z,U) = 0.5*||X - A*B*C - O||_F^2 + ||Z||_1 + (rho/2)*||O - Z + U||_F^2.
%
% The factors A, B, C are updated via inner ADMM (modified ALS) solving:
%
%      min_{Factor}  0.5*||X_mode - Factor*M||_F^2 + (lambda/2)*||Factor - Factor_old||_F^2,
%
% where M is built from the other factors as in the original paper:
%   For A: M = F, with
%        F(q+(s-1)*r, j+(t-1)*n2) = sum_{p=1}^{r} B(p,j,s)*C(p,q,t).
%
% Input:
%   X         - Input tensor (n1 x n2 x n3)
%   r         - Triple rank (r should be <= mid{n1, n2, n3})
%   maxIter   - Maximum number of global ADMM iterations (e.g., 100)
%   tol       - Global convergence tolerance (e.g., 1e-4)
%   rho       - Global ADMM penalty parameter (e.g., 1)
%   lambda    - Proximal parameter for the factor subproblems (e.g., 1e-3)
%   mu        - Inner ADMM penalty parameter for factor updates (e.g., 1)
%   maxInner  - Maximum number of inner ADMM iterations (e.g., 50)
%   tol_inner - Inner ADMM tolerance (e.g., 1e-4)
%
% Output:
%   A, B, C   - Factor tensors with dimensions:
%                A: n1 x r x r,
%                B: r x n2 x r,
%                C: r x r x n3.
%   O         - Sparse error tensor (n1 x n2 x n3)
%   errHist   - Global relative error history: ||X - (A*B*C + O)||_F / ||X||_F

    [n1, n2, n3] = size(X);
    Xnorm = norm(X(:));
    
    % Initialize factors A, B, C randomly.
    A = randn(n1, r, r);
    B = randn(r, n2, r);
    C = randn(r, r, n3);
    
    % Global ADMM variables for splitting O:
    % We enforce the constraint O = Z.
    O = zeros(n1, n2, n3);  % sparse error tensor
    Z = zeros(n1, n2, n3);  % auxiliary variable for O
    U = zeros(n1, n2, n3);  % dual variable for the splitting of O
    
    errHist = zeros(maxIter,1);
    
    for k = 1:maxIter
        % === Update Factors A, B, C using inner ADMM (modified ALS) ===
        % We update factors to minimize: ||(X - O) - A*B*C||_F^2.
        
        Y1 = reshape(X - O, [n1, n2*n3]);  % Unfold (X - O) along mode-1
        F = buildF(B, C);                  % Construct F of size (r^2) x (n2*n3)
        A_old = unfold(A, 1);               % Unfold A into n1 x (r^2)
        A_new_unf = update_factor_ADMM(Y1, F, A_old, lambda, mu, maxInner, tol_inner);
        A = reshape_A_from_A1(A_new_unf, n1, r);
        
        Y2 = reshape(permute(X - O, [2,1,3]), [n2, n1*n3]);  % Unfold (X - O) along mode-2
        G = buildG(A, C);                  % Build G for updating B, size: (r^2) x (n1*n3)
        B_old = unfold(B, 2);               % Unfold B into n2 x (r^2)
        B_new_unf = update_factor_ADMM(Y2, G, B_old, lambda, mu, maxInner, tol_inner);
        B = reshape_B_from_B2(B_new_unf, n2, r);
        
        Y3 = reshape(permute(X - O, [3,1,2]), [n3, n1*n2]);  % Unfold (X - O) along mode-3
        H = buildH(A, B);                  % Build H for updating C, size: (r^2) x (n1*n2)
        C_old = unfold(C, 3);               % Unfold C into n3 x (r^2)
        C_new_unf = update_factor_ADMM(Y3, H, C_old, lambda, mu, maxInner, tol_inner);
        C = reshape_C_from_C3(C_new_unf, n3, r);
        
        Xhat = triple_product(A, B, C);  
        
        O = ((X - Xhat) + rho*(Z - U))/(1+rho);
        
        Z = prox_lp(O + U, 1/rho, 0.5);
        
        U = U + (O - Z);
        
        errHist(k) = norm(X(:) - (Xhat(:) + O(:))) / Xnorm;
        if mod(k, 10) == 0
            fprintf('Global iter %d, rel. error = %.4e\n', k, errHist(k));
        end
        if errHist(k) < tol
            errHist = errHist(1:k);
            break;
        end
    end
end


function X_new = update_factor_ADMM(Y, M, X_old, lambda, mu, maxInner, tol_inner)


    [n, ~] = size(X_old);
    q = size(M,1);  % q should be equal to r^2.
    X = X_old;
    Z = X_old;
    U = zeros(n, q);
    
    MtM = M * M';
    Iq = eye(q);
    
    for iter = 1:maxInner
        % Z-update:
        Z = (Y * M' + mu*(X + U)) * pvin(MtM + mu*Iq);
        % X-update:
        X_new_temp = (lambda*X_old + mu*(Z - U))/(lambda+mu);
        % U-update:
        U = U + (X_new_temp - Z);
        if norm(X_new_temp - Z, 'fro') < tol_inner * norm(X_new_temp, 'fro')
            X = X_new_temp;
            break;
        end
        X = X_new_temp;
    end
    X_new = X;
end



function A = reshape_A_from_A1(A1, n1, r)
% Reshapes an n1 x (r^2) matrix A1 to a tensor A of size (n1 x r x r).
    A = zeros(n1, r, r);
    for i = 1:n1
        A(i,:,:) = reshape(A1(i,:), [r, r]);
    end
end

function B = reshape_B_from_B2(B2, n2, r)
% Reshapes an n2 x (r^2) matrix B2 to a tensor B of size (r x n2 x r).
    B = zeros(r, n2, r);
    for j = 1:n2
        B(:,j,:) = reshape(B2(j,:), [r, r]);
    end
end

function C = reshape_C_from_C3(C3, n3, r)
% Reshapes an n3 x (r^2) matrix C3 to a tensor C of size (r x r x n3).
    C = zeros(r, r, n3);
    for t = 1:n3
        C(:,:,t) = reshape(C3(t,:), [r, r]);
    end
end


function S = prox_lp(v, tau, p)
% PROX_LP computes the proximal operator for the ℓ_p quasi-norm with p < 1.
% It supports p = 1/2 (using a closed-form expression) and p = 2/3 (via an iterative method).
%
% Input:
%   v   - input array (scalar, vector, or matrix) 
%   tau - threshold parameter (nonnegative scalar)
%   p   - norm parameter, either 0.5 or 2/3
%
% Output:
%   S   - the result of prox_{tau ||.||_p}(v) computed elementwise.

if p == 0.5
    % Closed-form proximal operator for ℓ_{1/2} norm.
    S = zeros(size(v));
    z = abs(v);
    % A threshold below which the proximal operator gives zero.
    thresh = (3/2*tau)^(2/3);
    idx = (z > thresh);
    % Compute phi for indices where z is above the threshold.
    phi = acos( (tau/8) .* (3./z(idx)).^(3/2) );
    S(idx) = (2/3) * v(idx) .* (1 + cos((2*pi - phi)/3));
elseif p == 2/3
    % For p = 2/3, we solve for each element x = prox(v) by finding the root of:
    %   f(x) = x - v + tau*(2/3)*sign(x)*|x|^(-1/3) = 0.
    % We use a simple Newton's method iteration.
    S = zeros(size(v));
    maxIter = 100;
    tol_newton = 1e-6;
    for i = 1:numel(v)
        vi = v(i);
        % Initialize x with vi.
        x = vi;
        % Optionally, if vi is too small, set the prox to zero.
        if abs(vi) < 1e-3
            S(i) = 0;
            continue;
        end
        for iter = 1:maxIter
            % Avoid division by zero:
            if x == 0
                break;
            end
            f = x - vi + tau*(2/3)*sign(x)*abs(x)^(-1/3);
            % f'(x) = 1 - tau*(2/9)*abs(x)^(-4/3)
            fp = 1 - tau*(2/9)*abs(x)^(-4/3);
            x_new = x - f/fp;
            if abs(x_new - x) < tol_newton
                x = x_new;
                break;
            end
            x = x_new;
        end
        S(i) = x;
    end
else
    error('prox_lp: p = %g is not implemented', p);
end
end
