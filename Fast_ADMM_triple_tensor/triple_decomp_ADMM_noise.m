function [A, B, C, O, errHist] = triple_decomp_ADMM_noise(X, r, opts)
opts
%      min_{A,B,C,O} 0.5*||X - A*B*C - O||_F^2 + ||O||_1
%
%   X         - Input tensor (n1 x n2 x n3)
%   r         - Triple rank (r should be <= mid{n1, n2, n3})
%   maxIter   - Maximum number of global ADMM iterations (e.g., 100)
%   tol       - Global convergence tolerance (e.g., 1e-4)
%   rho       - Global ADMM penalty parameter (e.g., 1)
%   lambda    - Proximal parameter for the factor subproblems (e.g., 1e-3)
%   mu        - Inner ADMM penalty parameter for factor updates (e.g., 1)
%   maxInner  - Maximum number of inner ADMM iterations (e.g., 50)
%   tol_inner - Inner ADMM tolerance (e.g., 1e-4)
    maxIter = opts.maxIter;
    tol = opts.tol;
    rho = opts.rho;
    mu = opts.mu;
    maxInner = opts.maxIter;
    tol_inner = opts.tol;
    [n1, n2, n3] = size(X);
    Xnorm = norm(X(:));
    
    A = randn(n1, r, r);
    B = randn(r, n2, r);
    C = randn(r, r, n3);
    

    O = zeros(n1, n2, n3); 
    Z = zeros(n1, n2, n3); 
    U = zeros(n1, n2, n3);  
    
    errHist = zeros(maxIter,1);
    
    for k = 1:maxIter

        Y1 = reshape(X - O, [n1, n2*n3]);  
        F = buildF(B, C);                  
        A_old = unfold(A, 1);              
        A_new_unf = update_factor_ADMM(Y1, F, A_old, lambda, mu, maxInner, tol_inner);
        A = reshape_A_from_A1(A_new_unf, n1, r);
        
        Y2 = reshape(permute(X - O, [2,1,3]), [n2, n1*n3]); 
        G = buildG(A, C);                  
        B_old = unfold(B, 2);              
        B_new_unf = update_factor_ADMM(Y2, G, B_old, lambda, mu, maxInner, tol_inner);
        B = reshape_B_from_B2(B_new_unf, n2, r);
        
        Y3 = reshape(permute(X - O, [3,1,2]), [n3, n1*n2]); 
        H = buildH(A, B);                  
        C_old = unfold(C, 3);              
        C_new_unf = update_factor_ADMM(Y3, H, C_old, lambda, mu, maxInner, tol_inner);
        C = reshape_C_from_C3(C_new_unf, n3, r);
        
        Xhat = triple_product(A, B, C); 
        O = ((X - Xhat) + rho*(Z - U))/(1+rho);
        
        Z = soft_threshold(O + U, 1/rho);
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
        Z = (Y * M' + mu*(X + U)) * pinv(MtM + mu*Iq);
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
% PROX_LP computes the proximal operator for the L_p quasi-norm with p < 1.
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
    S = zeros(size(v));
    z = abs(v);
    thresh = (3/2*tau)^(2/3);
    idx = (z > thresh);
    phi = acos( (tau/8) .* (3./z(idx)).^(3/2) );
    S(idx) = (2/3) * v(idx) .* (1 + cos((2*pi - phi)/3));
elseif p == 2/3
    S = zeros(size(v));
    maxIter = 100;
    tol_newton = 1e-6;
    for i = 1:numel(v)
        vi = v(i);
        x = vi;
        if abs(vi) < 1e-3
            S(i) = 0;
            continue;
        end
        for iter = 1:maxIter
            if x == 0
                break;
            end
            f = x - vi + tau*(2/3)*sign(x)*abs(x)^(-1/3);
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
