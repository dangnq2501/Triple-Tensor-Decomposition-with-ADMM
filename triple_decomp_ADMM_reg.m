function [A, B, C, errHist] = triple_decomp_ADMM_reg(X, r, rho, lam, maxIter, tol)
% TRIPLE_DECOMP_ADMM_REG Thực hiện triple tensor decomposition sử dụng ADMM
% với điều khoản regularization (proximal update) cho A, B, C.
%
% [A, B, C, errHist] = triple_decomp_ADMM_reg(X, r, rho, lam, maxIter, tol)
%
%   Input:
%       X       - Tensor gốc có kích thước (n1 x n2 x n3)
%       r       - Hạng mục tiêu (chọn sao cho r <= mid{n1,n2,n3})
%       rho     - Tham số penalty trong ADMM (ρ > 0)
%       lam     - lamdba
%       maxIter - Max iteraction
%       tol     - convergence rate
%
% ADMM 
%        min_Z  0.5*||X - Z||²_F + (ρ/2)*||A*B*C - Z + U||²_F
%        min_{A} 0.5*ρ*||A*B*C - Y||²_F + 0.5*λ*||A - A_old||²_F, với Y = Z - U.
%
    [n1, n2, n3] = size(X);
    X_norm = norm(X(:));
    
    A = randn(n1, r, r);
    B = randn(r, n2, r);
    C = randn(r, r, n3);
    
    Z = X;
    U = zeros(size(X));
    
    errHist = zeros(maxIter,1);
    
    for k = 1:maxIter
        A_old = A; B_old = B; C_old = C;
        
        Xhat = triple_product(A, B, C);  
        Z = (X + rho*(Xhat - U)) / (1 + rho);
        
   
        Y = Z - U;
        
        Y1 = unfold(Y, 1);          % Y1 size = n1 x (n2*n3)
        F  = buildF(B, C);          % F size: (r^2) x (n2*n3)
        FFt = F * F';

        A_old_unf = zeros(n1, r^2);
        for i = 1:n1
            A_old_unf(i,:) = reshape(A_old(i,:,:), [1, r^2]);
        end

        A1 = (rho * Y1 * F' + lam * A_old_unf) / (rho * FFt + lam * eye(r^2));
        A = reshape_A_from_A1(A1, n1, r);
        
        Y2 = unfold(Y, 2);          % Y2 size = n2 x (n1*n3)
        G  = buildG(A, C);          % G size: (r^2) x (n1*n3)
        GGt = G * G';
        B_old_unf = zeros(n2, r^2);
        for j = 1:n2
            B_old_unf(j,:) = reshape(B_old(:,j,:), [1, r^2]);
        end
        B2 = (rho * Y2 * G' + lam * B_old_unf) / (rho * GGt + lam * eye(r^2));
        B = reshape_B_from_B2(B2, n2, r);
        
        Y3 = unfold(Y, 3);          % Y3: n3 x (n1*n2)
        H  = buildH(A, B);          % H size: (r^2) x (n1*n2)
        HHt = H * H';
        C_old_unf = zeros(n3, r^2);
        for t = 1:n3
            C_old_unf(t,:) = reshape(C_old(:,:,t), [1, r^2]);
        end
        C3 = (rho * Y3 * H' + lam * C_old_unf) / (rho * HHt + lam * eye(r^2));
        C = reshape_C_from_C3(C3, n3, r);
        
        Xhat = triple_product(A, B, C);
        U = U + (Xhat - Z);
        
        errHist(k) = norm(X(:) - Xhat(:)) / X_norm;
        
        if mod(k, 5) == 0
            fprintf('Iteration %d, Relative Error = %.4e\n', k, errHist(k));
        end
        
        if k > 1 && abs(errHist(k) - errHist(k-1)) < tol*errHist(k-1)
            errHist = errHist(1:k);
            break;
        end
    end
end


function Xn = unfold(X, mode)
    [n1, n2, n3] = size(X);
    switch mode
        case 1
            Xn = reshape(X, [n1, n2*n3]);
        case 2
            Xn = reshape(permute(X, [2, 1, 3]), [n2, n1*n3]);
        case 3
            Xn = reshape(permute(X, [3, 1, 2]), [n3, n1*n2]);
        otherwise
            error('Mode phải là 1, 2 hoặc 3.');
    end
end

function F = buildF(B, C)
    [r, n2, ~] = size(B);
    [~, ~, n3] = size(C);
    F = zeros(r^2, n2*n3);
    for j = 1:n2
        for t = 1:n3
            col_idx = j + (t-1)*n2;
            for q = 1:r
                for s = 1:r
                    row_idx = q + (s-1)*r;
                    F(row_idx, col_idx) = B(q, j, s) * C(q, s, t);
                end
            end
        end
    end
end

function G = buildG(A, C)
    [n1, r, ~] = size(A);
    [~, ~, n3] = size(C);
    G = zeros(r^2, n1*n3);
    for i = 1:n1
        for t = 1:n3
            col_idx = i + (t-1)*n1;
            for p = 1:r
                for s = 1:r
                    row_idx = p + (s-1)*r;
                    G(row_idx, col_idx) = A(i, p, s) * C(p, s, t);
                end
            end
        end
    end
end

function H = buildH(A, B)
    [n1, r, ~] = size(A);
    [~, n2, ~] = size(B);
    H = zeros(r^2, n1*n2);
    for i = 1:n1
        for j = 1:n2
            col_idx = i + (j-1)*n1;
            for p = 1:r
                for q = 1:r
                    row_idx = p + (q-1)*r;
                    H(row_idx, col_idx) = A(i, p, q) * B(p, j, q);
                end
            end
        end
    end
end

function A = reshape_A_from_A1(A1, n1, r)
    A = zeros(n1, r, r);
    for i = 1:n1
        A(i,:,:) = reshape(A1(i,:), [r, r]);
    end
end

function B = reshape_B_from_B2(B2, n2, r)
    B = zeros(r, n2, r);
    for j = 1:n2
        B(:, j, :) = reshape(B2(j,:), [r, r]);
    end
end

function C = reshape_C_from_C3(C3, n3, r)
    C = zeros(r, r, n3);
    for t = 1:n3
        C(:,:,t) = reshape(C3(t,:), [r, r]);
    end
end

function Xhat = triple_product(A, B, C)
    [n1, ~, ~] = size(A);
    [~, n2, ~] = size(B);
    [~, ~, n3] = size(C);
    Xhat = zeros(n1, n2, n3);
    for i = 1:n1
        for j = 1:n2
            for t = 1:n3
                s = 0;
                for p = 1:size(A,2)
                    for q = 1:size(A,3)
                        s = s + A(i,p,q) * B(p,j,q) * C(p,q,t);
                    end
                end
                Xhat(i,j,t) = s;
            end
        end
    end
end
