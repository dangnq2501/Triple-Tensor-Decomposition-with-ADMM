function [A, B, C, errHist] = triple_decomp_ADMM(X, r, rho, lambda, mu, maxIter, tol, maxInner, tol_inner)
%   min_{Factor} 0.5*||X_mode - Factor * M||_F^2 + (lambda/2)||Factor - Factor_old||_F^2,
%
%   X         - Tensor gốc (n1 x n2 x n3)
%   r         - Triple rank (chọn sao cho r <= mid{n1,n2,n3})
%   rho       - Tham số penalty global cho ADMM (ví dụ: 1)
%   lambda    - Hệ số proximal trong bài toán con (ví dụ: 1e-3)
%   mu        - Tham số penalty nội tại cho ADMM của mỗi factor (ví dụ: 1)


    [n1, n2, n3] = size(X);
    Xnorm = norm(X(:));
    
    % Khởi tạo yếu tố A, B, C ngẫu nhiên
    A = randn(n1, r, r);
    B = randn(r, n2, r);
    C = randn(r, r, n3);
    
    errHist = zeros(maxIter,1);
    
    for k = 1:maxIter
        % Tính reconstruction hiện tại
        Xhat = triple_product(A, B, C);
        errHist(k) = norm(X(:) - Xhat(:)) / Xnorm;
        if mod(k, 5) == 0
            fprintf('Global iter %d, rel. error = %.4e\n', k, errHist(k));
        end
        
        if k > 1 && abs(errHist(k) - errHist(k-1)) < tol*errHist(k-1)
            errHist = errHist(1:k);
            break;
        end
        
        X1 = reshape(X, [n1, n2*n3]);
        F = buildF(B, C);
        A_old_unf = unfold_A(A); 
        A_new_unf = update_factor_ADMM(X1, F, A_old_unf, lambda, mu, maxInner, tol_inner);
        A = reshape_A_from_A1(A_new_unf, n1, r);
        
        X2 = reshape(permute(X, [2, 1, 3]), [n2, n1*n3]);
        G = buildG(A, C);
        B_old_unf = unfold_B(B);  % n2 x (r^2)
        B_new_unf = update_factor_ADMM(X2, G, B_old_unf, lambda, mu, maxInner, tol_inner);
        B = reshape_B_from_B2(B_new_unf, n2, r);
        
        X3 = reshape(permute(X, [3, 1, 2]), [n3, n1*n2]);
        H = buildH(A, B);
        C_old_unf = unfold_C(C);  
        C_new_unf = update_factor_ADMM(X3, H, C_old_unf, lambda, mu, maxInner, tol_inner);
        C = reshape_C_from_C3(C_new_unf, n3, r);
    end
end

function X_new = update_factor_ADMM(Y, M, X_old, lambda, mu, maxInner, tol_inner)
%   min_{X}  0.5*||Y - X*M||_F^2 + (lambda/2)*||X - X_old||_F^2,
%
% Augmented Lagrangian:
%   L(X,Z,U) = 0.5*||Y - Z*M||_F^2 + (lambda/2)*||X - X_old||_F^2 + (mu/2)*||X - Z + U||_F^2.
%
%   Z-update:  Z = (Y*M' + mu*(X + U))/(M*M' + mu I)
%   X-update:  X = (lambda*X_old + mu*(Z - U))/(lambda + mu)
%   U-update:  U = U + (X - Z)
%

    [n, ~] = size(X_old);
    q = size(M,1);  % q = r^2
    X = X_old;
    Z = X_old;
    U = zeros(n, q);
    
    MtM = M * M';
    Iq = eye(q);
    
    for iter = 1:maxInner
        % Z-update:
        Z = (Y * M' + mu*(X + U)) / (MtM + mu*Iq);
        % X-update:
        X_new_temp = (lambda*X_old + mu*(Z - U))/(lambda+mu);
        % U-update:
        U = U + (X_new_temp - Z);
        % Kiểm tra hội tụ nội tại:
        if norm(X_new_temp - Z, 'fro') < tol_inner*norm(X_new_temp, 'fro')
            X = X_new_temp;
            break;
        end
        X = X_new_temp;
    end
    X_new = X;
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
        B(:,j,:) = reshape(B2(j,:), [r, r]);
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
                        s = s + A(i,p,q)*B(p,j,q)*C(p,q,t);
                    end
                end
                Xhat(i,j,t) = s;
            end
        end
    end
end
