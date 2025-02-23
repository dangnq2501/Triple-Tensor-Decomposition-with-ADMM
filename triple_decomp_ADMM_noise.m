function [A, B, C, O, errHist] = triple_decomp_ADMM_noise(X, r, rho, lambda, maxIter, tol)
    [n1, n2, n3] = size(X);
    Xnorm = norm(X(:)); 
    
    A = randn(n1, r, r);
    B = randn(r, n2, r);
    C = randn(r, r, n3);
    Y = zeros(n1, n2, n3);
    O = zeros(n1, n2, n3);
    Lambda = zeros(n1, n2, n3);
    Gamma = zeros(n1, n2, n3);
    
    errHist = zeros(maxIter,1);
    
    for k = 1:maxIter
        A = update_A(X, A, B, C);
        B = update_B(X, A, B, C);
        C = update_C(X, A, B, C);
        
        Y_new = (X - O + rho * (triple_product(A, B, C) + Lambda / rho)) / (1 + rho);
        O_new = soft_threshold(X - Y_new + Gamma / rho, lambda / rho);
        Lambda = Lambda + rho * (triple_product(A, B, C) - Y_new);
        Gamma = Gamma + rho * (X - Y_new - O_new);
        
        
        X_hat = triple_product(A, B, C);
        errHist(k) = norm(X(:) - X_hat(:)) / Xnorm;
        
        if mod(k, 10) == 0
            fprintf('Iteration %d, relative error = %.4e\n', k, errHist(k));
        end
        
        if k > 1 && abs(errHist(k) - errHist(k-1)) < tol * errHist(k-1)
            errHist = errHist(1:k);
            break;
        end
        
        Y = Y_new;
        O = O_new;
    end
end

function F = buildF(B, C)
    % B: (r x n2 x r), C: (r x r x n3)
    [r, n2, ~] = size(B);
    [~, ~, n3] = size(C);
    % F = zeros(r^2, n2*n3);
    % for j = 1:n2
    %     for t = 1:n3
    %         col_idx = j + (t-1)*n2;
    %         for q = 1:r
    %             for s = 1:r
    %                 row_idx = q + (s-1)*r;
    %                 F(row_idx, col_idx) = B(q,j,s) * C(q,s,t);
    %             end
    %         end
    %     end
    % end
    B_unfold = reshape(unfold(B, 2), [n2, r*r, 1]);  
    C_unfold = reshape(unfold(C, 3)', [1, r*r, n3]);
    F = B_unfold .* C_unfold;
    F = reshape(F, [n2, r, r, n3]);
    F = reshape(permute(F, [2, 3, 1, 4]), [r*r, n2*n3]);
end

function G = buildG(A, C)
    % A: (n1 x r x r), C: (r x r x n3)
    [n1, r, ~] = size(A);
    [~, ~, n3] = size(C);
    % G = zeros(r^2, n1*n3);
    % for i = 1:n1
    %     for t = 1:n3
    %         col_idx = i + (t-1)*n1;
    %         for p = 1:r
    %             for s = 1:r
    %                 row_idx = p + (s-1)*r;
    %                 G(row_idx, col_idx) = A(i,p,s) * C(p,s,t);
    %             end
    %         end
    %     end
    % end
    A_unfold = reshape(unfold(A, 1), [n1, r*r, 1]);  
    C_unfold = reshape(unfold(C, 3)', [1, r*r, n3]);
    G = A_unfold .* C_unfold;
    G = reshape(G, [n1, r, r, n3]);
    G = reshape(permute(G, [2, 3, 1, 4]), [r*r, n1*n3]);
end

function H = buildH(A, B)
    % A: (n1 x r x r), B: (r x n2 x r)
    [n1, r, ~] = size(A);
    [~, n2, ~] = size(B);
    % H = zeros(r^2, n1*n2);
    % for i = 1:n1
    %     for j = 1:n2
    %         col_idx = i + (j-1)*n1;
    %         for p = 1:r
    %             for q = 1:r
    %                 row_idx = p + (q-1)*r;
    %                 H(row_idx, col_idx) = A(i,p,q) * B(p,j,q);
    %             end
    %         end
    %     end
    % end
    A_unfold = reshape(unfold(A, 1), [n1, r*r, 1]);  
    B_unfold = reshape(unfold(B, 2)', [1, r*r, n2]);
    H = A_unfold .* B_unfold;
    H = reshape(H, [n1, r, r, n2]);
    H = reshape(permute(H, [2, 3, 1, 4]), [r*r, n1*n2]);
end

function O = soft_threshold(X, lam)
    % Hàm Soft-thresholding: O = sign(X) * max(|X| - lam, 0)
    O = sign(X) .* max(abs(X) - lam, 0);
end

function A = update_A(X, A, B, C)
    % Cập nhật A bằng ALS (Alternating Least Squares)
    X1 = unfold(X, 1);
    F = buildF(B, C);
    A1 = (X1 * F') / (F * F' + 1e-12 * eye(size(F,1)));
    A = reshape_A_from_A1(A1, size(A,1), size(A,2));
end

function B = update_B(X, A, B, C)
    % Cập nhật B bằng ALS
    X2 = unfold(X, 2);
    G = buildG(A, C);
    B2 = (X2 * G') / (G * G' + 1e-12 * eye(size(G,1)));
    B = reshape_B_from_B2(B2, size(B,2), size(B,1));
end

function C = update_C(X, A, B, C)
    % Cập nhật C bằng ALS
    X3 = unfold(X, 3);
    H = buildH(A, B);
    C3 = (X3 * H') / (H * H' + 1e-12 * eye(size(H,1)));
    C = reshape_C_from_C3(C3, size(C,3), size(C,1));
end

function Xn = unfold(X, mode)
    % Chuyển tensor X về dạng unfold (matrix)
    [n1, n2, n3] = size(X);
    switch mode
        case 1
            Xn = reshape(X, [n1, n2 * n3]);
        case 2
            Xn = reshape(permute(X, [2, 1, 3]), [n2, n1 * n3]);
        case 3
            Xn = reshape(permute(X, [3, 1, 2]), [n3, n1 * n2]);
        otherwise
            error('Mode must be 1, 2, or 3.');
    end
end

function Xhat = triple_product(A, B, C)
    % Tính triple product: Xhat = A * B * C
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
                        s = s + A(i, p, q) * B(p, j, q) * C(p, q, t);
                    end
                end
                Xhat(i, j, t) = s;
            end
        end
    end
end

function A = reshape_A_from_A1(A1, n1, r)
    A = zeros(n1, r, r);
    for i = 1:n1
        A(i, :, :) = reshape(A1(i, :), [r, r]);
    end
end

function B = reshape_B_from_B2(B2, n2, r)
    B = zeros(r, n2, r);
    for j = 1:n2
        B(:, j, :) = reshape(B2(j, :), [r, r]);
    end
end

function C = reshape_C_from_C3(C3, n3, r)
    C = zeros(r, r, n3);
    for t = 1:n3
        C(:, :, t) = reshape(C3(t, :), [r, r]);
    end
end
