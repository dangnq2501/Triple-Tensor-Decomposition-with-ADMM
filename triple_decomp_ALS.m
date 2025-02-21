function [A, B, C, errHist] = triple_decomp_ALS(X, r, maxIter, tol)
% TRIPLE_DECOMP_ALS thực hiện triple tensor decomposition bằng phương pháp
% Alternating Least Squares (ALS) nguyên thủy.
%
% [A, B, C, errHist] = triple_decomp_ALS(X, r, maxIter, tol)
%
% Input:
%   X       - Tensor gốc có kích thước (n1 x n2 x n3)
%   r       - Triple rank (chọn sao cho r <= mid{n1,n2,n3})
%   maxIter - Số vòng lặp tối đa (ví dụ: 100)
%   tol     - Ngưỡng hội tụ theo relative error (ví dụ: 1e-4)
%
% Output:
%   A, B, C - Các tensor phân rã với kích thước:
%               A: (n1 x r x r)
%               B: (r  x n2 x r)
%               C: (r  x r x n3)
%   errHist - Lịch sử relative error qua các vòng lặp
%
% Mục tiêu: Tìm A, B, C sao cho
%      X ≈ A * B * C,
% với định nghĩa triple product:
%      X(i,j,t) = sum_{p=1}^{r} sum_{q=1}^{r} A(i,p,q)*B(p,j,q)*C(p,q,t).
%
% Thuật toán:
%   1. Khởi tạo ngẫu nhiên A, B, C.
%   2. Vòng lặp cập nhật từng biến theo phương pháp ALS:
%         - Cập nhật A: unfold X theo mode-1: X(1) ≈ A(1) * F, với F = buildF(B,C).
%           => A(1) = X(1)*F'/(F*F').
%         - Cập nhật B: unfold X theo mode-2: X(2) ≈ B(2) * G, với G = buildG(A,C).
%           => B(2) = X(2)*G'/(G*G').
%         - Cập nhật C: unfold X theo mode-3: X(3) ≈ C(3) * H, với H = buildH(A,B).
%           => C(3) = X(3)*H'/(H*H').
%   3. Tính lại xấp xỉ Xhat = triple_product(A,B,C) và kiểm tra hội tụ.
%

    [n1, n2, n3] = size(X);
    Xnorm = norm(X(:));
    
    % Khởi tạo A, B, C ngẫu nhiên
    A = randn(n1, r, r);
    B = randn(r, n2, r);
    C = randn(r, r, n3);
    
    errHist = zeros(maxIter,1);
    
    for k = 1:maxIter
        % Tính xấp xỉ hiện tại: Xhat = A * B * C
        Xhat = triple_product(A, B, C);
        errHist(k) = norm(X(:) - Xhat(:)) / Xnorm;
        if mod(k, 5) == 0
            fprintf('Iteration %d, relative error = %.4e\n', k, errHist(k));
        end
        % Kiểm tra điều kiện dừng
        if k > 1 && abs(errHist(k)-errHist(k-1)) < tol*errHist(k-1)
            errHist = errHist(1:k);
            break;
        end
        
        % Cập nhật A (fix B, C):
        X1 = unfold(X, 1);            % X(1): n1 x (n2*n3)
        F  = buildF(B, C);            % F: (r^2) x (n2*n3)
        % Giải bài toán LS: A(1)*F ≈ X(1)  =>  A(1) = X(1)*F'/(F*F')
        A1 = (X1 * F') / (F * F' + 1e-12*eye(r^2));
        A = reshape_A_from_A1(A1, n1, r);
        
        % Cập nhật B (fix A, C):
        X2 = unfold(X, 2);            % X(2): n2 x (n1*n3)
        G  = buildG(A, C);            % G: (r^2) x (n1*n3)
        B2 = (X2 * G') / (G * G' + 1e-12*eye(r^2));
        B = reshape_B_from_B2(B2, n2, r);
        
        % Cập nhật C (fix A, B):
        X3 = unfold(X, 3);            % X(3): n3 x (n1*n2)
        H  = buildH(A, B);            % H: (r^2) x (n1*n2)
        C3 = (X3 * H') / (H * H' + 1e-12*eye(r^2));
        C = reshape_C_from_C3(C3, n3, r);
    end
end

%% Hàm phụ

function Xn = unfold(X, mode)
% UNFOLD chuyển tensor X (n1 x n2 x n3) thành ma trận theo mode được chỉ định.
    [n1, n2, n3] = size(X);
    switch mode
        case 1
            Xn = reshape(X, [n1, n2*n3]);
        case 2
            Xn = reshape(permute(X, [2, 1, 3]), [n2, n1*n3]);
        case 3
            Xn = reshape(permute(X, [3, 1, 2]), [n3, n1*n2]);
        otherwise
            error('Mode must be 1, 2, or 3.');
    end
end

function F = buildF(B, C)
% BUILDF xây dựng ma trận F từ tensor B và C.
% B: (r x n2 x r), C: (r x r x n3)
% Kết quả F có kích thước: (r^2) x (n2*n3)
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
% BUILDG xây dựng ma trận G từ tensor A và C.
% A: (n1 x r x r), C: (r x r x n3)
% Kết quả G có kích thước: (r^2) x (n1*n3)
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
% BUILDH xây dựng ma trận H từ tensor A và B.
% A: (n1 x r x r), B: (r x n2 x r)
% Kết quả H có kích thước: (r^2) x (n1*n2)
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
% RESHAPE_A_FROM_A1 chuyển ma trận A(1) (n1 x r^2) thành tensor A (n1 x r x r)
    A = zeros(n1, r, r);
    for i = 1:n1
        A(i,:,:) = reshape(A1(i,:), [r, r]);
    end
end

function B = reshape_B_from_B2(B2, n2, r)
% RESHAPE_B_FROM_B2 chuyển ma trận B(2) (n2 x r^2) thành tensor B (r x n2 x r)
    B = zeros(r, n2, r);
    for j = 1:n2
        B(:, j, :) = reshape(B2(j,:), [r, r]);
    end
end

function C = reshape_C_from_C3(C3, n3, r)
% RESHAPE_C_FROM_C3 chuyển ma trận C(3) (n3 x r^2) thành tensor C (r x r x n3)
    C = zeros(r, r, n3);
    for t = 1:n3
        C(:,:,t) = reshape(C3(t,:), [r, r]);
    end
end

function Xhat = triple_product(A, B, C)
% TRIPLE_PRODUCT tính triple product: Xhat = A * B * C.
% Với công thức: 
%   Xhat(i,j,t) = sum_{p=1}^{r} sum_{q=1}^{r} A(i,p,q)*B(p,j,q)*C(p,q,t)
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
