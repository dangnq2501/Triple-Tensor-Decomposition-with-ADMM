function [A, B, C, errHist] = triple_decomp_ADMM(X, r, rho, lambda, mu, maxIter, tol, maxInner, tol_inner)
% TRIPLE_DECOMP_ADMM_COMPLETE giải triple tensor decomposition cho tensor X
% theo hàm mục tiêu 0.5*||X - A*B*C||_F^2.
%
% Các cập nhật của A, B, C được thực hiện qua ADMM nội tại (cho mỗi factor)
% với bài toán con:
%
%   min_{Factor} 0.5*||X_mode - Factor * M||_F^2 + (lambda/2)||Factor - Factor_old||_F^2,
%
% trong đó:
%   - X_mode là mode-unfolding của X (mode tương ứng),
%   - M được xây dựng từ các yếu tố còn lại (F = buildF(B,C) cho A, v.v.),
%   - Factor_old là giá trị factor từ vòng trước.
%
% Global ADMM: ta giới thiệu biến phụ Z và biến dual U để ràng buộc
%   Z = A*B*C.
%
% Input:
%   X         - Tensor gốc (n1 x n2 x n3)
%   r         - Triple rank (chọn sao cho r <= mid{n1,n2,n3})
%   rho       - Tham số penalty global cho ADMM (ví dụ: 1)
%   lambda    - Hệ số proximal trong bài toán con (ví dụ: 1e-3)
%   mu        - Tham số penalty nội tại cho ADMM của mỗi factor (ví dụ: 1)
%   maxIter   - Số vòng lặp global (ví dụ: 100)
%   tol       - Ngưỡng hội tụ global (ví dụ: 1e-4)
%   maxInner  - Số vòng lặp tối đa cho ADMM nội tại (ví dụ: 50)
%   tol_inner - Ngưỡng hội tụ nội tại (ví dụ: 1e-4)
%
% Output:
%   A, B, C   - Các tensor phân rã với kích thước:
%                 A: (n1 x r x r), B: (r x n2 x r), C: (r x r x n3)
%   errHist   - Lịch sử relative error của hàm mục tiêu: ||X - A*B*C||_F/||X||_F

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
        fprintf('Global iter %d, rel. error = %.4e\n', k, errHist(k));
        if k > 1 && abs(errHist(k) - errHist(k-1)) < tol*errHist(k-1)
            errHist = errHist(1:k);
            break;
        end
        
        % --- Cập nhật A ---
        % X(1): unfold X theo mode-1 (n1 x (n2*n3))
        X1 = reshape(X, [n1, n2*n3]);
        % X(1) ở đây được dùng làm dữ liệu cho bài toán con của A.
        % F = buildF(B, C) (kích thước: (r^2) x (n2*n3))
        F = buildF(B, C);
        % Lấy giá trị unfolded của A từ vòng trước (A_old)
        A_old_unf = unfold_A(A);  % kích thước: n1 x (r^2)
        % Giải bài toán con cập nhật A qua ADMM nội tại:
        A_new_unf = update_factor_ADMM(X1, F, A_old_unf, lambda, mu, maxInner, tol_inner);
        % Reshape A_new_unf thành tensor A (n1 x r x r)
        A = reshape_A_from_A1(A_new_unf, n1, r);
        
        % --- Cập nhật B ---
        % Unfold X theo mode-2: (n2 x (n1*n3))
        X2 = reshape(permute(X, [2, 1, 3]), [n2, n1*n3]);
        % G = buildG(A, C) (kích thước: (r^2) x (n1*n3))
        G = buildG(A, C);
        B_old_unf = unfold_B(B);  % kích thước: n2 x (r^2)
        B_new_unf = update_factor_ADMM(X2, G, B_old_unf, lambda, mu, maxInner, tol_inner);
        B = reshape_B_from_B2(B_new_unf, n2, r);
        
        % --- Cập nhật C ---
        % Unfold X theo mode-3: (n3 x (n1*n2))
        X3 = reshape(permute(X, [3, 1, 2]), [n3, n1*n2]);
        % H = buildH(A, B) (kích thước: (r^2) x (n1*n2))
        H = buildH(A, B);
        C_old_unf = unfold_C(C);  % kích thước: n3 x (r^2)
        C_new_unf = update_factor_ADMM(X3, H, C_old_unf, lambda, mu, maxInner, tol_inner);
        C = reshape_C_from_C3(C_new_unf, n3, r);
    end
end

%% --- Hàm ADMM nội tại cho bài toán con cập nhật factor ---
function X_new = update_factor_ADMM(Y, M, X_old, lambda, mu, maxInner, tol_inner)
% Giải bài toán con:
%   min_{X}  0.5*||Y - X*M||_F^2 + (lambda/2)*||X - X_old||_F^2,
% sử dụng ADMM với splitting: đặt X = Z, với biến dual U.
%
% Augmented Lagrangian:
%   L(X,Z,U) = 0.5*||Y - Z*M||_F^2 + (lambda/2)*||X - X_old||_F^2 + (mu/2)*||X - Z + U||_F^2.
%
% Các bước ADMM nội tại:
%   Z-update:  Z = (Y*M' + mu*(X + U))/(M*M' + mu I)
%   X-update:  X = (lambda*X_old + mu*(Z - U))/(lambda + mu)
%   U-update:  U = U + (X - Z)
%
% Input:
%   Y        - Dữ liệu unfolded theo mode tương ứng (n x p)
%   M        - Ma trận phụ (kích thước: (r^2) x p)
%   X_old    - Factor unfolded từ vòng trước (n x (r^2))
%   lambda   - Hệ số proximal
%   mu       - Tham số penalty nội tại
%   maxInner - Số vòng lặp ADMM nội tại tối đa
%   tol_inner- Ngưỡng hội tụ nội tại
%
% Output:
%   X_new    - Factor cập nhật (n x (r^2))
    
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

function Xn = unfold(X, mode)
    sz = size(X);
    switch mode
        case 1
            Xn = reshape(X, sz(1), []);
        case 2
            Xn = reshape(permute(X, [2 1 3]), sz(2), []);
        case 3
            Xn = reshape(permute(X, [3 1 2]), sz(3), []);
        otherwise
            error('Mode phải là 1, 2 hoặc 3.');
    end
end

%% --- Hàm xây dựng ma trận phụ ---
function F = buildF(B, C)
    % B: (r x n2 x r), C: (r x r x n3)
    [r, n2, ~] = size(B);
    [~, ~, n3] = size(C);
    F = zeros(r^2, n2*n3);
    for j = 1:n2
        for t = 1:n3
            col_idx = j + (t-1)*n2;
            for q = 1:r
                for s = 1:r
                    row_idx = q + (s-1)*r;
                    F(row_idx, col_idx) = B(q,j,s) * C(q,s,t);
                end
            end
        end
    end
end

function G = buildG(A, C)
    % A: (n1 x r x r), C: (r x r x n3)
    [n1, r, ~] = size(A);
    [~, ~, n3] = size(C);
    G = zeros(r^2, n1*n3);
    for i = 1:n1
        for t = 1:n3
            col_idx = i + (t-1)*n1;
            for p = 1:r
                for s = 1:r
                    row_idx = p + (s-1)*r;
                    G(row_idx, col_idx) = A(i,p,s) * C(p,s,t);
                end
            end
        end
    end
end

function H = buildH(A, B)
    % A: (n1 x r x r), B: (r x n2 x r)
    [n1, r, ~] = size(A);
    [~, n2, ~] = size(B);
    H = zeros(r^2, n1*n2);
    for i = 1:n1
        for j = 1:n2
            col_idx = i + (j-1)*n1;
            for p = 1:r
                for q = 1:r
                    row_idx = p + (q-1)*r;
                    H(row_idx, col_idx) = A(i,p,q) * B(p,j,q);
                end
            end
        end
    end
end

%% --- Hàm “unfold” và “reshape” cho các yếu tố ---
function A_unf = unfold_A(A)
    % Unfold A (n1 x r x r) theo mode-1: trả về ma trận n1 x (r^2)
    [n1, r, ~] = size(A);
    A_unf = zeros(n1, r^2);
    for i = 1:n1
        A_unf(i,:) = reshape(squeeze(A(i,:,:)), [1, r^2]);
    end
end

function B_unf = unfold_B(B)
    % Unfold B (r x n2 x r) theo mode-2: trả về ma trận n2 x (r^2)
    [r, n2, ~] = size(B);
    B_unf = zeros(n2, r^2);
    for j = 1:n2
        B_unf(j,:) = reshape(squeeze(B(:,j,:)), [1, r^2]);
    end
end

function C_unf = unfold_C(C)
    % Unfold C (r x r x n3) theo mode-3: trả về ma trận n3 x (r^2)
    [r, ~, n3] = size(C);
    C_unf = zeros(n3, r^2);
    for t = 1:n3
        C_unf(t,:) = reshape(squeeze(C(:,:,t)), [1, r^2]);
    end
end

function A = reshape_A_from_A1(A1, n1, r)
    % Chuyển A(1) (n1 x r^2) thành tensor A (n1 x r x r)
    A = zeros(n1, r, r);
    for i = 1:n1
        A(i,:,:) = reshape(A1(i,:), [r, r]);
    end
end

function B = reshape_B_from_B2(B2, n2, r)
    % Chuyển B(2) (n2 x r^2) thành tensor B (r x n2 x r)
    B = zeros(r, n2, r);
    for j = 1:n2
        B(:,j,:) = reshape(B2(j,:), [r, r]);
    end
end

function C = reshape_C_from_C3(C3, n3, r)
    % Chuyển C(3) (n3 x r^2) thành tensor C (r x r x n3)
    C = zeros(r, r, n3);
    for t = 1:n3
        C(:,:,t) = reshape(C3(t,:), [r, r]);
    end
end

%% --- Hàm tính triple product ---
function Xhat = triple_product(A, B, C)
    % Tính triple product: Xhat = A * B * C.
    % Xhat(i,j,t) = sum_{p=1}^{r} sum_{q=1}^{r} A(i,p,q)*B(p,j,q)*C(p,q,t)
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
