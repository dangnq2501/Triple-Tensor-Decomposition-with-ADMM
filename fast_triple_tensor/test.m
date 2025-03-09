function [A, B, C, O, errHist] = triple_decomp_ADMM_outlier(X, r, rho, lambda, gamma_A, epsilon, p, theta, maxIter, tol)
    % INPUT:
    %   X       - Tensor dữ liệu gốc (n1 x n2 x n3)
    %   r       - Triple rank (giả sử A: n1 x r x r, B: r x n2 x r, C: r x r x n3)
    %   rho     - Tham số penalty ADMM toàn cục
    %   lambda  - Hệ số regularization cho outlier O (norm 1)
    %   gamma_A - Hệ số regularization nonconvex cho A (để thúc đẩy sparsity của A)
    %   gamma_O - Hệ số regularization cho O (norm 1)
    %   epsilon - Tham số nhỏ dùng trong tính trọng số (cho cả A và O)
    %   p       - Tham số nonconvex, p in (0,1) (cho A)
    %   theta   - Xác định dạng norm nonconvex (ví dụ: theta = 0.5 cho ℓ1/2, hoặc theta = 2/3 cho ℓ2/3)
    %   maxIter - Số vòng lặp toàn cục ADMM
    %   tol     - Ngưỡng hội tụ (relative error)
    %
    % OUTPUT:
    %   A, B, C - Các factor phân rã
    %   O       - Tensor outlier
    %   errHist - Lịch sử sai số: ||X - Y - O||_F/||X||_F, với Y = A*B*C (sau reshape)
    
    [n1, n2, n3] = size(X);
    Xnorm = norm(X(:));
    
    % Khởi tạo các factor A, B, C (theo kích thước phù hợp), biến phụ Y, outlier O, và duals Lambda, Gamma.
    A = randn(n1, r, r);
    B = randn(r, n2, r);
    C = randn(r, r, n3);
    Y = zeros(n1, n2, n3);
    O = zeros(n1, n2, n3);
    Lambda = zeros(n1, n2, n3);  % dual cho ràng buộc Y = A*B*C
    Gamma  = zeros(n1, n2, n3);  % dual cho ràng buộc X = Y + O
    
    errHist = zeros(maxIter,1);
    
    for k = 1:maxIter
        T = triple_product(A, B, C);
        Y_new = (X - O + rho*(T + Lambda/rho)) / (1 + rho);
        
        % --- Cập nhật biến O ---
        % Giải bài toán:
        %   min_O 0.5||X - Y_new - O||_F^2 + lambda||O||_1.
        % Để thúc đẩy sparsity, áp dụng weighted soft thresholding cho O với:
        %   W_O = 1./(|X - Y_new| + epsilon)
        W_O = ones(size(O));
        O_new = weighted_soft_threshold(X - Y_new + Gamma/rho, lambda/rho, W_O);
        

        Lambda = Lambda + rho * (T - Y_new);
        Gamma = Gamma + rho * (X - Y_new - O_new);
        A = update_A(X, A, B, C, gamma_A, epsilon, p, theta);
        B = update_B(X, A, B, C);  
        C = update_C(X, A, B, C);  
        
        % Tính sản phẩm triple của A, B, C.
        
        
        % --- Cập nhật biến phụ Y ---
        % Giải bài toán:
        %   min_Y  0.5||X - Y - O||_F^2 + (rho/2)||Y - (T + Lambda/rho)||_F^2.
        % Nghiệm đóng:

        
        errHist(k) = norm(X(:) - Y_new(:) - O_new(:)) / Xnorm;
        fprintf('Iteration %d, relative error = %.4e\n', k, errHist(k));
        
        if k > 1 && abs(errHist(k) - errHist(k-1)) < tol * errHist(k-1)
            errHist = errHist(1:k);
            break;
        end
        
        Y = Y_new;
        O = O_new;
    end
end

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Hàm update_A với sparsity nonconvex cho A (sử dụng weighted soft thresholding)
function A = update_A(X, A, B, C, gamma_A, epsilon, p, theta)
    % Cập nhật A theo ALS, sau đó áp dụng weighted soft thresholding để thúc đẩy sparsity.
    % Unfold X theo mode 1:
    X1 = unfold(X, 1);
    F = buildF(B, C);
    A_old_unf = (X1 * F') * pinv(F * F' + 1e-12 * eye(size(F,1)));
    
    % Tính trọng số cho A theo công thức:
    %   W_A = 1 ./ ((|A_old_unf| + epsilon).^(theta - p))
    W_A = 1 ./ ((abs(A_old_unf) + epsilon) .^ (theta - p));
    
    % Áp dụng weighted soft thresholding cho A:
    A_new_unf = weighted_soft_threshold(A_old_unf, gamma_A, W_A);
    
    % Reshape lại thành tensor A: kích thước (n1 x r x r)
    A = reshape_A_from_A1(A_new_unf, size(A,1), size(A,2));
end

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Hàm weighted soft thresholding: áp dụng ngưỡng mềm theo trọng số W
function Y = weighted_soft_threshold(X, tau, W)
    % Với mỗi phần tử của X, tính:
    %   Y = sign(X) .* max(|X| - tau .* W, 0)
    Y = sign(X) .* max(abs(X) - tau .* W, 0);
end

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Hàm soft thresholding tiêu chuẩn (dùng cho O)
function Y = soft_threshold(X, tau)
    Y = sign(X) .* max(abs(X) - tau, 0);
end

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Các hàm phụ update_B, update_C, triple_product, unfold, reshape_A_from_A1
% (Giả sử các hàm này được định nghĩa như phiên bản code trước; dưới đây là
% ví dụ minh họa đơn giản)

function B = update_B(X, A, B, C)
    X2 = unfold(X, 2);
    G = buildG(A, C);
    B_old_unf = (X2 * G') * pinv(G * G' + 1e-9* eye(size(G,1)));
    B = reshape_B_from_B2(B_old_unf, size(B,2), size(B,1));
end

function C = update_C(X, A, B, C)
    X3 = unfold(X, 3);
    H = buildH(A, B);
    C_old_unf = (X3 * H') * pinv(H * H' + 1e-9 * eye(size(H,1)));
    C = reshape_C_from_C3(C_old_unf, size(C,3), size(C,1));
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
            error('Mode must be 1, 2, or 3.');
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
                for pIdx = 1:size(A,2)
                    for qIdx = 1:size(A,3)
                        s = s + A(i,pIdx,qIdx)*B(pIdx,j,qIdx)*C(pIdx,qIdx,t);
                    end
                end
                Xhat(i,j,t) = s;
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
        B(:,j,:) = reshape(B2(j,:), [r, r]);
    end
end

function C = reshape_C_from_C3(C3, n3, r)
    C = zeros(r, r, n3);
    for t = 1:n3
        C(:,:,t) = reshape(C3(t,:), [r, r]);
    end
end

function F = buildF(B, C)
    [r, n2, ~] = size(B);
    [~, ~, n3] = size(C);
    B_unfold = reshape(unfold(B, 2), [n2, r*r, 1]);
    C_unfold = reshape(unfold(C, 3)', [1, r*r, n3]);
    F = B_unfold .* C_unfold;
    F = reshape(F, [n2, r, r, n3]);
    F = reshape(permute(F, [2,3,1,4]), [r*r, n2*n3]);
end

function G = buildG(A, C)
    [n1, r, ~] = size(A);
    [~, ~, n3] = size(C);
    A_unfold = reshape(unfold(A, 1), [n1, r*r, 1]);
    C_unfold = reshape(unfold(C, 3)', [1, r*r, n3]);
    G = A_unfold .* C_unfold;
    G = reshape(G, [n1, r, r, n3]);
    G = reshape(permute(G, [2,3,1,4]), [r*r, n1*n3]);
end

function H = buildH(A, B)
    [n1, r, ~] = size(A);
    [~, n2, ~] = size(B);
    A_unfold = reshape(unfold(A, 1), [n1, r*r, 1]);
    B_unfold = reshape(unfold(B, 2)', [1, r*r, n2]);
    H = A_unfold .* B_unfold;
    H = reshape(H, [n1, r, r, n2]);
    H = reshape(permute(H, [2,3,1,4]), [r*r, n1*n2]);
end
