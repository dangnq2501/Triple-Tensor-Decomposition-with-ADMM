function [A, B, C, errHist] = triple_decomp_ALS(X, r, maxIter, tol)
    [n1, n2, n3] = size(X);
    Xnorm = norm(X(:));
    
    A = randn(n1, r, r);
    B = randn(r, n2, r);
    C = randn(r, r, n3);
    
    errHist = zeros(maxIter,1);
    
    for k = 1:maxIter
        Xhat = triple_product(A, B, C);
        errHist(k) = norm(X(:) - Xhat(:)) / Xnorm;
        if mod(k, 5) == 0
            fprintf('Iteration %d, relative error = %.4e\n', k, errHist(k));
        end
        if k > 1 && abs(errHist(k)-errHist(k-1)) < tol*errHist(k-1)
            errHist = errHist(1:k);
            break;
        end
        
        X1 = unfold(X, 1);            % X(1): n1 x (n2*n3)
        F  = buildF(B, C);            % F: (r^2) x (n2*n3)
        A1 = (X1 * F') / (F * F' + 1e-12*eye(r^2));
        A = reshape_A_from_A1(A1, n1, r);
        
        X2 = unfold(X, 2);            % X(2): n2 x (n1*n3)
        G  = buildG(A, C);            % G: (r^2) x (n1*n3)
        B2 = (X2 * G') / (G * G' + 1e-12*eye(r^2));
        B = reshape_B_from_B2(B2, n2, r);
        
        X3 = unfold(X, 3);            % X(3): n3 x (n1*n2)
        H  = buildH(A, B);            % H: (r^2) x (n1*n2)
        C3 = (X3 * H') / (H * H' + 1e-12*eye(r^2));
        C = reshape_C_from_C3(C3, n3, r);
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
            error('Mode must be 1, 2, or 3.');
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
% RESHAPE_C_FROM_C3 chuyển ma trận C(3) (n3 x r^2) thành tensor C (r x r x n3)
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
