function [A, B, C, O, errHist] = triple_decomp_ADMM_noise(X, r, opts)
    maxIter = opts.maxIter;
    tol = opts.tol;
    lambda = opts.lambda
    rho = opts.rho;
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



function A = update_A(X, A, B, C)
    X1 = unfold(X, 1);
    F = buildF(B, C);
    A1 = (X1 * F') * pinv(F * F' + 1e-9 * eye(size(F,1)));
    A = reshape_A_from_A1(A1, size(A,1), size(A,2));
end

function B = update_B(X, A, B, C)
    % Cập nhật B bằng ALS
    X2 = unfold(X, 2);
    G = buildG(A, C);
    B2 = (X2 * G') * pinv(G * G' + 1e-9 * eye(size(G,1)));
    B = reshape_B_from_B2(B2, size(B,2), size(B,1));
end

function C = update_C(X, A, B, C)
    % Cập nhật C bằng ALS
    X3 = unfold(X, 3);
    H = buildH(A, B);
    C3 = (X3 * H') * pinv(H * H' + 1e-9 * eye(size(H,1)));
    C = reshape_C_from_C3(C3, size(C,3), size(C,1));
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
