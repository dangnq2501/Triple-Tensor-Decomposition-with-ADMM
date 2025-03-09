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
        A1 = (X1 * F') * pinv(F * F' + 1e-9*eye(r^2));
        A = reshape_A_from_A1(A1, n1, r);
        
        X2 = unfold(X, 2);            % X(2): n2 x (n1*n3)
        G  = buildG(A, C);            % G: (r^2) x (n1*n3)
        B2 = (X2 * G') * pinv(G * G' + 1e-9*eye(r^2));
        B = reshape_B_from_B2(B2, n2, r);
        
        X3 = unfold(X, 3);            % X(3): n3 x (n1*n2)
        H  = buildH(A, B);            % H: (r^2) x (n1*n2)
        C3 = (X3 * H') * pinv(H * H' + 1e-9*eye(r^2));
        C = reshape_C_from_C3(C3, n3, r);
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

