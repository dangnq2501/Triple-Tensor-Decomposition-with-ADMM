function [A, B, C, errHist] = triple_decomp_MALS_original(X, r, opts)

    maxIter = opts.maxIter;
    tol = opts.tol;
    lambda = opts.lambda

    [n1, n2, n3] = size(X);
    Xnorm = norm(X(:));
    
    % Initialize factors A, B, C randomly
    A = randn(n1, r, r);
    B = randn(r, n2, r);
    C = randn(r, r, n3);
    
    errHist = zeros(maxIter,1);
    
    for iter = 1:maxIter
        Xhat = triple_product(A, B, C);
        errHist(iter) = norm(X(:) - Xhat(:)) / Xnorm;
        if mod(iter, 10) == 0
        fprintf('Iteration %d, relative error = %.4e\n', iter, errHist(iter));
        end
        if iter > 1 && abs(errHist(iter)-errHist(iter-1)) < tol*errHist(iter-1)
            errHist = errHist(1:iter);
            break;
        end
        
        

        X1 = reshape(X, [n1, n2*n3]);  % size: n1 x (n2*n3)
        F = buildF(B, C);              % size: (r^2) x (n2*n3)
        A_old = unfold(A, 1);
        A_new_unf = (X1 * F' + lambda * A_old) * pinv(F * F' + lambda * eye(r^2));
        A = reshape_A_from_A1(A_new_unf, n1, r);
        

        X2 = reshape(permute(X, [2,1,3]), [n2, n1*n3]);  % size: n2 x (n1*n3)
        G = buildG(A, C);              % size: (r^2) x (n1*n3)
        B_old = unfold(B, 2);           % size: n2 x (r^2)
        B_new_unf = (X2 * G' + lambda * B_old) * pinv(G * G' + lambda * eye(r^2));
        B = reshape_B_from_B2(B_new_unf, n2, r);
        

        X3 = reshape(permute(X, [3,1,2]), [n3, n1*n2]);  % size: n3 x (n1*n2)
        H = buildH(A, B);            
        C_old = unfold(C, 3);
        C_new_unf = (X3 * H' + lambda * C_old) * pinv(H * H' + lambda * eye(r^2));
        C = reshape_C_from_C3(C_new_unf, n3, r);
    end
end


function A = reshape_A_from_A1(A1, n1, r)
% RESHAPE_A_FROM_A1 reshapes an n1 x (r^2) matrix A1 into a tensor A of size (n1 x r x r).
    A = zeros(n1, r, r);
    for i = 1:n1
        A(i,:,:) = reshape(A1(i,:), [r, r]);
    end
end

function B = reshape_B_from_B2(B2, n2, r)
% RESHAPE_B_FROM_B2 reshapes an n2 x (r^2) matrix B2 into a tensor B of size (r x n2 x r).
    B = zeros(r, n2, r);
    for j = 1:n2
        B(:,j,:) = reshape(B2(j,:), [r, r]);
    end
end

function C = reshape_C_from_C3(C3, n3, r)
% RESHAPE_C_FROM_C3 reshapes an n3 x (r^2) matrix C3 into a tensor C of size (r x r x n3).
    C = zeros(r, r, n3);
    for t = 1:n3
        C(:,:,t) = reshape(C3(t,:), [r, r]);
    end
end
