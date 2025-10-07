function [A, B, C, errHist] = triple_decomp_MALS(X, r, opts)
%   X: tensor (n1 x n2 x n3)
%   r: rank <= mid{n1, n2, n3}
%   maxIter: max Iterator
%   tol: relative change
    maxIter = opts.maxIter;
    tol = opts.tol;
    [n1, n2, n3] = size(X);
    Xf = norm(X(:));    

    A = randn(n1, r, r);
    B = randn(r,  n2, r);
    C = randn(r,  r, n3);
    errHist = zeros(maxIter,1);

    for it = 1:maxIter

        X1 = unfold(X, 1);            % n1 x (n2*n3)
        F  = buildF(B, C);            % (r^2) x (n2*n3)

        FFt = F*F';
        A1  = (X1 * F') * pniv(FFt + 1.0e-9*eye(size(FFt))); 
        Anew = zeros(n1, r, r);
        for i = 1:n1
            rowA1 = A1(i,:);
            Anew(i,:,:) = reshape(rowA1, [r, r]);
        end
        A = Anew;


        GGt = G*G';
        B2  = (X2 * G') * pinv(GGt + 1.0e-9*eye(size(GGt))); 
        Bnew = zeros(r, n2, r);
        for j = 1:n2
            rowB2 = B2(j,:);
            Bnew(:, j, :) = reshape(rowB2, [r, r]);
        end
        B = Bnew;

        HHt = H*H';
        C3  = (X3 * H') * pinv(HHt + 1.0e-9*eye(size(HHt))); 
        Cnew = zeros(r, r, n3);
        for t = 1:n3
            rowC3 = C3(t,:);
            Cnew(:,:,t) = reshape(rowC3, [r, r]);
        end
        C = Cnew;

        Xapprox = triple_product(A,B,C);  % tính A B C
        res = norm(X(:) - Xapprox(:));    % Frobenius
        errHist(it) = res / Xf;           % lỗi tương đối

        % if it>1 && abs(errHist(it)-errHist(it-1))<tol*errHist(it-1)
        %     errHist = errHist(1:it);
        %     break;
        % end
    end

end

