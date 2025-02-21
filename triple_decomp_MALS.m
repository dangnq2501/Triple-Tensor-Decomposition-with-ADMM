function [A, B, C, errHist] = triple_decomp_MALS(X, r, maxIter, tol)
%   X: tensor (n1 x n2 x n3)
%   r: rank <= mid{n1, n2, n3}
%   maxIter: max Iterator
%   tol: relative change

    [n1, n2, n3] = size(X);
    Xf = norm(X(:));    

    A = randn(n1, r, r);
    B = randn(r,  n2, r);
    C = randn(r,  r, n3);
    errHist = zeros(maxIter,1);

    for it = 1:maxIter

        % F = B A
        X1 = unfold(X, 1);            % n1 x (n2*n3)
        F  = buildF(B, C);            % (r^2) x (n2*n3)

        % LS: A(1)*F = X(1)  -> A(1) = X(1)*F' * inv(F*F')
        FFt = F*F';
        A1  = (X1 * F') / (FFt + 1.0e-12*eye(size(FFt))); 
        Anew = zeros(n1, r, r);
        for i = 1:n1
            rowA1 = A1(i,:);
            Anew(i,:,:) = reshape(rowA1, [r, r]);
        end
        A = Anew;

        X2 = unfold(X, 2);        % n2 x (n1*n3)
        G  = buildG(A, C);        % (r^2) x (n1*n3)
        GGt = G*G';
        B2  = (X2 * G') / (GGt + 1.0e-12*eye(size(GGt))); 
        Bnew = zeros(r, n2, r);
        for j = 1:n2
            rowB2 = B2(j,:);
            Bnew(:, j, :) = reshape(rowB2, [r, r]);
        end
        B = Bnew;

        % === Bước 3: Cập nhật C ===
        X3 = unfold(X, 3);        % n3 x (n1*n2)
        H  = buildH(A, B);        % (r^2) x (n1*n2)
        HHt = H*H';
        C3  = (X3 * H') / (HHt + 1.0e-12*eye(size(HHt))); 
        Cnew = zeros(r, r, n3);
        for t = 1:n3
            rowC3 = C3(t,:);
            Cnew(:,:,t) = reshape(rowC3, [r, r]);
        end
        C = Cnew;

        % === Tính sai số sau mỗi vòng lặp ===
        Xapprox = triple_product(A,B,C);  % tính A B C
        res = norm(X(:) - Xapprox(:));    % Frobenius
        errHist(it) = res / Xf;           % lỗi tương đối

        % Điều kiện dừng nếu lỗi thay đổi rất nhỏ
        % if it>1 && abs(errHist(it)-errHist(it-1))<tol*errHist(it-1)
        %     errHist = errHist(1:it);
        %     break;
        % end
    end

end

function Xhat = triple_product(A, B, C)
% Tính tích A B C -> Xhat
%   A: (n1 x r x r), B: (r x n2 x r), C: (r x r x n3)
%   Xhat: (n1 x n2 x n3)

    [n1, r, ~] = size(A);
    [~, n2, ~] = size(B);
    [~, ~, n3] = size(C);

    Xhat = zeros(n1,n2,n3);
    for i=1:n1
        for j=1:n2
            for t=1:n3
                s = 0;
                for p=1:r
                    for q=1:r
                        s = s + A(i,p,q)*B(p,j,q)*C(p,q,t);
                    end
                end
                Xhat(i,j,t) = s;
            end
        end
    end
end


function F = buildF(B, C)
% Xây dựng ma trận F (r^2 x n2*n3) để cập nhật A(1)
%
% Dựa trên công thức trong bài báo, với B la (r x n2 x r), 
% C la (r x r x n3). 
% Ý tưởng: F(k,:) tương ứng với một "vector hoá" trên (B, C).
%
    [r, n2, ~] = size(B);
    [~, ~, n3] = size(C);
    
    % Kết quả F: r^2 x (n2*n3)
    F = zeros(r^2, n2*n3);
    % Gán chỉ số cột = j + (t-1)*n2, chỉ số hàng = q + (s-1)*r
    for j = 1:n2
        for t = 1:n3
            col_idx = j + (t-1)*n2;  % cột
            for q = 1:r
                for s = 1:r
                    row_idx = q + (s-1)*r;  % hàng
                    F(row_idx, col_idx) = B(q,j,s) * ...
                                          C(q,s,t); 
                end
            end
        end
    end
end

function G = buildG(A, C)
% Xây dựng ma trận G (r^2 x n1*n3) để cập nhật B(2)
%
    [n1, r, ~] = size(A);
    [~, ~, n3] = size(C);
    
    G = zeros(r^2, n1*n3);
    for i = 1:n1
        for t = 1:n3
            col_idx = i + (t-1)*n1;
            for p = 1:r
                for s = 1:r
                    row_idx = p + (s-1)*r;
                    G(row_idx, col_idx) = A(i,p,s)* ...
                                          C(p,s,t);
                end
            end
        end
    end
end

function H = buildH(A, B)
% Xây dựng ma trận H (r^2 x n1*n2) để cập nhật C(3)
%
    [n1, r, ~] = size(A);
    [~, n2, ~] = size(B);
    
    H = zeros(r^2, n1*n2);
    for i = 1:n1
        for j = 1:n2
            col_idx = i + (j-1)*n1;
            for p = 1:r
                for q = 1:r
                    row_idx = p + (q-1)*r;
                    H(row_idx, col_idx) = A(i,p,q)* ...
                                          B(p,j,q);
                end
            end
        end
    end
end


