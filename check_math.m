r = 2;
n1 = 3;
n2 = 4;
n3 = 5;
A = randn(n1, r, r);
B = randn(r, n2, r);
C = randn(r, r, n3);
X = triple_product(A, B, C);


Y = abs(unfold(X, 1) - unfold(permute(A, [1, 3, 2]), 1) * kronF(B, C));
disp(min(abs(Y(:))));
Y = abs(unfold(X, 2) - unfold(permute(B, [2, 3, 1]), 1) * kronG(A, C));
disp(min(abs(Y(:))));
Y = abs(unfold(X, 3) - unfold(C, 3) * kronH(A, B));
disp(min(abs(Y(:))));
% Y = abs(unfold(X, 2) - unfold(permute(B, [3 2 1]), 2) * kronG(A, C));
% disp(min(Y(:)));
% Y = abs(buildH(A, B) - kronH(A, B));
% disp(min(Y(:)));
function F = kronF(B, C)
    [r, n2, ~] = size(B);
    [~, ~, n3] = size(C);
    B1 = kron(eye(r), unfold(permute(B, [3, 2, 1]), 1));
    C1 = kron(reshape(C, [r*r, n3]), eye(n2));
    F = B1 * C1;
    % B1 = kron(eye(n3), reshape(permute(B, [1, 3, 2]), [r*r, n2]));
    % C1 = kron(unfold(permute(C, [1 3 2]), 1), eye(r));
    % F = C1 * B1;
end

function G = kronG(A, C)
    [n1, r, ~] = size(A);
    [~, ~, n3] = size(C);
    A1 = kron(eye(r), unfold(A, 3));
    C1 = kron(unfold(permute(C, [2, 1, 3]), 3)', eye(n1));
    G = A1 * C1;
end

function H = kronH(A, B)
    [n1, r, ~] = size(A);
    [~, n2, ~] = size(B);
    A1 = kron(unfold(permute(A, [1, 3, 2]), 1)', eye(n2));
    B1 = kron(eye(r), unfold(B, 1));
    H = B1 * A1;
end

function Xhat = triple_product(A, B, C)
% TRIPLE_PRODUCT computes the product:
%   Xhat(i,j,t) = \sum_{p=1}^{r} \sum_{q=1}^{r} \sum_{s=1}^{r} A(i,q,s) * B(p,j,s) * C(p,q,t)
    [n1, ~, ~] = size(A);
    [~, n2, ~] = size(B);
    [~, ~, n3] = size(C);
    r = size(A,2);
    Xhat = zeros(n1, n2, n3);
    for i = 1:n1
        for j = 1:n2
            for t = 1:n3
                total = 0;
                for p = 1:r
                    for q = 1:r
                        for s = 1:r
                            total = total + A(i,q,s) * B(p,j,s) * C(p,q,t);
                        end
                    end
                end
                Xhat(i,j,t) = total;
            end
        end
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
                    temp = 0;
                    for p = 1:r
                        temp = temp + B(p,j,s)*C(p,q,t);
                    end
                    F(row_idx, col_idx) = temp;
                end
            end
        end
    end
end

function G = buildG(A, C)
    % BUILDG computes G for updating factor B.
    % A is of size (n1 x r x r) with entries A(i,q,s)
    % C is of size (r x r x n3) with entries C(p,q,t)
    % G is of size (r^2) x (n1*n3) and computed as:
    %   G(p+(s-1)*r, i+(t-1)*n1) = \sum_{q=1}^{r} A(i,q,s)*C(p,q,t)
    [n1, r, ~] = size(A);
    [~, ~, n3] = size(C);
    G = zeros(r^2, n1*n3);
    for i = 1:n1
        for t = 1:n3
            col_idx = i + (t-1)*n1;
            for p = 1:r
                for s = 1:r
                    row_idx = p + (s-1)*r;
                    temp = 0;
                    for q = 1:r
                        temp = temp + A(i,q,s)*C(p,q,t);
                    end
                    G(row_idx, col_idx) = temp;
                end
            end
        end
    end
end

function H = buildH(A, B)
    % BUILDH computes H for updating factor C.
    % A is of size (n1 x r x r) with entries A(i,q,s)
    % B is of size (r x n2 x r) with entries B(p,j,s)
    % H is of size (r^2) x (n1*n2) and computed as:
    %   H(p+(q-1)*r, i+(j-1)*n1) = \sum_{s=1}^{r} A(i,q,s)*B(p,j,s)
    [n1, r, ~] = size(A);
    [~, n2, ~] = size(B);
    H = zeros(r^2, n1*n2);
    for i = 1:n1
        for j = 1:n2
            col_idx = i + (j-1)*n1;
            for p = 1:r
                for q = 1:r
                    row_idx = p + (q-1)*r;
                    temp = 0;
                    for s = 1:r
                        temp = temp + A(i,q,s)*B(p,j,s);
                    end
                    H(row_idx, col_idx) = temp;
                end
            end
        end
    end
end

r = 5;
n1 = 101;
n2 = 102;
n3 = 103;
A = randn(n1, r, r)*3;
B = randn(r, n2, r)*5;
C = randn(r, r, n3)*2;
X = triple_product(A, B, C);
tic;
F = reshape(permute(B, [2, 3, 1]), [n2*r, r]) * reshape(C, [r, r*n3]);
F = permute(reshape(F, [n2, r, r, n3]), [3, 2, 1, 4]);
F = reshape(F, [r*r, n2*n3]);
Y = abs(unfold(X, 1) - unfold(A, 1) * F);
timer = toc;
fprintf("Loss: %.6f Run time F1 %.4f\n", min(Y(:)), timer);

tic;
Y = abs(unfold(permute(X, [1, 3, 2]), 1) - unfold(permute(A, [1, 3, 2]), 1) * kronF(B, C));
timer = toc;
fprintf("Loss: %.6f Run time F2 %.4f\n", min(Y(:)), timer);

tic;
G = reshape(permute(A, [1, 3, 2]), [n1*r, r]) * reshape(permute(C, [2, 1, 3]), [r, r*n3]);
G = permute(reshape(G, [n1, r, r, n3]), [3, 2, 1, 4]);
G = reshape(G, [r*r, n1*n3]); 
Y = abs(unfold(X, 2) - unfold(B, 2) * G);
timer = toc;
fprintf("Loss: %.6f Run time G1 %.4f\n", min(Y(:)), timer);

tic;
Y = abs(unfold(permute(X,[2, 3, 1]), 1) - unfold(permute(B, [2, 3, 1]), 1) * kronG(A, C));
timer = toc;
fprintf("Loss: %.6f Run time G2 %.4f\n", min(Y(:)), timer);

tic;
H = reshape(A, [n1*r, r]) * reshape(permute(B, [3, 1, 2]), [r, r*n2]);
H = permute(reshape(H, [n1, r, r, n2]), [3, 2, 1, 4]);
H = reshape(H, [r*r, n1*n2]); 
Y = abs(unfold(X, 3) - unfold(C, 3) * H);
timer = toc;
fprintf("Loss: %.6f Run time H1 %.4f\n", min(Y(:)), timer);

tic;
Y = abs(unfold(X, 3) - unfold(C, 3) * kronH(A, B));
timer = toc;
fprintf("Loss: %.6f Run time H2 %.4f\n", min(Y(:)), timer);
% Y = abs(buildH(A, B) - kronH(A, B));
% disp(min(Y(:)));
function F = kronF(B, C)
    [r, n2, ~] = size(B);
    [~, ~, n3] = size(C);
    B1 = kron(eye(r), unfold(permute(B, [3, 2, 1]), 1));
    C1 = kron(reshape(C, [r*r, n3]), eye(n2));
    F = B1 * C1;
end

function G = kronG(A, C)
    [n1, r, ~] = size(A);
    [~, ~, n3] = size(C);
    A1 = kron(eye(r), unfold(A, 3));
    C1 = kron(unfold(permute(C, [2, 1, 3]), 3)', eye(n1));
    G = A1 * C1;

end

function H = kronH(A, B)
    [n1, r, ~] = size(A);
    [~, n2, ~] = size(B);
    A1 = kron(unfold(permute(A, [1, 3, 2]), 1)', eye(n2));
    B1 = kron(eye(r), unfold(B, 1));
    H = B1 * A1;
end

function Xhat = triple_product(A, B, C)
% TRIPLE_PRODUCT computes the product:
%   Xhat(i,j,t) = \sum_{p=1}^{r} \sum_{q=1}^{r} \sum_{s=1}^{r} A(i,q,s) * B(p,j,s) * C(p,q,t)
    [n1, ~, ~] = size(A);
    [~, n2, ~] = size(B);
    [~, ~, n3] = size(C);
    r = size(A,2);
    Xhat = zeros(n1, n2, n3);
    for i = 1:n1
        for j = 1:n2
            for t = 1:n3
                total = 0;
                for p = 1:r
                    for q = 1:r
                        for s = 1:r
                            total = total + A(i,q,s) * B(p,j,s) * C(p,q,t);
                        end
                    end
                end
                Xhat(i,j,t) = total;
            end
        end
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
                    temp = 0;
                    for p = 1:r
                        temp = temp + B(p,j,s)*C(p,q,t);
                    end
                    F(row_idx, col_idx) = temp;
                end
            end
        end
    end
end

function G = buildG(A, C)
    % BUILDG computes G for updating factor B.
    % A is of size (n1 x r x r) with entries A(i,q,s)
    % C is of size (r x r x n3) with entries C(p,q,t)
    % G is of size (r^2) x (n1*n3) and computed as:
    %   G(p+(s-1)*r, i+(t-1)*n1) = \sum_{q=1}^{r} A(i,q,s)*C(p,q,t)
    [n1, r, ~] = size(A);
    [~, ~, n3] = size(C);
    G = zeros(r^2, n1*n3);
    for i = 1:n1
        for t = 1:n3
            col_idx = i + (t-1)*n1;
            for p = 1:r
                for s = 1:r
                    row_idx = p + (s-1)*r;
                    temp = 0;
                    for q = 1:r
                        temp = temp + A(i,q,s)*C(p,q,t);
                    end
                    G(row_idx, col_idx) = temp;
                end
            end
        end
    end
end

function H = buildH(A, B)
    % BUILDH computes H for updating factor C.
    % A is of size (n1 x r x r) with entries A(i,q,s)
    % B is of size (r x n2 x r) with entries B(p,j,s)
    % H is of size (r^2) x (n1*n2) and computed as:
    %   H(p+(q-1)*r, i+(j-1)*n1) = \sum_{s=1}^{r} A(i,q,s)*B(p,j,s)
    [n1, r, ~] = size(A);
    [~, n2, ~] = size(B);
    H = zeros(r^2, n1*n2);
    for i = 1:n1
        for j = 1:n2
            col_idx = i + (j-1)*n1;
            for p = 1:r
                for q = 1:r
                    row_idx = p + (q-1)*r;
                    temp = 0;
                    for s = 1:r
                        temp = temp + A(i,q,s)*B(p,j,s);
                    end
                    H(row_idx, col_idx) = temp;
                end
            end
        end
    end
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