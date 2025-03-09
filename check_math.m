r = 2;
n1 = 3;
n2 = 4;
n3 = 5;
A = randn(n1, r, r);
B = randn(r, n2, r);
C = randn(r, r, n3);
X = triple_product(A, B, C);
Fkron = kronF(B, C);
F = buildF(B, C);
A1 = unfold(permute(A, [1, 3, 2]), 1);

Y = abs(unfold(X, 1) - unfold(A, 1) * buildF(B, C));
disp(min(Y(:)));
Y = abs(unfold(X, 2) - unfold(B, 2) * buildG(A, C));
disp(min(Y(:)));
Y = abs(unfold(X, 3) - unfold(C, 3) * buildH(A, B));
disp(min(Y(:)));
% Y = abs(unfold(X, 2) - unfold(permute(B, [3 2 1]), 2) * kronG(A, C));
% disp(min(Y(:)));
% Y = abs(buildH(A, B) - kronH(A, B));
% disp(min(Y(:)));
function F = kronF(B, C)
    [r, n2, ~] = size(B);
    [~, ~, n3] = size(C);
    % B1 = kron(eye(r), unfold(permute(B, [2, 1, 3]), 3));
    % C1 = kron(unfold(C, 3)', eye(n2));
    % F = B1 * C1;
    B1 = kron(unfold(B, 2)', eye(n3));
    C1 = kron(unfold(permute(C, [1 3 2]), 1), eye(r));
    F = C1 * B1;
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
    % BUILDF computes F from factors B and C as defined in the paper.
    % B is of size (r x n2 x r) with entries B(p,j,s)
    % C is of size (r x r x n3) with entries C(p,q,t)
    % F is of size (r^2) x (n2*n3) and computed as:
    %   F(q+(s-1)*r, j+(t-1)*n2) = \sum_{p=1}^{r} B(p,j,s)*C(p,q,t)
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