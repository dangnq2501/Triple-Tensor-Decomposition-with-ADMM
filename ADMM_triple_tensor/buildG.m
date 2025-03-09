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