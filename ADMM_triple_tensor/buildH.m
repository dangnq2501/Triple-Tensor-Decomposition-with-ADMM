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