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