function Xhat = triple_product(A, B, C)
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
                            total = total + a(i,q,s) * b(p,j,s) * c(p,q,t);
                        end
                    end
                end
                xhat(i,j,t) = total;
            end
        end
    end
end