function Xhat = triple_product(A, B, C)
    % Tính triple product: Xhat = A * B * C
    [n1, ~, ~] = size(A);
    [~, n2, ~] = size(B);
    [~, ~, n3] = size(C);
    Xhat = zeros(n1, n2, n3);
    
    for i = 1:n1
        for j = 1:n2
            for t = 1:n3
                s = 0;
                for p = 1:size(A,2)
                    for q = 1:size(A,3)
                        s = s + A(i, p, q) * B(p, j, q) * C(p, q, t);
                    end
                end
                Xhat(i, j, t) = s;
            end
        end
    end
end