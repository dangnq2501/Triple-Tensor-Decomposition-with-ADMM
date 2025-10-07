function G = buildG(A, C)
    % A: (n1 x r x r), C: (r x r x n3)
    [n1, r, ~] = size(A);
    [~, ~, n3] = size(C);
    % G = zeros(r^2, n1*n3);
    % for i = 1:n1
    %     for t = 1:n3
    %         col_idx = i + (t-1)*n1;
    %         for p = 1:r
    %             for s = 1:r
    %                 row_idx = p + (s-1)*r;
    %                 G(row_idx, col_idx) = A(i,p,s) * C(p,s,t);
    %             end
    %         end
    %     end
    % end
    A_unfold = reshape(unfold(A, 1), [n1, r*r, 1]);  
    C_unfold = reshape(unfold(C, 3)', [1, r*r, n3]);
    G = A_unfold .* C_unfold;
    G = reshape(G, [n1, r, r, n3]);
    G = reshape(permute(G, [2, 3, 1, 4]), [r*r, n1*n3]);
end