function F = buildF(B, C)
    % B: (r x n2 x r), C: (r x r x n3)
    [r, n2, ~] = size(B);
    [~, ~, n3] = size(C);
    % F = zeros(r^2, n2*n3);
    % for j = 1:n2
    %     for t = 1:n3
    %         col_idx = j + (t-1)*n2;
    %         for q = 1:r
    %             for s = 1:r
    %                 row_idx = q + (s-1)*r;
    %                 F(row_idx, col_idx) = B(q,j,s) * C(q,s,t);
    %             end
    %         end
    %     end
    % end
    B_unfold = reshape(unfold(B, 2), [n2, r*r, 1]);  
    C_unfold = reshape(unfold(C, 3)', [1, r*r, n3]);
    F = B_unfold .* C_unfold;
    F = reshape(F, [n2, r, r, n3]);
    F = reshape(permute(F, [2, 3, 1, 4]), [r*r, n2*n3]);
end