function H = buildH(A, B)
    % A: (n1 x r x r), B: (r x n2 x r)
    [n1, r, ~] = size(A);
    [~, n2, ~] = size(B);
    % H = zeros(r^2, n1*n2);
    % for i = 1:n1
    %     for j = 1:n2
    %         col_idx = i + (j-1)*n1;
    %         for p = 1:r
    %             for q = 1:r
    %                 row_idx = p + (q-1)*r;
    %                 H(row_idx, col_idx) = A(i,p,q) * B(p,j,q);
    %             end
    %         end
    %     end
    % end
    A_unfold = reshape(unfold(A, 1), [n1, r*r, 1]);  
    B_unfold = reshape(unfold(B, 2)', [1, r*r, n2]);
    H = A_unfold .* B_unfold;
    H = reshape(H, [n1, r, r, n2]);
    H = reshape(permute(H, [2, 3, 1, 4]), [r*r, n1*n2]);
end