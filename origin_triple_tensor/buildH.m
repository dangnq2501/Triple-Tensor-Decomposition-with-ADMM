function H = buildH(A, B)
    % BUILDH computes H for updating factor C.
    % A is of size (n1 x r x r) with entries A(i,q,s)
    % B is of size (r x n2 x r) with entries B(p,j,s)
    % H is of size (r^2) x (n1*n2) and computed as:
    %   H(p+(q-1)*r, i+(j-1)*n1) = \sum_{s=1}^{r} A(i,q,s)*B(p,j,s)
    [n1, r, ~] = size(A);
    [~, n2, ~] = size(B);
    H = reshape(A, [n1*r, r]) * reshape(permute(B, [3, 1, 2]), [r, r*n2]);
    H = permute(reshape(H, [n1, r, r, n2]), [3, 2, 1, 4]);
    H = reshape(H, [r*r, n1*n2]); 
end