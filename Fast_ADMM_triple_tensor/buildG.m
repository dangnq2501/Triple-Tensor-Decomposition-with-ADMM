function G = buildG(A, C)
    % BUILDG computes G for updating factor B.
    % A is of size (n1 x r x r) with entries A(i,q,s)
    % C is of size (r x r x n3) with entries C(p,q,t)
    % G is of size (r^2) x (n1*n3) and computed as:
    %   G(p+(s-1)*r, i+(t-1)*n1) = \sum_{q=1}^{r} A(i,q,s)*C(p,q,t)
    [n1, r, ~] = size(A);
    [~, ~, n3] = size(C);
    G = reshape(permute(A, [1, 3, 2]), [n1*r, r]) * reshape(permute(C, [2, 1, 3]), [r, r*n3]);
    G = permute(reshape(G, [n1, r, r, n3]), [3, 2, 1, 4]);
    G = reshape(G, [r*r, n1*n3]); 
end