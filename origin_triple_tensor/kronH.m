function H = kronH(A, B)
    [n1, r, ~] = size(A);
    [~, n2, ~] = size(B);
    A1 = kron(unfold(permute(A, [1, 3, 2]), 1)', eye(n2));
    B1 = kron(eye(r), unfold(B, 1));
    H = B1 * A1;
end