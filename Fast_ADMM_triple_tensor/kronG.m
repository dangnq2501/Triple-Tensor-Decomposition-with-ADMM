function G = kronG(A, C)
    [n1, r, ~] = size(A);
    [~, ~, n3] = size(C);
    A1 = kron(eye(r), unfold(A, 3));
    C1 = kron(unfold(permute(C, [2, 1, 3]), 3)', eye(n1));
    G = A1 * C1;
end
