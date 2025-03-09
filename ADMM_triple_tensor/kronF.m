function F = kronF(B, C)
    [r, n2, ~] = size(B);
    [~, ~, n3] = size(C);
    B1 = kron(eye(r), unfold(permute(B, [3, 2, 1]), 1));
    C1 = kron(reshape(C, [r*r, n3]), eye(n2));
    F = B1 * C1;
    % B1 = kron(eye(n3), reshape(permute(B, [1, 3, 2]), [r*r, n2]));
    % C1 = kron(unfold(permute(C, [1 3 2]), 1), eye(r));
    % F = C1 * B1;
end