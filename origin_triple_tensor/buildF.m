function F = buildF(B, C)
    [r, n2, ~] = size(B);
    [~, ~, n3] = size(C);
    F = reshape(permute(B, [2, 3, 1]), [n2*r, r]) * reshape(C, [r, r*n3]);
    F = permute(reshape(F, [n2, r, r, n3]), [3, 2, 1, 4]);
    F = reshape(F, [r*r, n2*n3]);
end