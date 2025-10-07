function Xhat = triple_product(A, B, C)
    % Tính triple product: Xhat = A * B * C
    [n1, ~, ~] = size(A);
    [~, n2, ~] = size(B);
    [~, ~, n3] = size(C);
    Xhat = unfold(A, 1) * buildF(B, C);
    Xhat = reshape(Xhat, [n1, n2, n3]);
end