function Xn = unfold(X, mode)
    % Chuyển tensor X về dạng unfold (matrix)
    [n1, n2, n3] = size(X);
    switch mode
        case 1
            Xn = reshape(X, [n1, n2 * n3]);
        case 2
            Xn = reshape(permute(X, [2, 1, 3]), [n2, n1 * n3]);
        case 3
            Xn = reshape(permute(X, [3, 1, 2]), [n3, n1 * n2]);
        otherwise
            error('Mode must be 1, 2, or 3.');
    end
end