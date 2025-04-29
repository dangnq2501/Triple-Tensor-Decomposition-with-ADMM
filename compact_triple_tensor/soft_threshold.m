function O = soft_threshold(X, lam)
    O = sign(X) .* max(abs(X) - lam, 0);
end