function S = soft_threshold(X, tau)
    S = sign(X) .* max(abs(X) - tau, 0);
end