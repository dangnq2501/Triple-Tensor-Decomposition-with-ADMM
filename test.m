
r = 20;
n1 = 100; n2 = 80; n3 = 60;
A = randn(n1, r, r);
B = randn(r, n2, r);
C = randn(r, r, n3);
X = triple_product(A, B, C)+randn(n1, n2, n3);
maxIter = 60;
tol = 1e-5;

[~,~,~,errHist_ALS] = triple_decomp_ADMM(X, r, 0.01, 0.1, 1, maxIter, tol, maxIter, tol);
figure;
plot(errHist_ALS, 'o-','LineWidth',1.5);
xlabel('Iteration'); ylabel('Relative Error');
title('Triple Decomposition (ADMM) - Noise');
grid on;

% [~,~,~,errHist_MALS] = triple_decomp_ALS(X, r, maxIter, tol);
% figure;
% plot(errHist_MALS, 'o-','LineWidth',1.5);
% xlabel('Iteration'); ylabel('Relative Error');
% title('Triple Decomposition (MALS) - Without noise');
% grid on;
% 
% [~,~,~,errHist_ALS] = triple_decomp_MALS(X, r, maxIter, tol);
% figure;
% plot(errHist_ALS, 'o-','LineWidth',1.5);
% xlabel('Iteration'); ylabel('Relative Error');
% title('Triple Decomposition (ALS) - Without noise');
% grid on;

function Xhat = triple_product(A, B, C)
    [n1, ~, ~] = size(A);
    [~, n2, ~] = size(B);
    [~, ~, n3] = size(C);
    Xhat = zeros(n1, n2, n3);
    for i = 1:n1
        for j = 1:n2
            for t = 1:n3
                s = 0;
                for p = 1:size(A,2)
                    for q = 1:size(A,3)
                        s = s + A(i,p,q) * B(p,j,q) * C(p,q,t);
                    end
                end
                Xhat(i,j,t) = s;
            end
        end
    end
end
