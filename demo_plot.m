
% r = 20;
% n1 = 100; n2 = 80; n3 = 60;
% A = randn(n1, r, r);
% B = randn(r, n2, r);
% C = randn(r, r, n3);
% X_noise = randn(n1, n2, n3);
% X = triple_product(A, B, C)+X_noise;

load('X_Hall.mat');
X = X_video(:, :, 1:200);
r = 10;
maxIter = 50;
tol = 1e-5;
figure;
tic;
[A,B,C,errHist_MALS] = triple_decomp_MALS(X, r, maxIter, tol);
time_MALS = toc;
fprintf('MALS time consuming: %.4f s\n', time_MALS);
result_MALS = triple_product(A,B,C);
plot(errHist_MALS, 's-','LineWidth',1.5,'DisplayName','MALS');
hold on;
tic;
save('video_MALS.mat', 'result_MALS');
[A,B,C,errHist_ADMM] = triple_decomp_ADMM(X, r, 0.01, 0.1, 1, maxIter, tol, maxIter, tol);
time_ADMM = toc;
result_ADMM = triple_product(A,B,C);
plot(errHist_ADMM, '^-','LineWidth',1.5, 'DisplayName','ADMM');
fprintf('ADMM time consuming: %.4f s\n', time_ADMM);

% hold on;
% xlabel('Iteration'); ylabel('Relative Error');
% title('Triple Decomposition  - Strong noise');
save('video_ADMM.mat', 'result_ADMM');

% figure;

% xlabel('Iteration'); ylabel('Relative Error');
% title('Triple Decomposition  - Strong noise');
% grid on;

% figure;
tic;
[A,B,C,errHist_ALS] = triple_decomp_ALS(X, r, maxIter, tol);
time_ALS = toc;
result_ALS = triple_product(A,B,C);
plot(errHist_ALS, 'o-','LineWidth',1.5,'DisplayName', 'ALS');
xlabel('Iteration'); ylabel('Relative Error');
title('Triple Decomposition  - Strong noise');
save('video_ALS.mat', 'result_ALS');
fprintf('ALS time consuming: %.4f s\n', time_ALS);

legend('show');
grid on;



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
