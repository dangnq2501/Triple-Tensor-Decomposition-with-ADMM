
% load('X_Hall.mat');
% X = X_video(:, :, 1:200);
load('office.mat');
X = gray_images(:, :, 1:200);
name = "office";
file_name = name+"_time_consuming.txt";
file_id = fopen(file_name, "a");

disp(size(X));
r = 15;
maxIter = 30;
tol = 1e-5;
figure;

tic;
[A,B,C,O,errHist_ALS] = triple_decomp_ADMM_ALS(X, r, maxIter, tol, 1);
time_ALS = toc;
result_ADMM_reg = triple_product(A,B,C);
total = result_ADMM_reg+O;
video_name = name + "_ADMM_ABC_ALS.mat";
save(video_name, 'result_ADMM_reg');
video_name = name + "_ADMM_O_ALS.mat";
save(video_name, 'O');
video_name = name + "_ADMM_ABCO_ALS.mat";
save(video_name, 'total')
plot(errHist_ALS, '^-','LineWidth',1.5, 'DisplayName','ADMM_ALS');
fprintf(file_id, '%s ADMM time consuming: %.3f s\n', name, time_ALS);
hold on;

tic;
[A,B,C,O,errHist_MALS] = triple_decomp_ADMM_MALS(X, r, maxIter, tol, 1);
time_MALS = toc;
result_ADMM_reg = triple_product(A,B,C);
total = result_ADMM_reg+O;
video_name = name+'_ADMM_ABC_MALS.mat';
save(video_name, 'result_ADMM_reg');
video_name = name+'_ADMM_O_MALS.mat';
save(video_name, 'O');
video_name = name+'_ADMM_ABCO_MALS.mat';
save(video_name, 'total')
plot(errHist_MALS, 'o-','LineWidth',1.5, 'DisplayName','ADMM_MALS');
fprintf(file_id, '%s ADMM MALS time consuming: %.3f s\n' ,name, time_MALS);

% hold on;

tic;
[A,B,C,O,errHist_ADMM] = triple_decomp_ADMM_noise(X, r, 1, 0.01, maxIter, tol);
time_ADMM = toc;
result_ADMM_reg = triple_product(A,B,C);
total = result_ADMM_reg+O;
video_name = name + 'faster_ADMM_ABC.mat';
save(video_name, 'result_ADMM_reg');
video_name = name + 'faster_ADMM_O.mat';
save(video_name, 'O');
video_name = name + 'faster_ADMM_ABCO.mat';
save(video_name, 'total');
plot(errHist_ADMM, '^-','LineWidth',1.5, 'DisplayName','ADMM');
fprintf(file_id, '%s FASTER ADMM time consuming: %.3f s\n', name, time_ADMM);

% hold on;
tic;
[A,B,C, O,errHist_origin] =  triple_decomp_ADMM_origin_noise(X, r, maxIter, tol, 3, 0.01, 3, maxIter, tol);
time_origin = toc;
result_ADMM_reg = triple_product_origin(A,B,C);
fprintf(file_id, "%s ADMM origin timer: %.3f s\n", name, time_origin);
total = result_ADMM_reg+O;
video_name = name + '_ADMM_ABC_origin.mat';
save(video_name, 'result_ADMM_reg');
video_name = name + '_ADMM_O_origin.mat';
save(video_name, 'O');
video_name = name + '_ADMM_ABCO_origin.mat';
save(video_name', 'total');
plot(errHist_origin, 's-','LineWidth',1.5,'DisplayName','ADMM origin');


xlabel('Iteration'); ylabel('Relative Error');
title('Triple Decomposition  - Strong noise');

legend('show');
grid on;
saveas(gcf, 'err_plot.png');

function Xhat = triple_product_origin(A, B, C)

    [n1, ~, ~] = size(A);
    [~, n2, ~] = size(B);
    [~, ~, n3] = size(C);
    r = size(A,2);
    Xhat = zeros(n1, n2, n3);
    for i = 1:n1
        for j = 1:n2
            for t = 1:n3
                total = 0;
                for p = 1:r
                    for q = 1:r
                        for s = 1:r
                            total = total + A(i,q,s) * B(p,j,s) * C(p,q,t);
                        end
                    end
                end
                Xhat(i,j,t) = total;
            end
        end
    end
end

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
