clc;
clear;
close all;
addpath(genpath(cd));
rng(42); % Reproducibility
% missing_rates = [0.05, 0.1, 0.15];
missing_rates = [0.1];
datasets = ["taxi", "sensor", "network","chicago"];
subdims = [10, 6, 16, 8];
%% Load initial data
for missing_id = 1:length(missing_rates)
    missing_ratio = missing_rates(missing_id);
    for i = 1:length(datasets)
        name = datasets(i);
        load(name + ".mat");  % Load variable X or T
        if name == "sensor"
            X = T;
        end
        X = double(X);
        if name == "taxi"
            X = X(:, :, 1:500);
        end
         if max(X(:))>1
            X = X/max(X(:));
        end
        [I J K] = size(X);
        X = reshape(X, [I, J, subdims(i), K/subdims(i)]);
        %% Step 1: Generate Missing Mask
        Ysz = size(X);
        total_elements = numel(X);
        num_to_zero = round(missing_ratio * total_elements);
        zero_indices = randperm(total_elements, num_to_zero);
        mask_missing = false(Ysz);
        mask_missing(zero_indices) = true;
        
        Omega = (~mask_missing);   % Observed mask
        Y = X;
        Y(mask_missing) = 0;
        gt = X(mask_missing);
        gt_2 = X(~mask_missing);
    
        fprintf('=== The sample ratio is %4.2f ===\n', missing_ratio);
        Ndim      = ndims(X);
        Nway      = size(X);
    
        
        %% Perform  algorithms
        % initialization of the parameters
        % Please refer to our paper to set the parameters
        opts=[];
        opts.max_R = [0,  6,  6,  6;
                      0,  0,  6,  6;
                      0,  0,  0,  6;
                      0,  0,  0,  0];  
        opts.R     = [0,  2,  2,  2;
                      0,  0,  2,  2;
                      0,  0,  0,  2;
                      0,  0,  0,  0];
        %     R     = [0,  R_{1,2}, R_{1,3}, ..., R_{1,N};
        %              0,     0,    R_{2,3}, ..., R_{2,N};
        %              ...
        %              0,     0,       0,    ..., R_{N-1,N};
        %              0,     0,       0,    ...,   0     ];
        opts.tol   = 1e-5;
        opts.maxit = 100;
        opts.rho   = 5;
        opts.origin = X;
        %%%%%
        fprintf('\n');
        t0= tic;
        % Please see the difference between 'inc_FCTN_TC' and 'inc_FCTN_TC_end' in README.txt. 
        %[Re_tensor{i},G,Out]        = inc_FCTN_TC(F,Omega,opts);
        [Re_tensor,G,Out]        = inc_FCTN_TC(Y,Omega,opts);
        time                     = toc(t0);
        [psnr, ssim]          = quality_ybz(X, Re_tensor);
        [rmse_fctn, nrmse_fctn] =  evaluate(Re_tensor, X, true(size(X)));
        errHist = Out.RSE;
        method = "FCTN";
        if missing_ratio == 0.1
            save(sprintf("%s_%s_errHist.mat", name, method), 'errHist');
        end
        %% Show result
        fprintf('\n');
        fprintf('================== Result =====================\n');
        fprintf("RE: %.4e - PSN: %.4e - SSIM: %.4e - Timer: %.2f\n", nrmse_fctn, psnr, ssim, time);
        % fprintf(' %8.8s    %5.4s    %5.4s    \n','method','PSNR', 'SSIM' );
        %     fprintf(' %8.8s    %5.3f    %5.3f    \n',...
        %     methodname{enList(i)},psnr(enList(i)), ssim(enList(i)));
        % 
        % fprintf('================== Result =====================\n');
        % figure,
        % show_figResult(Re_tensor,T,min(T(:)),max(T(:)),methodname,enList,1,prod(Nway(3:end)))
    end
end

function [rmse, nrmse] = evaluate(X, gt, mask)
    X_hat = double(X);
    X_hat = X_hat(mask);
    
    rmse = norm(X_hat(:)-gt(:));
    nrmse = rmse / norm(gt(:));
    clear X_hat;
    
end