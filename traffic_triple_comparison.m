clc; clear;
close all;
addpath(genpath(pwd));
rng(0); % Reproducibility
plot_rate = 0.15;
% missing_rates = [0.05, 0.1, 0.2];
missing_rates = [plot_rate];
% datasets = ["sensor", "network", "taxi","chicago"];
datasets = ["sensor", "chicago"];

% datasets = ["taxi", "sensor"];
subdims = [6, 16, 10, 8];
FCTN_2 = 1;
TRIPLE = 1;
TTNN = 1;
RING = 1;
FCTN = 0;
SOFIA = 1;
r = 5;
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
        %% Step 1: Generate Missing Mask
        Ysz = size(X);
        total_elements = numel(X);
        num_to_zero = round(missing_ratio * total_elements);
        zero_indices = randperm(total_elements, num_to_zero);
        mask_missing = false(Ysz);
        mask_missing(zero_indices) = true;
        fprintf('\n===== Dataset: %s Missing Radio %.3f=====\n', name, missing_ratio);
        Y = X;
        Y(mask_missing) = 0;
        % Y(mask_missing) = max(X(:))*3;
        gt = X(mask_missing);
        gt_2 = X(~mask_missing);
        o_mask= true(size(X));
        %% Step 2: Run ADMM-based Triple Decomposition
        if TRIPLE
            opts.maxIter = 100;
            opts.tol = 1e-5;
            opts.mu = 1e-3;
            opts.lambda = 1.8;
            opts.lambda2 = 1e-3;
            opts.rho = 1.25;
            opts.alphaA = 1e-3;
            opts.alphaB = 1e-3;
            opts.disp = 1;
            opts.origin = X;
            tic;
            % [A, B, C, O, E, Out] = triple_ADMM_masked(Y,  ~mask_missing, r, opts);
            % errHist = Out.errHist;
            [A, B, C, O, errHist] = triple_decomp_ADMM(Y, r, opts);
            
            method = "triple_re";
            if missing_ratio == plot_rate
                save(sprintf("%s_%s_errHist.mat", name, method), 'errHist');
            end
            timer_re = toc;
            X_hat_re = triple_product(A, B, C);
            [rmse_re, nrmse_re] = evaluate(X_hat_re, gt, mask_missing);
            [rmse_re_2, nrmse_re_2] = evaluate(O, gt_2, ~mask_missing);
            % [rmse_re_3, nrmse_re_3] = evaluate(X_hat_re+O, Y, true(size(X)));
            [rmse_re_3, nrmse_re_3] = evaluate(X_hat_re, X, true(size(X)));
            [psnr_re, ssim_re]          = quality_ybz(X, X_hat_re);
            clear X_hat_re;
            fprintf('TRIPLE ADMM - RMSE: %.4e, NRMSE: %.4e, SRMSE: %.4e, SNRMSE: %.4e, TRMSE: %.4e, TNRMSE: %.4e, PSNR: %.4e, SSIM: %.4e, Time: %.2f s\n', rmse_re, nrmse_re, rmse_re_2, nrmse_re_2, rmse_re_3, nrmse_re_3, psnr_re, ssim_re, timer_re);

        end
        
        if SOFIA
    
            Omega = ones(size(Y));
            Omega(mask_missing) = 0;
       
            Omega   = logical(double((Omega)));
            N       = ndims(Y);
            ntimes  = Ysz(N);
            colons  = repmat({':'}, 1, N-1);
            
            m = 168; 
            if name == "taxi"
                m = 7;
            end
            if name == "sensor"
                m = 144;
            end
            cycles = 3;        
            ti = size(X, 3);
            Y_init = Y(colons{:},1:ti);
            Omega_init = Omega(colons{:},1:ti);
            lambda1    = 0.1;
            lambda2    = 0.001;
            lambda3    = 10;
            tol        = 1e-5;
            t_begin_init = tic();
            R = 3;
            maxEpoch = 100;
            needOutlier = true;
            [U_init,X_hat_init,O_init,errHist, ~] = sofia_init(Y_init,Omega_init,R,m,lambda1,lambda2,lambda3,maxEpoch,tol, X);
            timer_sofia = toc(t_begin_init);
            method = "sofia";
           
            X_hat_init = double(X_hat_init);
            O_init = double(O_init);
            [rmse_sofia, nrmse_sofia] = evaluate(X_hat_init, gt, mask_missing);
            [rmse_sofia_2, nrmse_sofia_2] = evaluate(O_init, gt_2, ~mask_missing);
            % [rmse_sofia_3, nrmse_sofia_3] = evaluate(X_hat_init+O_init, Y, true(size(X)));
            [rmse_sofia_3, nrmse_sofia_3] = evaluate(X_hat_init, X, true(size(X)));
            [psnr_sofia, ssim_sofia]          = quality_ybz(X, X_hat_init);
            % tensor2video(X_hat_init, sprintf("%s_%_Xhat.avi", name, method));
            % tensor2video(O_init, sprintf("%s_%s_O.avi", name, method));
            save(sprintf("%s_%s_errHist.mat", name, method),"errHist");
            fprintf('SOFIA - RMSE: %.2e, RRE: %.2e, SRMSE: %.2e, SRRE: %.2e, TRMSE: %.2e, TRRE: %.2e, PSNR: %.4e, SSIM: %.4e, Time: %.2f s\n', rmse_sofia, nrmse_sofia, rmse_sofia_2, nrmse_sofia_2, rmse_sofia_3, nrmse_sofia_3, psnr_sofia, ssim_sofia, timer_sofia);

            clear X_hat_init;
            clear O_init;
        end
        if TTNN
                 Nway = size(X);     
                 N = numel(Nway);                
                  [I1, J1, K1] = size(X);
                 lambda = 50;
                 f = 5;
                 gamma = 0.001;
                 deta = 0.002;
                tic;
                 [Z, S, iter, relerr, errHist] = TT_TRPCA(Y, lambda, f, gamma, deta, X);
                 timer_ttnn = toc;
                     method = "ttnn";
                [rmse_ttnn, nrmse_ttnn] = evaluate(Z, gt, mask_missing);
                [rmse_ttnn_2, nrmse_ttnn_2] = evaluate(S, gt_2, ~mask_missing);
                [rmse_ttnn_3, nrmse_ttnn_3] = evaluate(Z, X, true(size(X)));
                [psnr_ttnn, ssim_ttnn]          = quality_ybz(X, Z);
                if missing_ratio == plot_rate
                    % len = length(errHist);
                    % if len < 10
                    %     disp(errHist);
                    % else
                    %     disp(errHist(len-10:len));
                    % end
                    save(sprintf("%s_%s_errHist.mat", name, method), 'errHist');
                end
            clear Z;
            clear S;
            fprintf('TTNN - RMSE: %.4e, RRE: %.4e, SRMSE: %.4e, SRRE: %.4e, TRMSE: %.4e, TRRE: %.4e, PSNR: %.4e, SSIM: %.4e, Time: %.2f s\n', rmse_ttnn, nrmse_ttnn, rmse_ttnn_2, nrmse_ttnn_2, rmse_ttnn_3, nrmse_ttnn_3, psnr_ttnn, ssim_ttnn, timer_ttnn);

        end

        if RING
            % ring_mask = true(size(Y));
            ring_mask = ~mask_missing;
            [X_hat_ring ,O_ring,RC,timer_ring,errHist]=RTRC(Y,ring_mask,1e-1,false, X);
            method = "ring";
            if missing_ratio == plot_rate
                save(sprintf("%s_%s_errHist.mat", name, method), 'errHist');
            end
            [rmse_ring, nrmse_ring] = evaluate(X_hat_ring, gt, mask_missing);
            [rmse_ring_2, nrmse_ring_2] = evaluate(O_ring, gt_2, ~mask_missing);
            % [rmse_ring_3, nrmse_ring_3] = evaluate(X_hat_ring+O_ring, Y, true(size(X)));
            [rmse_ring_3, nrmse_ring_3] = evaluate(X_hat_ring, X, true(size(X)));
            [psnr_ring, ssim_ring]          = quality_ybz(X, X_hat_ring);
            clear O_ring;
            clear X_hat_ring;
            fprintf('Ring - RMSE: %.4e, NRMSE: %.4e, SRMSE: %.4e, SNRMSE: %.4e, TRMSE: %.4e, TRMSE: %.4e, PSNR: %.4e, SSIM: %.4e, Time: %.2f s\n', rmse_ring, nrmse_ring, rmse_ring_2, nrmse_ring_2, rmse_ring_3, nrmse_ring_3, psnr_ring, ssim_ring, timer_ring);

        end

        if FCTN
            [I, J, K] = size(X);
            X0 = reshape(X, [I J K/subdims(i)  subdims(i)]);
            Y0 = reshape(Y, [I J  K/subdims(i) subdims(i)]);

            Ndim      = ndims(X0);
            Nway      = size(X0);
            % 
            % 
            % %% Perform  algorithms
            % % initialization of the parameters
            % % Please refer to our paper to set the parameters
            % opts=[];
            % opts.max_R = [0,  6,  6,  6;
            %               0,  0,  6,  6;
            %               0,  0,  0,  6;
            %               0,  0,  0,  0];  
            % opts.R     = [0,  2,  2,  2;
            %               0,  0,  2,  2;
            %               0,  0,  0,  2;
            %               0,  0,  0,  0];
            % opts.tol   = 1e-5;
            % opts.maxit = 100;
            % opts.rho   = 0.1;
            % opts.origin = Y0;
            % %%%%%
            % fprintf('\n');
            % t0= tic;
            % % Please see the difference between 'inc_FCTN_TC' and 'inc_FCTN_TC_end' in README.txt. 
            % %[Re_tensor{i},G,Out]        = inc_FCTN_TC(F,Omega,opts);
            % [Re_tensor,G,Out]        = inc_FCTN_TC(Y0,Omega,opts);
            % time_fctn                     = toc(t0);
            % [psnr_fctn, ssim_fctn]          = quality_ybz(X0, Re_tensor);
            % [rmse_fctn, nrmse_fctn] =  evaluate(Re_tensor, X0, true(size(X0)));
            % errHist = Out.RSE;
            % method = "FCTN";
            % if missing_ratio == plot_rate
            %     save(sprintf("%s_%s_errHist.mat", name, method), 'errHist');
            % end
            % clear Re_tensor;
            % clear G;
            % fprintf("FCTN - RE: %.4e - PSN: %.4e - SSIM: %.4e - Timer: %.2f\n", nrmse_fctn, psnr_fctn, ssim_fctn, time_fctn);
            %% Perform RC_FCTN
            % Ind  = zeros(Nway);
            Ind  = ones(Nway);
            Ind(~mask_missing)  = 1;
            % lamb = 5000;
            gamma = 1e-3;
            deta = 1e-3;
            f = 10;
            % lambda = lamb/sqrt(max(Nway(1),Nway(2))*Nway(3)*Nway(4));
            % disp(lambda);
            lambda = 1.8;
            opts.gamma = gamma;
            opts.tol = 1e-6;
            opts.deta = deta;
            opts.maxit = 100;
            opts.f = f;
            opts.Xtrue = X0;
            t0=tic;
            [Re_tensor, S, Out,iter] = RC_FCTN(Y0, lambda, Ind, opts);
            Re_tensor = reshape(Re_tensor, [I, J, K]);
            S = reshape(S, [I, J, K]);
            time_fctn=toc(t0);
            errHist = Out.RSE_real;
            len = length(errHist);
            % disp( errHist(len-10:len));
            method = "FCTN";
            if missing_ratio == plot_rate
                        save(sprintf("%s_%s_errHist.mat", name, method), "errHist");
            end
                        [rmse_fctn, nrmse_fctn] = evaluate(Re_tensor, gt, mask_missing);
                        [rmse_fctn_2, nrmse_fctn_2] = evaluate(S, gt_2, ~mask_missing);
                        [rmse_fctn_3, nrmse_fctn_3] = evaluate(Re_tensor, X, true(size(S)));
             [psnr_fctn, ssim_fctn]          = quality_ybz(X0, Re_tensor);
            % [rmse_fctn, nrmse_fctn] =  evaluate(Re_tensor, X0, true(size(X0)));
            % errHist = Out.RSE;
            % method = "FCTN";
            % if missing_ratio == plot_rate
            %     save(sprintf("%s_%s_errHist.mat", name, method), 'errHist');
            % end
            % clear Re_tensor;
            % clear G;
            fprintf("FCTN - Missing REL %.4e RE: %.4e - PSN: %.4e - SSIM: %.4e - Timer: %.2f\n", nrmse_fctn, nrmse_fctn_3, psnr_fctn, ssim_fctn, time_fctn);
            clear Re_tensor;
            clear S;
                end

        if FCTN_2
            [I, J, K] = size(X);
            X0 = reshape(X, [I J K/subdims(i)  subdims(i)]);
            Y0 = reshape(Y, [I J  K/subdims(i) subdims(i)]);

            Ndim      = ndims(X0);
            Nway      = size(X0);

            opts=[];
            opts.max_R = [0,  6,  6,  6;
                          0,  0,  6,  6;
                          0,  0,  0,  6;
                          0,  0,  0,  0];  
            opts.R     = [0,  2,  2,  2;
                          0,  0,  2,  2;
                          0,  0,  0,  2;
                          0,  0,  0,  0];
            opts.tol   = 1e-5;
            opts.maxit = 100;
            opts.rho   = 0.1;
            opts.origin = Y0;
            %%%%%
            fprintf('\n');
            t0= tic;
            % Please see the difference between 'inc_FCTN_TC' and 'inc_FCTN_TC_end' in README.txt. 
            %[Re_tensor{i},G,Out]        = inc_FCTN_TC(F,Omega,opts);
            [Re_tensor,G, Xt,Out]        = inc_FCTN_TC(Y0,Omega,opts);
            time_fctn                     = toc(t0);
            [rmse_fctn, nrmse_fctn] = evaluate(Xt, gt, mask_missing);
            [psnr_fctn, ssim_fctn]          = quality_ybz(X0, Re_tensor);
            [rmse_fctn_2, nrmse_fctn_2] =  evaluate(Re_tensor, X0, true(size(X0)));
            errHist = Out.RSE;
            method = "FCTN";
            if missing_ratio == plot_rate
                save(sprintf("%s_%s_errHist.mat", name, method), 'errHist');
            end
            clear Re_tensor;
            clear G;
            fprintf("FCTN 2 - Missing-Re: %.4e RE: %.4e - PSN: %.4e - SSIM: %.4e - Timer: %.2f\n", nrmse_fctn, nrmse_fctn_2, psnr_fctn, ssim_fctn, time_fctn);
           
        end
    end
end

function [rmse, nrmse] = evaluate(X, gt, mask)
    X_hat = double(X);
    X_hat = X_hat(mask);
    
    rmse = norm(X_hat(:)-gt(:));
    nrmse = rmse / norm(gt(:));
    clear X_hat;
    
end
