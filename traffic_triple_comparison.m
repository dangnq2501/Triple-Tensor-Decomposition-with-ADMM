clc; clear;
close all;
addpath(genpath(pwd));
rng(0); % Reproducibility
plot_rate = 0.15;
% missing_rates = [0.05, 0.1, 0.2];
missing_rates = [plot_rate];
datasets = ["sensor", "network", "taxi","chicago"];
subdims = [6, 16, 10, 8];
TRIPLE = 0;
TTNN = 0;
RING = 0;
FCTN = 1;
SOFIA = 0;
r = 5;
for missing_id = 1:length(missing_rates)
    missing_ratio = missing_rates(missing_id);
    for i = 1:length(datasets)
        name = datasets(i);
        load(name + ".mat");  % Load variable X or T
       
        X = double(T);
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
            [rmse_re_3, nrmse_re_3] = evaluate(X_hat_re, X, true(size(X)));
            clear X_hat_re;
            fprintf('TRIPLE ADMM - RRE: %.2f, Time: %.2f s\n', nrmse_re_3, timer_re);

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
            [rmse_sofia_3, nrmse_sofia_3] = evaluate(X_hat_init, X, true(size(X)));
           
            save(sprintf("%s_%s_errHist.mat", name, method),"errHist");
            fprintf('SOFIA - RRE: %.2f, Time: %.2f s\n', nrmse_sofia_3,  timer_sofia);

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
                [rmse_ttnn_3, nrmse_ttnn_3] = evaluate(Z, X, true(size(X)));
                if missing_ratio == plot_rate
                    save(sprintf("%s_%s_errHist.mat", name, method), 'errHist');
                end
            clear Z;
            clear S;
            fprintf('TTNN - RRE: %.2f, Time: %.2f s\n', nrmse_ttnn_3, timer_ttnn);

        end

        if RING
            % ring_mask = true(size(Y));
            ring_mask = ~mask_missing;
            [X_hat_ring ,O_ring,RC,timer_ring,errHist]=RTRC(Y,ring_mask,1e-1,false, X);
            method = "ring";
            if missing_ratio == plot_rate
                save(sprintf("%s_%s_errHist.mat", name, method), 'errHist');
            end
            [rmse_ring_3, nrmse_ring_3] = evaluate(X_hat_ring, X, true(size(X)));
            clear O_ring;
            clear X_hat_ring;
            fprintf('Ring - RRE: %.2f, Time: %.2f s\n', nrmse_ring_3, timer_ring);

        end

        if FCTN
            [I, J, K] = size(X);
            fprintf("K = %d, subdi = %d\n", K, subdims(i));
            X0 = reshape(X, [I J K/subdims(i)  subdims(i)]);
            Y0 = reshape(Y, [I J  K/subdims(i) subdims(i)]);

            Ndim      = ndims(X0);
            Nway      = size(X0);
           
            Ind  = ones(Nway);
            Ind(~mask_missing)  = 1;
            lamb = 5000;
            gamma = 1e-3;
            deta = 1e-3;
            f = 0.1;
            lambda = lamb/sqrt(max(Nway(1),Nway(2))*Nway(3)*Nway(4));
            disp(lambda);
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
            method = "FCTN";
            if missing_ratio == plot_rate
                        save(sprintf("%s_%s_errHist.mat", name, method), "errHist");
            end
                       
            [rmse_fctn_3, nrmse_fctn_3] = evaluate(Re_tensor, X, true(size(S)));
          
            fprintf("FCTN - Missing RRE: %.2f, Timer: %.2f\n", nrmse_fctn_3, time_fctn);
            clear Re_tensor;
            clear S;
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
