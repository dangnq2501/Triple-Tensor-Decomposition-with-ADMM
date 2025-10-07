clc; clear;
addpath(genpath(pwd));
rng(0);
datasets = ["PETS2006", "sofa", "highway", "office"];
plot_rate = 0;
missing_rates = [plot_rate];

subdims = [20, 20, 20, 20, 20];
TRIPLE = 0;
TTNN = 0;
SOFIA = 0;
RING = 1;
FCTN = 0;
FCTN_2 = 0;
r = 5;
for missing_id = 1:length(missing_rates)
    missing_ratio = missing_rates(missing_id);
    for i = 1:length(datasets)
        name = datasets(i);
        load(name + ".mat");  % Load variable X or T
        X = double(gray_images);
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
        save(sprintf("%s_raw.mat", name), 'Y');
        % load(sprintf("%s_raw.mat", name));
        
        % Y(mask_missing) = max(X(:))*3;
        gt = X(mask_missing);
        gt_2 = X(~mask_missing);
        o_mask= true(size(X));
        %% Step 2: Run ADMM-based Triple Decomposition
        if TRIPLE
            opts.maxIter = 100;
            opts.tol = 1e-5;
            opts.mu = 1e-2;
            opts.lambda = 1.8;
            opts.lambda2 = 1e-2;
            opts.rho = 1.2;
            opts.alphaA = 1e-3;
            opts.alphaB = 1e-3;
            opts.disp = 1;
            opts.origin = X;
            tic;
            % [A, B, C, O, E, Out] = triple_ADMM_masked(Y,  ~mask_missing, r, opts);
            % errHist = Out.errHist;
            [A, B, C, O, errHist] = triple_decomp_ADMM_outlier(Y, r, opts);
            timer_re = toc;
            X_hat_re = triple_product(A, B, C);            
            method = "triple_re";
            if missing_ratio == plot_rate
                save(sprintf("%s_%s_errHist.mat", name, method), 'errHist');
                save(sprintf("%s_%s_Xhat.mat", name, method), 'X_hat_re');
                save(sprintf("%s_%s_O.mat", name, method), 'O');
            end

            [rmse_re, nrmse_re] = evaluate(X_hat_re, gt, mask_missing);
            [rmse_re_2, nrmse_re_2] = evaluate(O, gt_2, ~mask_missing);
            [rmse_re_3, nrmse_re_3] = evaluate(X_hat_re+O, X, true(size(X)));
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
            
            m = 1; 
            cycles = 1;        
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
            [rmse_sofia_3, nrmse_sofia_3] = evaluate(X_hat_init+O_init, X, true(size(X)));
            [psnr_sofia, ssim_sofia]          = quality_ybz(X, X_hat_init);
            % tensor2video(X_hat_init, sprintf("%s_%_Xhat.avi", name, method));
            % tensor2video(O_init, sprintf("%s_%s_O.avi", name, method));
            if missing_ratio == plot_rate
                save(sprintf("%s_%s_errHist.mat", name, method), 'errHist');
                save(sprintf("%s_%s_Xhat.mat", name, method), 'X_hat_init');
                save(sprintf("%s_%s_O.mat", name, method), 'O_init');
            end 
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
                 [Z, S, iter, relerr] = TT_TRPCA(Y, lambda, f, gamma, deta, X);
                 timer_ttnn = toc;
                     method = "ttnn";
                [rmse_ttnn, nrmse_ttnn] = evaluate(Z, gt, mask_missing);
                [rmse_ttnn_2, nrmse_ttnn_2] = evaluate(S, gt_2, ~mask_missing);
                [rmse_ttnn_3, nrmse_ttnn_3] = evaluate(Z+S, X, true(size(X)));
                [psnr_ttnn, ssim_ttnn]          = quality_ybz(X, Z);
                errHist = relerr;
                if missing_ratio == plot_rate
                    save(sprintf("%s_%s_Xhat.mat", name, method), 'Z');
                    save(sprintf("%s_%s_O.mat", name, method), 'S');
                    save(sprintf("%s_%s_errHist.mat", name, method), 'errHist');
                end
            clear Z;
            clear S;
            fprintf('TTNN - RMSE: %.4e, RRE: %.4e, SRMSE: %.4e, SRRE: %.4e, TRMSE: %.4e, TRRE: %.4e, PSNR: %.4e, SSIM: %.4e, Time: %.2f s\n', rmse_ttnn, nrmse_ttnn, rmse_ttnn_2, nrmse_ttnn_2, rmse_ttnn_3, nrmse_ttnn_3, psnr_ttnn, ssim_ttnn, timer_ttnn);

        end

        if RING
            % ring_mask = ~mask_missing;
            ring_mask = true(size(Y));
            [X_hat_ring ,O_ring,RC,timer_ring,errHist]=RTRC(Y,ring_mask,1e-3,false, X);
            method = "ring";
            if missing_ratio == plot_rate
                save(sprintf("%s_%s_errHist.mat", name, method), 'errHist');
                save(sprintf("%s_%s_Xhat.mat", name, method), 'X_hat_ring');
                    save(sprintf("%s_%s_O.mat", name, method), 'O_ring');
            end
            [rmse_ring, nrmse_ring] = evaluate(X_hat_ring, gt, mask_missing);
            [rmse_ring_2, nrmse_ring_2] = evaluate(O_ring, gt_2, ~mask_missing);
            [rmse_ring_3, nrmse_ring_3] = evaluate(X_hat_ring+O_ring, X, true(size(X)));
            [psnr_ring, ssim_ring]          = quality_ybz(X, X_hat_ring);
            clear O_ring;
            clear X_hat_ring;
            fprintf('Ring - RMSE: %.4e, NRMSE: %.4e, SRMSE: %.4e, SNRMSE: %.4e, TRMSE: %.4e, TRMSE: %.4e, PSNR: %.4e, SSIM: %.4e, Time: %.2f s\n', rmse_ring, nrmse_ring, rmse_ring_2, nrmse_ring_2, rmse_ring_3, nrmse_ring_3, psnr_ring, ssim_ring, timer_ring);

        end
        if FCTN
            [I, J, K] = size(X);
            X0 = reshape(X, [I J subdims(i) K/subdims(i) ]);
            Y0 = reshape(Y, [I J  subdims(i) K/subdims(i)]);

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
            Ind  = zeros(Nway);
            Ind(~mask_missing)  = 1;
            lamb = 1;
            gamma = 1e-3;
            deta = 1e-3;
            f = 0.7;
            % lambda = lamb/sqrt(max(Nway(1),Nway(2))*Nway(3)*Nway(4));
            lambda = 1.8;
            opts.gamma = gamma;
            opts.tol = 1e-4;
            opts.deta = deta;
            opts.maxit = 100;
            opts.f = f;
            opts.Xtrue = Y0;
            t0=tic;
            [Re_tensor, S, Out,iter] = RC_FCTN(Y0, lambda, Ind, opts);
            Re_tensor = reshape(Re_tensor, [I, J, K]);
            S = reshape(S, [I, J, K]);
            time=toc(t0);
            errHist = Out.RSE_real;
            method = "FCTN";
            if missing_ratio == plot_rate
                save(sprintf("%s_%s_errHist.mat", name, method), 'errHist');
                save(sprintf("%s_%s_Xhat.mat", name, method), 'Re_tensor');
                    save(sprintf("%s_%s_O.mat", name, method), 'S');
            end
                        % [rmse_fctn, nrmse_fctn] = evaluate(Re_tensor, gt, mask_missing);
                        % [rmse_fctn_2, nrmse_fctn_2] = evaluate(S, gt_2, ~mask_missing);
                        % [rmse_fctn_3, nrmse_fctn_3] = evaluate(Re_tensor+S, X, true(size(S)));
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
            Omega = ones(size(Y0));
            [Re_tensor,G, Xt,Out]        = inc_FCTN_TC(Y0,Omega,opts);
            time_fctn                     = toc(t0);
            [rmse_fctn, nrmse_fctn] = evaluate(Xt, gt, mask_missing);
            [psnr_fctn, ssim_fctn]          = quality_ybz(X0, Re_tensor);
            [rmse_fctn_2, nrmse_fctn_2] =  evaluate(Re_tensor, X0, true(size(X0)));
            errHist = Out.RSE;
            fprintf("FCTN - RE: %.4e - PSN: %.4e - SSIM: %.4e - Timer: %.2f\n", nrmse_fctn, psnr_fctn, ssim_fctn, time_fctn);

            method = "FCTN_2";
            if missing_ratio == plot_rate
                save(sprintf("%s_%s_errHist.mat", name, method), 'errHist');
                save(sprintf("%s_%s_Xhat.mat", name, method), 'Re_tensor');
                method = "FCTN_3";
                save(sprintf("%s_%s_Xhat.mat", name, method), 'Xt');
            end
            clear Re_tensor;
            clear G;

           
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

function [] = tensor2video(X, video_name)          
X_uint8 = im2uint8(mat2gray(X));

v = VideoWriter(video_name);  
open(v);

numFrames = size(X_uint8, 3);
for t = 1:numFrames
    frame = X_uint8(:, :, t); 
    writeVideo(v, frame);
end

close(v);
end


function [] = eval(origin, groundtruth, background, foreground)
% disp(size(origin));
% disp(size(background));
% disp(size(foreground));
if ~isequal(size(origin), size(background), size(groundtruth))
    error('Unmatched origin, background and groundtruth');
end

numFrames = size(origin, 3);


pred_mask = zeros(size(origin));

for i = 1:numFrames
    frame_origin = double(origin(:,:,i));
    frame_bg     = double(background(:,:,i));
    % diff_img = abs(frame_origin-frame_bg); 
    diff_img = abs(foreground(:, :, i));
    
    level = graythresh(diff_img);
    thresh_val = level * 255;


    pred_mask(:,:,i) = diff_img > 50;
end
% tensor2video(pred_mask, "highway_foreground.avi");
gt_mask = (groundtruth == 255) ;
ns_mask = (groundtruth == 170);
TP_total = 0; FP_total = 0; FN_total = 0; TN_total = 0;
total_pixels = numel(gt_mask);

for i = 1:numFrames
    p_mask = pred_mask(:,:,i);
    g_mask = gt_mask(:,:,i);
    m_mask = ns_mask(:, :, i);
    TP = sum(p_mask(:) & (g_mask(:)|m_mask(:)));
    FP = sum(p_mask(:) & (~g_mask(:)));
    FN = sum(~p_mask(:) & (g_mask(:)));
    TN = sum(~p_mask(:) & ((~g_mask(:))|m_mask(:)));
    
    TP_total = TP_total + TP;
    FP_total = FP_total + FP;
    FN_total = FN_total + FN;
    TN_total = TN_total + TN;
end

precision = TP_total / (TP_total + FP_total);
recall    = TP_total / (TP_total + FN_total);
F1        = 2 * precision * recall / (precision + recall);
PWC       = 100 * (FP_total + FN_total) / total_pixels;
fprintf("TP: %d FP: %d TN: %d FN: %d \n", TP_total, FP_total, TN_total, FN_total);
fprintf('Precision: %.4f\n', precision);
fprintf('Recall   : %.4f\n', recall);
fprintf('F1 Score : %.4f\n', F1);
fprintf('PWC      : %.4f%%\n', PWC);
end


function [] = mAP(origin, groundtruth, background, foreground)
numFrames = size(origin, 3);
APs = zeros(numFrames, 1);
alpha = 0.5; 
cnt = 0;
for i = 1:numFrames
    frame_origin = double(origin(:,:,i));
    frame_bg     = double(background(:,:,i));
    diff_img     = abs(foreground(:, :, i));
    % diff_img = abs(frame_origin-frame_bg);    
    level = graythresh(diff_img);
    T = level  * 255;
    pred_prob = 1 ./ (1 + exp(-alpha * (diff_img - T)));
    
    gt_mask = double(groundtruth(:,:,i) == 255);
    if numel(unique(gt_mask)) < 2
        APs(i) = NaN;
        continue;
    end
    [recall, precision, ~] = perfcurve(gt_mask(:), pred_prob(:), 1, 'xCrit', 'reca', 'yCrit', 'prec');

    validIdx = ~isnan(precision) & ~isnan(recall);
    if sum(validIdx) > 1
        APs(i) = trapz(recall(validIdx), precision(validIdx));
        cnt = cnt + 1;
    else

        APs(i) = 0; % hoặc một giá trị phù hợp nếu không đủ điểm để tính tích phân
    end
end
mAP = mean(APs(~isnan(APs)));
fprintf('Mean Average Precision (mAP): %.4f\n\n', mAP);
end
