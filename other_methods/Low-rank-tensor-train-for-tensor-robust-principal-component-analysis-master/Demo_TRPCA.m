clc; clear all; close all;
addpath(genpath(cd));
rand('seed', 42); 

EN_SNN   = 1;
EN_TNN   = 1;
EN_TTNN  = 1;
methodname  = {'SNN','TNN','TTNN'};
dataset = ["highway", "office", "sofa", "PETS2006"];

for data_id = 1:length(dataset)
    name = dataset(data_id);
    load(name+".mat");
    X0 = double(gray_images);
    X0 = X0/(max(X0(:)));
    maxP = max(abs(X0(:)));
    [n1 n2 n3] = size(X0);
    Xn = X0;
    %% 10%
    rhos = 0;
    ind = find(rand(n1*n2*n3,1)<rhos);
    Xn(ind) = rand(length(ind),1);

        %% SNN
        j = 1;
        if EN_SNN
            %%%%
            fprintf('\n');
            disp(['performing ',methodname{j}, ' ... ']);
            
            opts.mu = 1e-3;
            opts.tol = 1e-5;
            opts.rho = 1.2;
            opts.max_iter = 500;
            opts.DEBUG = 1;
            
            alpha = [7 9 2.6];
            [Xhat,E,err,iter] = trpca_snn(Xn,alpha,opts);
            method = "snn";
            tensor2video(Xhat, sprintf("%s_%s_Xhat.avi", name, method));
            tensor2video(E, sprintf("%s_%s_O.avi", name, method));
            save(sprintf("%s_%s_Xhat.mat", name, method), 'Xhat');
            save(sprintf("%s_%s_O.mat", name, method), 'E');
            errHist = err;
            save(sprintf("%s_%s_errHist.mat", name, method), 'errHist');
            Xhat = max(Xhat,0);
            Xhat = min(Xhat,maxP);
            alpha1 = alpha(1);
            alpha2 = alpha(2);
            alpha3 = alpha(3);
         
           PSNRvector = zeros(1,n3);
           for i = 1:1:n3
               J = 255*X0(:,:,i);
               I = 255*Xhat(:,:,i);
               PSNRvector(1,i) = PSNR(J,I,n1,n2);
           end       
            MPSNR = mean(PSNRvector);
        
            SSIMvector = zeros(1,n3);
            for i = 1:1:n3
                J = 255*X0(:,:,i);
                I = 255*Xhat(:,:,i);
                SSIMvector(1,i) = SSIM(J,I);
            end       
            MSSIM = mean(SSIMvector);
        
            imname = [num2str(name{1}),'_SNN_result_rho_',num2str(rhos,'%.1f'),'_psnr_',num2str(MPSNR,'%.2f'),'_ssim_',num2str(MSSIM,'%.4f'),'_alpha1_',num2str(alpha1,'%.2f'),'_alpha2_',num2str(alpha2,'%.2f'),'_alpha3_',num2str(alpha3,'%.2f'),'.mat'];
            save(imname,'Xhat');
            end
         end
        
        %% TNN
        j = j+1;
        if EN_TNN
            %%%%
            fprintf('\n');
            disp(['performing ',methodname{j}, ' ... ']);
            for lam = [2]
                for mu = [1e-4 1e-5] 
            opts.mu = mu;
            opts.tol = 1e-6;
            opts.rho = 1.1;
            opts.max_iter = 100;
            opts.DEBUG = 1;
        
            [N1,N2,N3] = size(Xn);
            lambda = lam/sqrt(max(N1,N2)*N3);
            [Xhat,E,err,iter, errHist] = trpca_tnn(Xn,lambda,opts);
            method = "tnn";
            config = sprintf("%s_%.2e_%.2e", method, lam, mu);
            tensor2video(Xhat, sprintf("%s_%s_Xhat.avi", name, method));
            tensor2video(E, sprintf("%s_%s_O.avi", name, method));
            save(sprintf("%s_%s_Xhat.mat", name, method), 'Xhat');
            save(sprintf("%s_%s_O.mat", name, method), 'E');
            save(sprintf("%s_%s_errHist.mat", name, method), 'errHist');
            Xhat = max(Xhat,0);
            Xhat = min(Xhat,maxP);
            
            PSNRvector = zeros(1,n3);
            for i = 1:1:n3
                J = 255*X0(:,:,i);
                I = 255*Xhat(:,:,i);
                PSNRvector(1,i) = PSNR(J,I,n1,n2);
            end
             MPSNR = mean(PSNRvector);
        
             SSIMvector = zeros(1,n3);
             for i = 1:1:n3
                 J = 255*X0(:,:,i);
                 I = 255*Xhat(:,:,i);
                 SSIMvector(1,i) = SSIM(J,I);
             end
                end
            end

             % MSSIM = mean(SSIMvector);
             % 
             % imname = [num2str(name{1}),'_TNN_result_rho_',num2str(rhos,'%.1f'),'_psnr_',num2str(MPSNR,'%.2f'),'_ssim_',num2str(MSSIM,'%.4f'),'_lambda_',num2str(lambda,'%.3f'),'.mat'];
             % save(imname,'Xhat');
        end
        
        %% TTNN
        j = j+1;
        if EN_TTNN
            %%%%
             fprintf('\n');
             disp(['performing ',methodname{j}, ' ... ']);
          for lambda = [7e-2]
              for deta = [2e-3]
             % Initial parameters
             Nway = size(X0);     % 9th-order dimensions for KA
             N = numel(Nway);                % numel返回Nway中的数量   
              [I1, J1, K1] = size(X0);
             f = 2;
             gamma = 0.003;        
             [Z, S, iter, relerr] = TT_TRPCA(X0, lambda, f, gamma, deta);
                 method = "ttnn";
            config = sprintf("%s_%.2e_%.2e", method, lambda, deta);
            tensor2video(Z, sprintf("%s_%s_Xhat.avi", name, method));
            tensor2video(S, sprintf("%s_%s_O.avi", name, method));
            save(sprintf("%s_%s_Xhat.mat", name, method), 'Z');
            save(sprintf("%s_%s_O.mat", name, method), 'S');
            errHist = relerr;
            save(sprintf("%s_%s_errHist.mat", name, method), 'errHist');
        end
         % Z_img = CastKet2Image22( Z, n1, n2, I1, J1 );
         % S_img = CastKet2Image22( S, n1, n2, I1, J1 );       
         % Z_img = max(Z_img,0);
         % Z_img = min(Z_img,maxP);   
         % 
         %  PSNRvector = zeros(1,n3);
         %  for i = 1:1:n3
         %      J = 255*X0(:,:,i);
         %      I = 255*Z_img(:,:,i);
         %      PSNRvector(1,i) = PSNR(J,I,n1,n2);
         %  end  
         %  MPSNR = mean(PSNRvector);
         % 
         %  SSIMvector = zeros(1,n3);
         %  for i = 1:1:n3
         %      J = 255*X0(:,:,i);
         %      I = 255*Z_img(:,:,i);
         %      SSIMvector(1,i) = SSIM(J,I);
         %  end
         %  MSSIM = mean(SSIMvector);
         % 
         %  imname = [num2str(name{1}),'_TTNN_result_rho_',num2str(rhos,'%.1f'),'_psnr_',num2str(MPSNR,'%.2f'),'_ssim_',num2str(MSSIM,'%.4f'),'_lambda_',num2str(lambda,'%.2f'),'_f_',num2str(f,'%.1f'),'_gamma_',num2str(gamma,'%.3f'),'_deta_',num2str(deta,'%.3f'),'.mat'];
         %  save(imname,'Z_img','S_img');
    end
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
