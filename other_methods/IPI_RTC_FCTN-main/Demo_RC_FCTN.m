clc;clear;
close all;
addpath(genpath('lib'));
addpath(genpath('RTC_FCTN'));

%% Load initial data
addpath(genpath('gray_video_mat_data_v2'));
addpath(genpath('groundtruth_v2'));
%% Load initial data
dataset = ["highway", "office", "sofa", "PETS2006"];
% dataset = ["highway", "PETS2006"];
for data_id = 1:length(dataset)
    name = dataset(data_id);
    load(name+".mat");
    [I, J, K] = size(gray_images);
    X = reshape(double(gray_images), [I J 1 K]);
    disp(size(X));
    % load('CV_bunny_test.mat')
    if max(X(:))>1
        X = X/max(X(:));
    end
    load(name+"_gt.mat");
    groundTruth = gray_images;

    %% Generate observed data
    sample_ratio = 1;
    fprintf('=== The sample ratio is %4.2f ===\n', sample_ratio);
    T         = X;
    Ndim      = ndims(T);
    Nway      = size(T) ;
    Nhsi = T;
    % for i=1:Nway(3)
    %     for j=1:Nway(4)
    %         Nhsi(:,:,i,j) = imnoise(T(:,:,i,j),'salt & pepper',0.1);
    %     end
    % end
    Omega     = find(rand(prod(Nway),1)<=sample_ratio);
    F         = zeros(Nway);
    F(Omega)  = Nhsi(Omega);
    
    %% Perform RC_FCTN
    Ind  = zeros(Nway);
    % Ind(Omega)  = 1;
    for lamb = [5]
        for gamma = [1e-4]
            for deta = [1e-3]
                for f = [0.7]
                    disp(deta);
                    lambda = lamb/sqrt(max(Nway(1),Nway(2))*Nway(3));
                    opts.gamma = gamma;
                    opts.tol = 1e-6;
                    opts.deta = deta;
                    opts.maxit = 100;
                    opts.f = f;
                    opts.Xtrue = T;
                    opts.Xtrue(groundTruth>=200) = 0;
                    t0=tic;
                    [Re_tensor, S, Out,iter] = RC_FCTN(F, lambda, Ind, opts);
                    Re_tensor = reshape(Re_tensor, [I, J, K]);
                    S = reshape(S, [I, J, K]);
                    time=toc(t0);
                    errHist = Out.RSE_real;
                    method = "FCTN";
                    config = sprintf("%s_%.2e", method, gamma);
                    save(sprintf("%s_%s_Xhat.mat", name, method), "Re_tensor");
                    save(sprintf("%s_%s_errHist.mat", name, method), "errHist");
                    save(sprintf("%s_%s_O.mat", name, method), "S");
                    tensor2video(Re_tensor, sprintf("%s_%s_Xhat.avi", name, method));
                    tensor2video(S, sprintf("%s_%s_O.avi", name, method));
                    [psnr,ssim]=MSIQA(T*255, Re_tensor*255);

                    imname=['HSV_SaP=0.1_SR=0.3_lambda_',num2str(lamb),'_gamma_',num2str(gamma),'_deta_',num2str(deta),'_f_',num2str(f),'_psnr_',num2str(psnr),'_ssim_',num2str(ssim),'_time_',num2str(time),'.mat'];
                    % save(imname,'Re_tensor');    % save results
                end
            end
        end
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