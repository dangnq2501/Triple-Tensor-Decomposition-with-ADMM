clc; clear;

close all;
addpath(genpath('lib'));
addpath(genpath('RTC_FCTN'));
addpath(genpath('gray_video_mat_Ddta_v2'));
%% Load initial data
% load('HSV_test.mat')
name = "highway";
load(name+".mat");
method = "FCTN";
X = double(gray_images);
[I J K] = size(X);
X = reshape(X, [I J 1 K]);
if max(X(:))>1
    X = X/max(X(:));
end

%% Generate observed data
sample_ratio = 1;
fprintf('=== The sample ratio is %4.2f ===\n', sample_ratio);
T         = X;
Ndim = ndims(T);
Nway = size(T);
% Ndim      = ndims(T)+1;
% Nway      = [size(T) 1];
disp(Nway);
for i=1:Nway(3)
    for j=1:Nway(4)
        % Nhsi(:,:,i,j) = imnoise(T(:,:,i,j),'salt & pepper',0.1);
        Nhsi(:,:,i,j) = T(:,:,i,j);
    end
end
Omega     = find(rand(prod(Nway),1)<=sample_ratio);
F         = zeros(Nway);
F(Omega)  = Nhsi(Omega);


%% Interpolation initialization
A = reshape(F,Nway(1),Nway(2),[]);
Nhsi = reshape(Nhsi,Nway(1),Nway(2),[]);
Ind= zeros(size(A));
Ind(Omega)  = 1;
B = padarray(A,[20,20,20],'symmetric','both');
C = padarray(Ind,[20,20,20],'symmetric','both');
%a0 = interpolate2(B,C);
a1 = interpolate(shiftdim(B,1),shiftdim(C,1));
a1(a1<0) = 0;
a1(a1>1) = 1;
a1 = a1(21:end-20,21:end-20,21:end-20);
a1 = shiftdim(a1,2);
a1(Omega) = Nhsi(Omega);

a2 = interpolate(shiftdim(B,2),shiftdim(C,2));
a2(a2<0) = 0;
a2(a2>1) = 1;
a2 = a2(21:end-20,21:end-20,21:end-20);
a2 = shiftdim(a2,1);
a2(Omega) = Nhsi(Omega);

a3(Omega) = Nhsi(Omega);
a = 0.5*a1+0.5*a2;
X0 = a;
X1 = reshape(X0,Nway);

%% Perform  RNC_FCTN
opts=[];
for r12=[25]
    for r13=[7]
        for r14 = [3]
            for beta = [1]
                for rho = [1e-4]
                    for lamb = [1]
                        opts.max_R = [0,  r12,  r13,  r14;
                            0,   0,   r13,  r14;
                            0,   0,    0,   r13;
                            0,   0,    0,    0];
                        opts.R     = [0,  15,  2,  2;
                            0,  0,   2,  2;
                            0,  0,   0,  2;
                            0,  0,   0,  0];
                        opts.tol   = 1e-6;
                        opts.maxit = 100;
                        opts.rho   = rho;
                        opts.beta  = beta;
                        opts.rh  = 1;
                        opts.Xtrue = T;
                        lambda = lamb/sqrt(max(Nway(1),Nway(2))*Nway(3)*Nway(4));
                        t0= tic;
                        [Re_tensor,G,E,Out,iter]    = RNC_FCTN(X1,lambda,Omega,opts);
                        time                        = toc(t0);
                        Re_tensor = reshape(Re_tensor, [I J K]);
                        E = reshape(E, [I J K]);
                        save(sprintf("%s_%s_Xhat.mat", name, method), "Re_tensor");
                        save(sprintf("%s_%s_errHist.mat", name, method), "errHist");
                        save(sprintf("%s_%s_O.mat", name, method), "E");
                        tensor2video(Re_tensor, sprintf("%s_%s_Xhat.avi", name, method));
                        tensor2video(S, sprintf("%s_%s_O.avi", name, method));
                        % % [psnr2, ssim2]          = MSIQA(T*255, Re_tensor*255) 
                    end
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