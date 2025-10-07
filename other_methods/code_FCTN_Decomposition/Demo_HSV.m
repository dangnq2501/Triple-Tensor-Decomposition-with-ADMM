%% =================================================================
% This script runs the FCTN decomposition-based TC method
%
% More detail can be found in [1]
% [1] Yu-Bang Zheng, Ting-Zhu Huang*, Xi-Le Zhao*, Qibin Zhao, Tai-Xiang Jiang
%     Fully-Connected Tensor Network Decomposition and Its Application
%     to Higher-Order Tensor Completion
%
% Please make sure your data is in range [0, 1].
%
% Created by Yu-Bang Zheng (zhengyubang@163.com)
% Jun. 06, 2020
% Updated by Yu-Bang Zheng
% Dec. 06, 2020

%% =================================================================
%clc;
clear;
close all;
addpath(genpath('lib'));
addpath(genpath('data'));

%%
methodname    = {'Observed', 'FCTN-TC'};
Mnum          = length(methodname);
Re_tensor     = cell(Mnum,1);
psnr          = zeros(Mnum,1);
ssim          = zeros(Mnum,1);
time          = zeros(Mnum,1);

%% Load initial data
load('HSV_test.mat')
if max(X(:))>1
    X = X/max(X(:));
end

%% Sampling with random position
sample_ratio = 0.05;
fprintf('=== The sample ratio is %4.2f ===\n', sample_ratio);
T         = X;
Ndim      = ndims(T);
Nway      = size(T);
rand('seed',2);
Omega     = find(rand(prod(Nway),1)<sample_ratio);
F         = zeros(Nway);
F(Omega)  = T(Omega);
%%
i  = 1;
Re_tensor{i} = F;
[psnr(i), ssim(i)] = quality_ybz(T*255, Re_tensor{i}*255);
enList = 1;

%% Perform  algorithms
i = i+1;
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
opts.maxit = 1000;
opts.rho   = 0.1;
%%%%%
fprintf('\n');
disp(['performing ',methodname{i}, ' ... ']);
t0= tic;
% Please see the difference between 'inc_FCTN_TC' and 'inc_FCTN_TC_end' in README.txt. 
%[Re_tensor{i},G,Out]        = inc_FCTN_TC(F,Omega,opts);
[Re_tensor{i},G,Out]        = inc_FCTN_TC_end(F,Omega,opts);
time(i)                     = toc(t0);
[psnr(i), ssim(i)]          = quality_ybz(T*255, Re_tensor{i}*255);
enList = [enList,i];

%% Show result
fprintf('\n');
fprintf('================== Result =====================\n');
fprintf(' %8.8s    %5.4s    %5.4s    \n','method','PSNR', 'SSIM' );
for i = 1:length(enList)
    fprintf(' %8.8s    %5.3f    %5.3f    \n',...
    methodname{enList(i)},psnr(enList(i)), ssim(enList(i)));
end
fprintf('================== Result =====================\n');
figure,
show_figResult(Re_tensor,T,min(T(:)),max(T(:)),methodname,enList,1,prod(Nway(3:end)))
