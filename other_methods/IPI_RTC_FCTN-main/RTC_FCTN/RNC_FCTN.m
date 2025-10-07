function [X, G, E, Out,k] = RNC_FCTN(F,lambda,Omega,opts)

% Solve the Robust Tensor Completion (RTC) problem based on FCTN
% decomposition by PAM

% ---------------------------------------------
% Input:
%  F           -    Observed data 
%  lambda       -   >0, parameter
%  Omega       -    position index
%  opts        -    Structure value in Matlab. The fields are
%      opts.tol        -   termination tolerance
%      opts.maxit      -   maximum number of iterations      
%      opts.Xtrue      -   Clean data
%      opts.rho        -   > 0, is a proximal parameter
%      opts.rh         -   rh>=1, rh used to increase beta
%      opts.beta       -   penalty parameter
%      opts.R/max_R    -   FCTN-rank

% Output:
%       X       -    Recovered low-rank component
%       G       -    FCTN factors
%       E       -    Recovered sparse component
%       Out     -    Relative error
%       k       -    number of iterations

% Written by Yun-Yang Liu (lyymath@126.com)

if isfield(opts, 'tol');         tol   = opts.tol;              end
if isfield(opts, 'maxit');       maxit = opts.maxit;            end
if isfield(opts, 'Xtrue');       XT    = opts.Xtrue;            end
if isfield(opts, 'rho');         rho   = opts.rho;              end
if isfield(opts, 'rh');          rh    = opts.rh;               end
if isfield(opts, 'beta');        beta  = opts.beta;             end
if isfield(opts, 'R');           R     = opts.R;                end
if isfield(opts, 'max_R');       max_R = opts.max_R;            end

N = ndims(F); 
Nway = size(F);
X = F;
E = zeros(Nway);
Y = F;
tempdim = diag(Nway)+R+R';
max_tempdim = diag(Nway)+max_R+max_R';

%% initialization
G = cell(1,N);
for i = 1:N
    G{i} = rand(tempdim(i,:));
end

Out.RSE = [];Out.RSE_real = [];Out.PSNR=[];

r_change =0.01;

for k = 1:maxit
    Xold = X;
    %% Update G 
    for i = 1:N
        Xi = my_Unfold(X,Nway,i);
        Gi = my_Unfold(G{i},tempdim(i,:),i);
        Girest = tnreshape(tnprod_rest(G,i),N,i);
        tempC = Xi*Girest'+rho*Gi;
        tempA = Girest*Girest'+rho*eye(size(Gi,2));
        G{i}  = my_Fold(tempC*pinv(tempA),tempdim(i,:),i);
    end
    
    %% Update X 
    X = (tnprod(G)+rho*Xold+beta*(Y-E))/(1+rho+beta); 
    
    %% Update E
    E = softThres((beta*(Y-X)+rho*E)/(beta+rho), lambda/(beta+rho));
    
    %% Update M
    Y = (beta*(X+E)+rho*Y)/(beta+rho);
    Y(Omega)=F(Omega);
    
    %% check the convergence
    rse=norm(X(:)-Xold(:))/norm(Xold(:));
    Out.RSE = [Out.RSE,rse];
    rse_real = norm(XT(:)-X(:)-E(:))/norm(XT(:));
    Out.RSE_real = [Out.RSE_real, rse_real];
    if k>10&&rse < tol 
        break;
    end
    
    rank_inc=double(tempdim<max_tempdim);
    if rse<r_change && sum(rank_inc(:))~=0
    G = rank_inc_adaptive(G,rank_inc,N);
    tempdim = tempdim+rank_inc;
    r_change = r_change*0.5;
    end
    beta = rh*beta;
end
end


function [G]=rank_inc_adaptive(G,rank_inc,N)
    % increase the estimated rank
    for j = 1:N
    G{j} = padarray(G{j},rank_inc(j,:),rand(1),'post');
    end
end
    
function x = softThres(a, tau)
x = sign(a).* max( abs(a) - tau, 0);
end

