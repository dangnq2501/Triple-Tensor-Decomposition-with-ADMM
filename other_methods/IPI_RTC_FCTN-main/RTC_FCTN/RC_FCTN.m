function [X, S, Out,k] = RC_FCTN(X_noise, lambda, Ind1, opts)

% Solve the Robust Tensor Completion (RTC) problem based on FCTN Nuclear Norm by ADMM

% ---------------------------------------------
% Input:
%   X_noise    -   Observed data
%   lambda     -   >0, parameter
%   Ind1       -   position index
%   opts       -    Structure value in Matlab. The fields are
%      opts.tol        -   termination tolerance
%      opts.maxit      -   maximum number of iterations
%      opts.Xtrue      -   Clean data
%      opts.f          -   Control the threshold in SVT
%      opts.gamma/delta    -   parameter in ADMM

% Output:
%       X       -    Recovered low-rank component
%       S       -    Recovered sparse component
%       Out     -    Relative error
%       k       -    number of iterations

% Written by Yun-Yang Liu (lyymath@126.com)


if isfield(opts, 'tol');         tol   = opts.tol;              end
if isfield(opts, 'maxit');       maxit = opts.maxit;            end
if isfield(opts, 'Xtrue');       XT    = opts.Xtrue;            end
if isfield(opts, 'f');           f     = opts.f;                end
if isfield(opts, 'gamma');       gamma = opts.gamma;            end
if isfield(opts, 'deta');        deta  = opts.deta;             end


dim = ndims(X_noise);
Nway    = size(X_noise);
Order   = myorder(dim);    %Generate the rearrangement of vector [1,2,бнбн,N]
alpha = weightFCTN(Nway,Order);  %Compute the weight alpha_k
mu  = f*alpha;
Omega=find(Ind1==0);
Ind2=zeros(Nway);
Ind2(Omega)=1;
Out.RSE_real=[];Out.RSE=[];Out.Chg=[];
%% initialization
n1=size(Order,1);
n2 = ceil(dim/2);

% X = X_noise;
X = zeros(Nway);
Y = X_noise;
L = cell(1,n1);
Z = cell(1,n1);
for i = 1:n1
    L{i} = zeros(Nway);
    Z{i} = zeros(Nway);
end
S = zeros(Nway);
P = zeros(Nway);
Q = zeros(Nway);
E = zeros(Nway);

k = 1;
rse = 1;  %rse > tol
while  k <= maxit
    k = k+1;
    Xlast = X;
    Slast = S;
    %%  L - subproblem
    for n = 1:n1
        order = Order(n,:);
        temp = permute(X-Z{n}/mu(n),order);
        A = reshape(temp,prod(Nway(order(1:n2))),[]);
        Ln = SVT(A, alpha(n)./mu(n) );
        Ln = reshape(Ln, Nway(order));
        L{n}= ipermute(Ln,order);
    end
    
    %% S - subproblem
    S = prox_l1(E-Q/deta, lambda/deta);
    
    %% X,E- subproblem
    temp = mu(1)*(L{1}+Z{1}./mu(1));
    for n = 2:n1
        temp = temp + mu(n)*(L{n}+Z{n}./mu(n));
    end
    M  = temp+gamma*( Y+ P/gamma );
    N  = gamma*( Y+ P/gamma )+deta*( S+Q/deta );
    tt = ( gamma^2-(sum(mu) + gamma)*(gamma + deta) );
    X  = ( gamma*N-(gamma+deta)*M)./tt;
    E  = ( gamma*M-(sum(mu)+gamma)*N)./tt;
    
    %% Y - subproblem
    Y = Ind2.*(X + E - P/gamma)+Ind1.*X_noise;
    
    %% Update Z, P, Q
    for n=1:n1
        Z{n} =  Z{n} + mu(n)*(L{n}-X);
    end
    P = P + gamma*(Y-X-E);
    Q = Q + deta*(S-E);
    %%  Check convergence
    RSE = norm(X(:)+S(:)-XT(:))/norm(XT(:));
    Out.RSE_real = [Out.RSE_real,RSE];
    rse = norm(X(:)-Xlast(:))/norm(Xlast(:));
    Out.RSE = [Out.RSE,rse];
    
    chg1 = max(abs(X(:)-Xlast(:)));chg2 = max(abs(S(:)-Slast(:)));
    chg = max(chg1,chg2);
    Out.Chg = [Out.Chg,chg];
    
    % if chg < tol
    %     break;
    % end
    
    gamma = 1.5*gamma;
    deta  = 1.5*deta;
end
end

function Order = myorder(N)
%Generate the  rearrangement of vector [1,2,бнбн,N]
a=[1:N];
b=perms(a);
c=b(:,1:N/2);
c=sort(c,2);
c=unique(c,'row');
e=1;f=1;
for i=N:-1:N/2+1
    e=e*i;
    f=f*(N-i+1);
end
n1=e/(f*2);
Order=zeros(n1,N);
for i=1:n1
    Order(i,:)=[c(i,:) setdiff(a,c(i,:))];
end
end

function [lambda] = weightFCTN(Nway,Order)
%Compute the weight
n1=length(Nway);
n2=size(Order,1);
for k = 1:n2
    order=Order(k,:);
    IR = prod(Nway(order(1:n1/2)));
    IL = prod(Nway(order(n1/2+1:n1)));
    lambda(k) = max(IL,IR);
%     lambda(k) = min(IL,IR);
end
lambda = lambda/(sum(lambda));
end