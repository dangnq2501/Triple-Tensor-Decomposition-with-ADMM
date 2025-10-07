 % 'inc_FCTN_TC_end.m' supposes that the size of the factor G_k is 
 % R_{1,k}*R_{2,k}*...*R_{k-1,k}*R_{k,k+1}*...*R_{k,N}*I_k. 
 % (can set the FCTN rank as 1 in Matlab)
   
function [X, G, Out] = inc_FCTN_TC_end(F,Omega,opts)
if isfield(opts, 'tol');         tol   = opts.tol;              end
if isfield(opts, 'maxit');       maxit = opts.maxit;            end
if isfield(opts, 'rho');         rho   = opts.rho;              end
if isfield(opts, 'R');           R     = opts.R;                end
if isfield(opts, 'max_R');       max_R     = opts.max_R;        end
origin = opts.origin;
N = ndims(F); 
Nway = size(F);
X = F;

tempdim = R+R';  tempdim(tempdim==0) = []; 
tempdim = reshape(tempdim,N-1,N);   tempdim = tempdim'; 
tempdim(:,end+1) = Nway';

max_tempdim = max_R+max_R';  max_tempdim(max_tempdim==0) = []; 
max_tempdim = reshape(max_tempdim,N-1,N);  max_tempdim = max_tempdim';  
max_tempdim(:,end+1) = Nway';

G = cell(1,N);
for i = 1:N
    G{i} = rand(tempdim(i,:)); 
end

Out.RSE = [];

r_change =0.01;

for k = 1:maxit
    Xold = X;
    % Update G 
    for i = 1:N
        Xi = my_Unfold(X,Nway,i);
        Gi = my_Unfold(G{i},tempdim(i,:),N);
        Girest = tnreshape_new(tnprod_rest_new(G,i),N);
        tempC = Xi*Girest'+rho*Gi;
        tempA = Girest*Girest'+rho*eye(size(Gi,2));
        G{i}  = my_Fold(tempC*pinv(tempA),tempdim(i,:),N);
    end
    
    % Update X 
    X = (tnprod_new(G)+rho*Xold)/(1+rho);
    X(Omega) = F(Omega);
    
    %% check the convergence
    rse=norm(X(:)-origin(:))/norm(origin(:));
    Out.RSE = [Out.RSE,rse];
    
    if mod(k, 10) == 0  ||   k == 1 
        fprintf('inc_FCTN-TC: iter = %d   RSE = %f   \n', k, rse);
    end
    
    if rse < tol 
        break;
    end
    
    rank_inc=double(tempdim<max_tempdim);
    if rse<r_change && sum(rank_inc(:))~=0
    G = rank_inc_adaptive(G,rank_inc,N);
    tempdim = tempdim+rank_inc;
    r_change = r_change*0.5;
    end
    
end
end


function [G]=rank_inc_adaptive(G,rank_inc,N)
    % increase the estimated rank
    for j = 1:N
    G{j} = padarray(G{j},rank_inc(j,:),rand(1),'post');
    end
end
    

