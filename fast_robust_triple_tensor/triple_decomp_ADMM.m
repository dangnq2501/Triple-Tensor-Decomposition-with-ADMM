function [A, B, C, O, errHist] = triple_decomp_ADMM(D, r, opts)
% INPUT:
%   D      : dữ liệu (n1*n2*n3)
%   r      : triple rank
%   opts.muL, opts.rhoL : penalty & multiplier cho constraint D-L-O
%   opts.muO, opts.rhoO : penalty & multiplier cho constraint O-E
%   opts.lambda         : weight on ||E||_1
%   opts.maxIter, opts.tol, opts.disp
%
% OUTPUT:
%   A,B,C  : factors triple L ≈ triple_product(A,B,C)
%   O,E    : sparse components (clone E)
%   errHist: history of residual

[n1,n2,n3] = size(D);
muL   = opts.mu;    rhoL = opts.rho;    muL_max = opts.mu*1e6;
muO   = opts.mu;    rhoO = opts.rho;    muO_max = opts.mu*1e6;
lambda= opts.lambda;
lambda2= opts.lambda2;
maxIter=opts.maxIter; tol=opts.tol; disp=opts.disp;

% Initialize variables
A = randn(n1,r,r);   B = randn(r,n2,r);   C = randn(r,r,n3);
O = zeros(n1,n2,n3); E = O;
Y_L = zeros(n1,n2,n3);   % dual for D-L-O
Y_O = zeros(n1,n2,n3);   % dual for O-E

normD = norm(D(:));
errHist = zeros(maxIter,1);

for k = 1:maxIter
    %% 1) Update L (A,B,C) base on T = D - O + Y_L/muL
    T = D - O + (1/muL)*Y_L;
    A = update_A(T, A, B, C, lambda2);
    B = update_B(T, A, B, C, lambda2);
    C = update_C(T, A, B, C);

    L = triple_product(A,B,C);

    %% 2) Update O:  min_O  (muL/2)||D-L-O + Y_L/muL||^2 + (muO/2)||O-E + Y_O/muO||^2
    R1 = D - L + (1/muL)*Y_L;
    R2 = E - (1/muO)*Y_O;
    O  = (muL*R1 + muO*R2) / (muL + muO);

    %% 3) Update E by soft-threshold base on O + Y_O/muO
    R3 = O + (1/muO)*Y_O;
    E  = sign(R3) .* max(abs(R3) - lambda/muO, 0);

    %% 4) Update duals
    resL = D - L - O;
    resO = O - E;
    Y_L = Y_L + muL * resL;
    Y_O = Y_O + muO * resO;

    %% 5) Update mu
    muL = min(muL*rhoL, muL_max);
    muO = min(muO*rhoO, muO_max);

    errHist(k) = norm(resL(:)) / normD + norm(resO(:)) / normD;
    if disp && mod(k,10)==0
        fprintf("Iter %d, errL=%.2e, errO=%.2e\n", k, norm(resL(:))/normD, norm(resO(:))/normD);
    end
    if k>1 && abs(errHist(k)-errHist(k-1))<tol*errHist(k-1)
        break;
    end
end

errHist = errHist(1:k);

end


function A = update_A(X, A, B, C, alphaA)
    X1 = unfold(X,1);
    F  = buildF(B,C);

    G = F * F.' + alphaA * eye(size(F,1));
    A1 = (X1 * F.') * pinv(G);

    A = reshape_A_from_A1(A1, size(A,1), size(A,2));
end

function B = update_B(X, A, B, C, alphaB)
    X2 = unfold(X, 2);
    G = buildG(A, C);
    B_old_unf = (X2 * G') * pinv(G * G' + alphaB* eye(size(G,1)));
    B = reshape_B_from_B2(B_old_unf, size(B,2), size(B,1));
end

function C = update_C(X, A, B, C)
    X3 = unfold(X, 3);
    H = buildH(A, B);
    C_old_unf = (X3 * H') * pinv(H * H' + 1e-9 * eye(size(H,1)));
    C = reshape_C_from_C3(C_old_unf, size(C,3), size(C,1));
end

function Xn = unfold(X, mode)
    [n1, n2, n3] = size(X);
    switch mode
        case 1
            Xn = reshape(X, [n1, n2*n3]);
        case 2
            Xn = reshape(permute(X, [2, 1, 3]), [n2, n1*n3]);
        case 3
            Xn = reshape(permute(X, [3, 1, 2]), [n3, n1*n2]);
        otherwise
            error('Mode must be 1, 2, or 3.');
    end
end

function A = reshape_A_from_A1(A1, n1, r)
    A = zeros(n1, r, r);
    for i = 1:n1
        A(i,:,:) = reshape(A1(i,:), [r, r]);
    end
end

function B = reshape_B_from_B2(B2, n2, r)
    B = zeros(r, n2, r);
    for j = 1:n2
        B(:,j,:) = reshape(B2(j,:), [r, r]);
    end
end

function C = reshape_C_from_C3(C3, n3, r)
    C = zeros(r, r, n3);
    for t = 1:n3
        C(:,:,t) = reshape(C3(t,:), [r, r]);
    end
end

function F = buildF(B, C)
    [r, n2, ~] = size(B);
    [~, ~, n3] = size(C);
    B_unfold = reshape(unfold(B, 2), [n2, r*r, 1]);
    C_unfold = reshape(unfold(C, 3)', [1, r*r, n3]);
    F = B_unfold .* C_unfold;
    F = reshape(F, [n2, r, r, n3]);
    F = reshape(permute(F, [2,3,1,4]), [r*r, n2*n3]);
end

function G = buildG(A, C)
    [n1, r, ~] = size(A);
    [~, ~, n3] = size(C);
    A_unfold = reshape(unfold(A, 1), [n1, r*r, 1]);
    C_unfold = reshape(unfold(C, 3)', [1, r*r, n3]);
    G = A_unfold .* C_unfold;
    G = reshape(G, [n1, r, r, n3]);
    G = reshape(permute(G, [2,3,1,4]), [r*r, n1*n3]);
end

function H = buildH(A, B)
    [n1, r, ~] = size(A);
    [~, n2, ~] = size(B);
    A_unfold = reshape(unfold(A, 1), [n1, r*r, 1]);
    B_unfold = reshape(unfold(B, 2)', [1, r*r, n2]);
    H = A_unfold .* B_unfold;
    H = reshape(H, [n1, r, r, n2]);
    H = reshape(permute(H, [2,3,1,4]), [r*r, n1*n2]);
end