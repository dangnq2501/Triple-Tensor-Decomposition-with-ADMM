load('eeg_tensor.mat');
maxIter = 50;
tol = 1e-5;
for r = 1:3
    [A,B,C,errHist] = triple_decomp_ADMM_reg(eeg_tensor, r*20, 2, 0.1, maxIter, tol);
    
    figure;
    plot(errHist, 'o-','LineWidth',1.5);
    xlabel('Iteration'); ylabel('Relative Error');
    title('Triple Decomposition ADMM - Convergence');
    grid on;
end