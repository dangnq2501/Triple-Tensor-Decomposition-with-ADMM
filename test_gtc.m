load('/MATLAB Drive/GTC/L_PeMS08.mat');
%load('/MATLAB Drive/GTC/L_PeMS04.mat')
figure;
imagesc(L); % Hiển thị dữ liệu traffic (ma trận thời gian - không gian)
colorbar;
title('Traffic Volume Data (Original)');
xlabel('Time');
ylabel('Sensor Locations');