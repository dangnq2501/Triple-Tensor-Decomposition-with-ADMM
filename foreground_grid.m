clc; clear;
close all;
addpath(genpath("data"));
datasets = ["highway", "sofa", "office", "PETS2006"];
titles = ["Highway", "Sofa", "Office", "PETS2006"];
% datasets = ["highway"];

methods = ["Observed", "gt","ttnn", "sofia", "ring", "FCTN", "triple_re"];
runtimes_sec = [
    NaN, NaN, 201.47, 370.57, 1031.97, 50.64, 33.68;  % highway
    NaN, NaN, 225.50, 419.57, 1147.48, 56.92, 37.05;  % sofa
    NaN, NaN, 226.36, 424.15, 1148.17, 56.64, 43.98;  % office
    NaN, NaN, 229.23, 395.39, 1215.11, 92.62, 35.93;  % PETS2006
];
frame_ids = [200, 50, 50, 50, 50];
cols = length(methods);
rows = length(datasets);
figure;
cnt = 0;
t = tiledlayout(rows, cols, 'TileSpacing', 'compact', 'Padding','compact');
for dataset_id = 1:length(datasets)
    for method_id = 1:length(methods)
    
        dataname = sprintf("%s_%s_Xhat.mat", datasets(dataset_id), methods(method_id));
        if methods(method_id) == "gt" 
            dataname = sprintf("%s_%s.mat", datasets(dataset_id), methods(method_id));
        elseif methods(method_id) == "Observed"
            dataname = sprintf("%s_raw.mat", datasets(dataset_id));
        elseif methods(method_id) == "FCTN"
            dataname = sprintf("%s_%s_Xhat.mat", datasets(dataset_id), methods(method_id));
        end
        disp(dataname);
        data = load(dataname);
        variable = fieldnames(data);
        X = data.(variable{1});
        % disp(size(X));
        cnt = cnt + 1;
   
        idx = cnt;
        ax = nexttile(cnt);
      
        frame_id = frame_ids(dataset_id);

        imagesc(X(:,:,frame_id:frame_id));
        colormap gray; axis off;         
        rt = runtimes_sec(dataset_id, method_id);
        if isnan(rt)
            rt_str = "";
        else
            rt_str = sprintf('%.2f s', rt);
        end
        text(ax, 0.5, -0.06, rt_str, 'Units','normalized', ...
             'HorizontalAlignment','center', 'VerticalAlignment','top', ...
             'FontSize', 11, 'FontWeight', 'bold');

        if dataset_id == 1
            method_name = methods(method_id);
            if methods(method_id) == "triple_re"
                method_name = "Triple";
            elseif methods(method_id) == "gt"
                method_name = "FG";
            elseif methods(method_id) == "ttnn"
                method_name = "TTNN";
            elseif methods(method_id) == "sofia"
                method_name = "Sofia";
            elseif methods(method_id) == "ring"
                method_name = "TRLRF";
            elseif methods(method_id) == "FCTN_3"
                method_name = "RC-FCTN";
            end
            title( method_name, 'FontWeight', 'bold','FontSize', 12);
            % text(ax, -0.1, 0.5,  methods(method_id), ...
            %     'Units', 'normalized', ...
            %     'HorizontalAlignment', 'center', ...
            %     'FontWeight', 'bold', ...
            %     'FontSize', 12, ...
            %     'Rotation', 90);
        end
        if method_id == 1
            text(ax, -0.3, 0.5,  titles(dataset_id), ...
                'Units', 'normalized', ...
                'HorizontalAlignment', 'center', ...
                'FontWeight', 'bold', ...
                'FontSize', 13, ...
                'Rotation', 90);
            % title( titles(dataset_id), 'FontWeight', 'bold','FontSize', 14);
        end

    end
end 


% 
% for i = 1:n
%     subplot(rows, cols, i);
%     imagesc(video(:,:,frame_ids(i)));
%     colormap gray; axis off; axis image;
%     title(sprintf('Frame %d', frame_ids(i)));
% end
% sgtitle('Gray Video');

saveas(gcf, 'gray_video_frames_grid.fig');

function [] = tensor2video(X, video_name)          
X_uint8 = im2uint8(mat2gray(X));

v = VideoWriter(video_name);  
open(v);

numFrames = size(X_uint8, 3);
for t = 1:numFrames
    frame = X_uint8(:, :, t); 
    writeVideo(v, frame);
end
end