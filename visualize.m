name = 'highway';
    video_name = name+'_ADMM_ABC.mat';
    load(video_name);
    video_name= name+'_ADMM_ADMM_ABC.avi';
    tensor2video(result_ADMM_reg,video_name);

    video_name = name+'_ADMM_O.mat';
    load(video_name);
    video_name= name+'_ADMM_ADMM_O.avi';
    tensor2video(O,video_name);

    video_name = name+'_ADMM_ABCO.mat';
    load(video_name);
    video_name= name+'_ADMM_ADMM_ABCO.avi';
    tensor2video(total,video_name);

load('video_ADMM_ABC_MALS.mat');
tensor2video(result_ADMM_reg, 'highway_ADMM_MALS_ABC.avi');

load('video_ADMM_O_MALS.mat');
tensor2video(O, 'highway_ADMM_MALS_O.avi');

load('video_ADMM_ABCO_MALS.mat');
tensor2video(total, 'highway_ADMM_MALS_ABCO.avi');

load('video_ADMM_ABC_origin.mat');
tensor2video(result_ADMM_reg, 'highway_ADMM_ABC_origin.avi');

load('video_ADMM_O_origin.mat');
tensor2video(O, 'highway_ADMM_O_origin.avi');

load('video_ADMM_ABCO_origin.mat');
tensor2video(total, 'highway_ADMM_ABCO_origin.avi');
                                             
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
