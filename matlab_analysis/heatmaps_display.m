clear all
close all
clc

foldername_inferences = 'SlidingWindow_Inferences/';
SlideName='PACCT1_8025594_AperioUUID26162';

load(['heatmaps/processed_' SlideName '.mat']);
im = im2double(imread(['heatmaps/thumb_' SlideName '.jpg']));

%% process gradient map
% truncation of high values
th = quantile(map_P_grad(:), 0.97); 
map_P_trunc = map_P_grad;
map_P_trunc(map_P_grad > th) = th;

% smoothing
eps = 0.0001;
map_P_smooth = imgaussfilt(log(map_P_grad+eps), 1); 

% resize to fit thumb
mask_valid = map_P_score ~= 0;
S = 12.8; 
mask_valid = imresize(mask_valid, S);
map_grad_processed = imresize(map_P_smooth, S, "bilinear");

% scale
min_grad = log(0.0001) + 3;
max_grad = max(map_P_smooth(:)) + 0.2;
map_grad_processed = (map_grad_processed - min_grad)/(max_grad - min_grad);

map_grad_processed = min(1, max(0, map_grad_processed)); % Ensure scores are in [0, 1]


%% process score map
% resize to fit thumb
map_score_processed = imresize(map_P_score, S);

% clean noise
map_P_score_valid = map_score_processed(mask_valid & (map_grad_processed > 0.6));

fprintf('Score range: %.1f - %.1f \n', quantile(map_P_score_valid, 0.025), quantile(map_P_score_valid, 0.975))

% crop to fit thumb
map_score_processed(end:size(im,1),:) = 0;
map_score_processed(:,end:size(im,2)) = 0;

% scale for presentation
score_scale = 60;
map_score_processed = map_score_processed/score_scale;
map_score_processed = max(0, min(1, map_score_processed)); % Ensure scores are in [0, 1]
%% define colormap
color_low = [0, 0.2, 0.7];  % Blue
color_midlow = [0.2, 0.5, 0];  % Olive
color_midhigh = [1, 1, 0];     % Yellow
color_high = [1, 0, 0]; % Red

% Define breakpoints in the map_final_score (0 -> 1)
breakpoints = [0, 0.33, 0.66, 1];

% Interpolate colors based on map_final_score

% Interpolate for each color channel
color_overlay = cat(3, ...
    interp1(breakpoints, [color_low(1), color_midlow(1), color_midhigh(1), color_high(1)], map_score_processed), ...
    interp1(breakpoints, [color_low(2), color_midlow(2), color_midhigh(2), color_high(2)], map_score_processed), ...
    interp1(breakpoints, [color_low(3), color_midlow(3), color_midhigh(3), color_high(3)], map_score_processed));

colormap_matrix = [ ...
    interp1(breakpoints, [color_low(1), color_midlow(1), color_midhigh(1), color_high(1)], linspace(0,1,256))', ... % Red channel
    interp1(breakpoints, [color_low(2), color_midlow(2), color_midhigh(2), color_high(2)], linspace(0,1,256))', ... % Green channel
    interp1(breakpoints, [color_low(3), color_midlow(3), color_midhigh(3), color_high(3)], linspace(0,1,256))'];    % Blue channel

%% 
% crop to fit thumb
map_grad_processed(end:size(im,1),:) = 0;
map_grad_processed(:,end:size(im,2)) = 0;


alpha = map_grad_processed;
im_overlay = im .* (1 - alpha) + color_overlay .* alpha;

figure
imshow(alpha,[]);
figure;
imshow(map_score_processed,[])
figure;
imshow(im);
figure;
imshow(im_overlay);
colorbar;
colormap(colormap_matrix); % Apply the custom colormap
clim([0, score_scale]);


%% write output
% imwrite(color_overlay, ['output/' SlideName '_gradient_score_heatmap.png'], 'Alpha', double(alpha));
