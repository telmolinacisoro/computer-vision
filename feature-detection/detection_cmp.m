%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%              LABORATORY 1
%%%              COMPUTER VISION 2023-2024
%%%              FEATURE DETECTION AND COMPARISON
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

clear all
clc
close all

addpath data

% Load input image
image = imread('sunflower.jpg');

if (size(image,3))
    image = rgb2gray(image);
end

% Benchmarking feature detection methods
methods = {'FAST', 'SIFT', 'SURF', 'KAZE', 'BRISK', 'ORB', 'HARRIS', 'MSER'};
detection_times = zeros(1, numel(methods));

for i = 1:numel(methods)
    method = methods{i};
    
    fprintf('Detecting features using %s...\n', method);
    
    % Start the timer
    tic;
    
    % Compute features using different detection strategies
    switch method
        case 'FAST'
            features_fast = detectFASTFeatures(image, 'MinQuality', 0.12, 'MinContrast', 0.18);
        case 'SIFT'
            features_sift = detectSIFTFeatures(image, 'ContrastThreshold', 0.01999, 'EdgeThreshold', 10.0, 'NumLayersInOctave', 4, 'Sigma', 0.879);
        case 'SURF'
            features_surf = detectSURFFeatures(image, 'MetricThreshold', 600.0, 'NumOctaves', 4, 'NumScaleLevels', 6);
        case 'KAZE'
            features_kaze = detectKAZEFeatures(image, 'Diffusion', "sharpedge", 'Threshold', 0.00059, 'NumOctaves', 4, 'NumScaleLevels', 6);
        case 'BRISK'
            features_brisk = detectBRISKFeatures(image, 'MinContrast', 0.18, 'MinQuality', 0.12, 'NumOctaves', 4);
        case 'ORB'
            features_orb = detectORBFeatures(image, 'ScaleFactor', 1.005, 'NumLevels', 1);
        case 'HARRIS'
            features_harris = detectHarrisFeatures(image, 'MinQuality', 0.03, 'FilterSize', 5);
        case 'MSER'
            features_mser = detectMSERFeatures(image, 'ThresholdDelta', 1.1, 'RegionAreaRange', [30 14000], 'MaxAreaVariation', 0.21);
    end
    
    % Stop the timer and record the elapsed time
    detection_times(i) = toc;
    
    disp(['Detection time for ', method, ': ', num2str(detection_times(i), '%.4f'), ' seconds']);
end


%--------------------------------------------------------------------------
% Visualize a qualitative comparison between feature detection methods
figure(1)
subplot(241)
imshow(image)
hold on
plot(features_fast.Location(:,1),features_fast.Location(:,2),'*r','MarkerSize',4)
hold off
title('FAST')
subplot(242)
imshow(image)
hold on
plot(features_sift.Location(:,1),features_sift.Location(:,2),'*r','MarkerSize',4)
hold off
title('SIFT')
subplot(243)
imshow(image)
hold on
plot(features_surf.Location(:,1),features_surf.Location(:,2),'*r','MarkerSize',4)
hold off
title('SURF')
subplot(244)
imshow(image)
hold on
plot(features_kaze.Location(:,1),features_kaze.Location(:,2),'*r','MarkerSize',4)
hold off
title('KAZE')
subplot(245)
imshow(image)
hold on
plot(features_brisk.Location(:,1),features_brisk.Location(:,2),'*r','MarkerSize',4)
hold off
title('BRISK')
subplot(246)
imshow(image)
hold on
plot(features_orb.Location(:,1),features_orb.Location(:,2),'*r','MarkerSize',4)
hold off
title('ORB')
subplot(247)
imshow(image)
hold on
plot(features_harris.Location(:,1),features_harris.Location(:,2),'*r','MarkerSize',4)
hold off
title('HARRIS')
subplot(248)
imshow(image)
hold on
plot(features_mser.Location(:,1),features_mser.Location(:,2),'*r','MarkerSize',4)
hold off
title('MSER')
%--------------------------------------------------------------------------

