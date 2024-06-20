clear all
clc
close all


im1 = "img1.ppm";
folder_array = {'bikes', 'boat', 'graf', 'leuven'};

% Ask user to input folder
folder_input = input('Enter folder name (bikes, boat, graf, leuven): ', 's');

% Check if the input folder is valid
if any(strcmp(folder_input, folder_array))
    fprintf('Computing for folder: %s\n', folder_input);

% Iterate through each folder

%for i = 1:numel(folder_array)
%    folder = folder_array{i};
%    fprintf('Computing for folder: %s\n', folder);


    % Iterate through each image
    for i = 2:6
        im2 = ['img', num2str(i), '.ppm'];
        file_path = ['H1to', num2str(i), 'p'];
        folder_path = fullfile('../data_set/', folder_input, file_path);

    
        % Read the CSV file into a matrix
        matrix = dlmread(folder_path);
        
        % Call the function homography_estimate    
        [H, error, image1] = homography_estimate(im1,im2,folder_input,matrix);

        % Print error of the estimate with respect to the matrix
        fprintf('Error of image1 and image%d with respect to GT is: %f\n', i, error);

        % Transform the matrix with the estimate H
        [est_im, est_H] = imTrans(image1, H);
    
        % Transform the matrix with the real H
        [ground_truth_im, ground_truth_H] = imTrans(image1, matrix);
        
        % Plot the ground truth image
        % Plot the estimate image
        im3 = appendimages(est_im,ground_truth_im);

        % Show a figure with lines joining the accepted matches.
        figure('Position', [10 10 size(im3,2) size(im3,1)]);
        colormap('gray');
        imdisp(im3);

        % Compute the error of the transformed matrices
        error_matrix = est_H - ground_truth_H;
        frob_norm_matrix = norm(ground_truth_H, 'fro');
        error_ = norm(error_matrix, 'fro') / frob_norm_matrix;

        % Print the error
        fprintf('Error between the computed homographies and the ground truth of image1 and image%d is: %f\n', i, error_);

        % Pause to display the image
        %pause;
    end
else
    fprintf('Invalid folder name. Please choose one of the provided folders: bikes, boat, graf, leuven\n');
end

%end