function [H, error, image1] = homography_estimate(im1,im2, folder, matrix)

% Construct the full path to the folder
folder_path = fullfile('../data_set/', folder);

% Add the folder path to the MATLAB search path
addpath(folder_path);
addpath('../lib/');

% Call sift on both images
[image1, descriptors1, locs1] = sift(im1);
[image2, descriptors2, locs2] = sift(im2);

%showkeys(image1, locs1);
%showkeys(image2, locs2);

% Find matches between the images
[desc1 loca1 desc2 loca2 matchings mnb] = match(im1,im2);

% Find matching points
[pts1 pts2] = get_matching_pts(locs1, locs2, matchings);

% Run RANSAC
[H, inliers] = ransacfithomography(pts1, pts2, 0.01);

% Compute the error
error_matrix = H - matrix;
frob_norm_matrix = norm(matrix, 'fro');
error = norm(error_matrix, 'fro') / frob_norm_matrix;

end