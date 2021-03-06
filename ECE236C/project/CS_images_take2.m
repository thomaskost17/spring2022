%%
 %  File: CS_images_take2.m
 %  Author: Thomas Kost
 %  
 %  Date: 24 May 2022
 %  
 %  @brief Determination of Sparse sensors given a tailored basis
 %
 clc, clear all, close all;
%% Run Variables:
 run_CVX = false;
 run_ADMM = true;
 run_PDHG = true;
 
 %% Load Data:
 im_paths = dir(fullfile('CroppedYale\yaleB01\', '*0.pgm'));
 num_im = numel(im_paths);
 im_size = size(imread(fullfile(im_paths(1).folder,im_paths(1).name)))/2; %and downsample
 dataset = zeros(num_im, im_size(1),im_size(2),'uint8');
 for i = 1:num_im
      im = imread(fullfile(im_paths(i).folder,im_paths(i).name));
      dataset(i,:,:) = im(1:2:end, 1:2:end);
 end
 disp("Data Read in...");
 
 %% Set sim parameters
 vector_dim = im_size(1)*im_size(2);
 A = eye(vector_dim);
data = cast(reshape(dataset,[num_im, vector_dim]),'double')';
[U,S,V] = svd(data);
r = 21;
psi = zeros(vector_dim,vector_dim, r);
for i= 1:r
    psi(:,:,i) = U(:,i)*U(:,i)';
end
data_size = size(psi);
disp("SVD complete...");
lambda = 1/sqrt(max(data_size));
 %% CVX optimization
 if run_CVX
     disp("Running CVX Optimization...");
     sum_var = zeros(data_size(1:2));
     cvx_begin
        variable z(r);
        for i = 1:r
            sum_var = sum_var + z(i)*psi(:,:,i);
        end
        minimize(-log(det(sum_var)) + lambda*norm(z,1));
        subject to
            ones(1,r)*z == r;
            0<= z<=1;
     cvx_end
 
 % Visualize CVX Results
 
 end
 %% ADMM
 if run_ADMM
     disp("Running ADMM Optimization...");
 end
 
 %% PDHG
 if run_PDHG
     disp("Running PDHG Optimiztion...")
 end
 