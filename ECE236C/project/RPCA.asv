%%
 %  File: RPCA.m
 %  Author: Thomas Kost
 %  
 %  Date: 17 May 2022
 %  
 %  @brief use of RPCA for noise reduction
 %
 clc, clear all, close all;
 %% Run Variables:
 run_CVX = false;
 run_ADMM = true;
 run_PDHG = true;
 run_noise_result = false;
 %% Load Data:
 im_paths = dir(fullfile('CroppedYale\yaleB01\', '*0.pgm'));
 num_im = numel(im_paths);
 im_size = size(imread(fullfile(im_paths(1).folder,im_paths(1).name)));
 dataset = zeros(num_im, im_size(1),im_size(2),'uint8');
 for i = 1:num_im
     dataset(i,:,:) = imread(fullfile(im_paths(i).folder,im_paths(i).name));
 end
 vector_dim = im_size(1)*im_size(2);
 data = cast(reshape(dataset,[num_im, vector_dim]),'double')';
 data_size = size(data);
 disp("Data Read in...");
 
 % Set Lambda
 lambda =1/sqrt(max(data_size));
rand_index = randi([1 num_im],1,1); 

%% CVX
if run_CVX
    disp("Running CVX Optimization...")
    tStart_CVX = tic;
    cvx_begin
    variables L(data_size(1) data_size(2)) S(data_size(1) data_size(2))
    minimize(norm_nuc(L) +lambda*norm(S,1));
    subject to
        L+S==img_dbl
    cvx_end
    tEnd_CVX = toc(tStart_CVX);
end

%% ADMM
if run_ADMM
    disp("Running ADMM Optimization...");
    L = zeros(data_size);
    S = zeros(data_size);
    Z = zeros(data_size);
    X = data;
    tolerance = 1e-7;
    t = data_size(1)*data_size(2)/(4*sum(abs(X(:))));
    count=0;
    tStart_ADMM = tic;

    while(norm(X-L-S,'fro')>tolerance*norm(X,'fro') && count <1000)
        L = prox_nuc(X-S+Z/t,1/t);
        S = prox_l1(X-L+Z/t,lambda/t);
        Z = Z+ t*(X-L-S);
        if ~mod(count,10)
            disp(['ADDM itter: ', num2str(count)]);
        end
        count = count+1;
    end
    tEnd_ADMM = toc(tStart_ADMM);
    disp(['ADMM Algorithm Time: ', num2str(tEnd_ADMM)]);
    % Visualize ADMM
    % Pick a random image
    img = reshape(dataset(rand_index,:,:), im_size);
    L_img = uint8(reshape(L(:,rand_index), im_size));
    S_img = uint8(reshape(S(:,rand_index), im_size));
    RPCA_result = figure();
    subplot(1,3,1);
    imshow(img);
    subplot(1,3,2);
    imshow(L_img);
    subplot(1,3,3);
    imshow(S_img);
end
if run_PDHG
    tStart_PDHG = tic;
    disp("Running ADMM Optimization...");
    L = zeros(data_size);
    L_prev = zeros(data_size);
    Z = zeros(data_size);
    X = data;
    tolerance = 1e-12;
    t = data_size(1)*data_size(2)/(4*sum(abs(X(:))));
    count=0;
    delta=inf;
    while(delta>tolerance && count <1000)
        L_prev = L;
        L = prox_nuc(X-S+Z/t,1/t);
        Z = prox_g_star(Z+ t*(2*L-L_prev),X,t,lambda);
        if ~mod(count,10)
            disp(['PDHG itter: ', num2str(count)]);
        end
        count = count+1;
        primal = norm_nuc(L) +lambda*norm(X-L,1);
        dual = trace(Z'*X); %rest is indicator functions
        delta = norm(L-L_prev,'fro')/norm(L,'fro');
    end
    tEnd_PDHG = toc(tStart_PDHG);
    disp(['PDHG Algorithm Time: ', num2str(tEnd_PDHG)]);
    % Visualize results
    S = X-L;
    img = reshape(dataset(rand_index,:,:), im_size);
    L_img = uint8(reshape(L(:,rand_index), im_size));
    S_img = uint8(reshape(S(:,rand_index), im_size));
    PDHG_result = figure();
    subplot(1,3,1);
    imshow(img);
    subplot(1,3,2);
    imshow(L_img);
    subplot(1,3,3);
    imshow(S_img);
end

if run_noise_result
    ;
end
function prox_x = prox_nuc(X,t)
    [U,S,V] = svd(X,'econ');
    [n,m] = size(S);
    S = max(S-t,0);
    prox_x = U*S*V';
end
function prox_x = prox_l1(X,t)
    prox_x = sign(X).*max(abs(X)-t,zeros);
end
function prox_x = prox_g_star(Y,X,t,lambda)
        prox_x = Y-(t/lambda)*(X +sign((lambda/t)*Y-X).*max(...
            abs((lambda/t)*Y-X)-(lambda^2/t),zeros));
end