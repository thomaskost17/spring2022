%%
 %  File: CS_images.m
 %  Author: Thomas Kost
 %  
 %  Date: 6 May 2022
 %  
 %  @brief Determination of Sparse sensors given a tailored basis
 %
 clc, clear all, close all;
%% Run Variables:
 run_CVX = false;
 run_ADMM = true;
%% Load Data:
 im_paths = dir(fullfile('CroppedYale\yaleB01\', '*0.pgm'));
 num_im = numel(im_paths);
 im_size = size(imread(fullfile(im_paths(1).folder,im_paths(1).name)))/2;
 dataset = zeros(num_im, im_size(1),im_size(2),'uint8');
 for i = 1:num_im
      im = imread(fullfile(im_paths(i).folder,im_paths(i).name));
      dataset(i,:,:) = im(1:2:end, 1:2:end);
 end
 disp("Data Read in...");
 % Pick a random image
rand_index = randi([0 num_im],1,1); 

%% Pick Example and remove from basis
figure();
subplot(1,2,1);
vector_dim = im_size(1)*im_size(2);
orig_im_vec = reshape(dataset(1,:,:), (vector_dim,1));
orig_im = reshape(dataset(1,:,:), im_size);
dataset = dataset(2:end,:,:);
imshow(orig_im);
% Add Salt and Pepper Noise
p = 0.7;
noise_probs = rand(im_size);
noisy_image = orig_im - uint8(noise_probs<(p/2)).*orig_im  + (255-orig_im).*uint8(noise_probs>(1-p/2));
subplot(1,2,2);
imshow(noisy_image)

% Create Variables for optimization
y = reshape(noisy_image, [vector_dim,1]);
data = cast(reshape(dataset,[num_im-1, vector_dim]),'double')';
[U,S,V] = svd(data);
disp("SVD complete...");
r = 35;
psi = U(:,1:r);
% S_tilde = S(1:r,1:r);
% V_tilde = V(:,1:r)';
% psi = U_tilde;
%data_approx = U_tilde*S_tilde*V_tilde;
p = r;
shape_C = [p,vector_dim];
data_size = size(psi);

%% Find C via CVX
if run_CVX
    tStart_CVX = tic;
    cvx_begin 
    variable C(shape_C(1), shape_C(2));
    minimize(norm(C*psi,'fro') + norm(C,1));
    subject to
         C*psi - eye(p) == semidefinite(p);
        %C*ones(num_im-1,1) == ones(p,1)
    cvx_end
    tEnd_CVX = toc(tStart_CVX);
% Visualize reconstruction
%C = C.*(abs(C)>1e-3);
theta = C*psi;
end


%% Find C via ADMM
if run_ADMM
    tStart_ADMM = tic;
    disp("Running ADMM Optimization...");
    %Initialize variables
    Theta = randn(shape_C(1));
    Z = randn(shape_C(1));
    C = randn(shape_C);
    B = randn(shape_C);
    Y = randn(shape_C);
    
    %constants 
    gamma = 1e-4;
    t = data_size(1)*data_size(2)/(4*sum(abs(psi(:))));
    lambda = 1/sqrt(max(data_size));
    tolerance = 1e-7;
    tStart_MP = tic;
    H = pinv(psi*psi'+t*eye(data_size(1)));
    tEnd_MP = toc(tStart_MP);
    disp(['MP inverse complete...(', num2str(tEnd_MP), ' seconds)']);
    count = 0;
    while((norm(Theta-C*psi,'fro')> tolerance*norm(C*psi,'fro') ||...
            norm(B-C,'fro') > tolerance*norm(C,'fro'))...
            && count <1000)
        C = (t*Theta*psi'+t*B-Z*psi'-Y)*H/(2+t);
        Theta = P_posdef(C*psi +Z/t,gamma);
        B = prox_l1(C+Y/t,1/t);
        Z = Z+t*(C*psi-Theta);
        Y = Y+t*(C-B);
        if ~mod(count,10)
            disp(['ADDM itter: ', num2str(count)]);
        end
        count = count+1;
        
    end
    
    tEnd_ADMM = toc(tStart_ADMM);
    disp(['ADMM Algorithm Time: ', num2str(tEnd_ADMM)]);

    % Visualize results
    [M,I] = max(C);
    C_prime = zeros(shape_C);
    index = sub2ind(shape_C, [1:r],I);
    C_prime(index)=1;
    Theta_prime = C_prime*psi;
    measurement = orig_im_vec(I);
    x = Theta_prime\cast(measurement,'double');
    face_recon = psi*x;
    face_recon_scale = face_recon +abs(min(min(face_recon)));
    face_recon_scale = face_recon_scale*(255/max(max(face_recon_scale)));
    red_ch = orig_im_vec;
    green_ch = orig_im_vec;
    red_ch(I) = 255;
    green_ch(I) =0;
    rgb_im = cat(3,reshape(red_ch, im_size), reshape(green_ch, im_size), reshape(green_ch,im_size));
    figure;
    subplot(1,3,1)
    imshow(orig_im);
    subplot(1,3,2)
    imshow(rgb_im);
    subplot(1,3,3)
    imshow(uint8(face_recon_scale));
end


function proj_x = P_posdef(X,gamma)
    [V,D] = eig(X);
    D = diag(max(diag(D),gamma));
    proj_x = V*D*pinv(V);
end
function prox_x = prox_l1(X,t)
    prox_x = sign(X).*max(abs(X)-t,zeros);
end