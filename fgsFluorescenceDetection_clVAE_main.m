%% Contrastive Variational Autoencoder for Fluorescence Detection in 
%% Fluorescence Guided Neurosurgery Imaging - Main

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Script Name : fgsFluorescenceDetection_clVAE_main.m
%
% Description : A training sequence for a contarstive learning Variational 
%               Autoencoder for the detection of fluorescence in 
%               intraoperative neurosurgical images is performed within this script. 
%               The sequence includes data preprocessing, composition of 
%               training and validation data sets, initialisation of the 
%               cl VAE and the training, as well as the training implementation.
%
% Author      : Anna Schaufler
% Date        : 20.07.2025
% Version     : 1.0
%
% Notice      : This script requires image and annotation data from the 
%               publicly available FGS imaging dataset 
%               (https://doi.org/10.5281/zenodo.15260349). Scripts, images and 
%               annotation masks need to be organized in a particular file 
%               structure. The required folder structure is described 
%               in the README file.
%
%               To change the beta factor of the trained variational autoencoder, 
%               change the variable beta_max in the section "Initialize VAE net."
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

script_fullpath = mfilename('fullpath');
[script_dir, ~, ~] = fileparts(script_fullpath);
raw_data_path = fullfile(script_dir, 'Images and Masks');

% preprocess images and generate pixelwise input data
[ALA_visible, ref_all, vals_all] = FD_clVAE_CreateFGSDataset(raw_data_path);

idx = randperm(length(ref_all(:,1)), 550e3); %60k datapoints for from image background for the normal class 
normalData = ref_all(idx(1:50e3),:); %10k of the normal class data for pairing with anomalies
data = ref_all(idx(50e3+1:end),:); %50k of the normal class data for unsupervised learning VAE input
% split training and validation data
cv = cvpartition(length(data), 'HoldOut', 0.2);
trainData = data(training(cv), :);
testData = data(test(cv), :);

% anomaly data for contrastive learning
idx = randperm(length(ALA_visible(:,1)), 50e3);
anomalies = ALA_visible(idx,:);

% generate normal-anomaly pairs
pairs = [normalData; anomalies];
labels = [zeros(size(normalData,1), 1); ones(size(anomalies,1), 1)]; % 0 fot normal, 1 for anomaly

disp('finished data initialization');
%% Initialize VAE net
% Net parameters
inputSize = 10; 
latentDim = 3;

% Encoder net
encoderLG = layerGraph([
    featureInputLayer(inputSize, 'Normalization', 'none', 'Name', 'input_encoder')
    tanhLayer("Name", 'tanh1')
    fullyConnectedLayer(2 * latentDim, 'Name', 'fc_encoder')
    ]);

% Decoder-Netzwerk
decoderLG = layerGraph([
    featureInputLayer(latentDim, 'Normalization', 'none', 'Name', 'input_decoder')
    tanhLayer("Name",'tanh2')
    fullyConnectedLayer(inputSize, 'Name', 'fc_decoder')
    ]);

% Convert net to dlnetwork
encoderBeta1_fgs = dlnetwork(encoderLG);
decoderBeta1_fgs = dlnetwork(decoderLG);

% Training options
executionEnvironment = "auto";
numEpochs = 1000;
miniBatchSize = 256;
lr = 1e-4; % learning rate
numIterations = floor(cv.TrainSize / miniBatchSize);
beta_max = 1; 
margin = 3;

% initialize the average gradient and the average gradient-square 
% decay rates of adam optimizer with empty arrays
avgGradientsEncoder = [];
avgGradientsSquaredEncoder = [];
avgGradientsDecoder = [];
avgGradientsSquaredDecoder = [];

% initialize the evidence lower bound, reconstruction loss,
% kullback-Leiber-Divergence loss
elbo = zeros(numEpochs,1);
rec_loss = zeros(numEpochs,1);
kl_loss = zeros(numEpochs,1);

% Initialize training monitor
monitor = trainingProgressMonitor( ...
    Metrics= ["ELBO", "Reconstruction", "KL"],...
    Info="Epoch", ...
    XLabel="Iteration");

% Initialize training loop
epoch = 0;
iteration = 0;

% Initialize factor for cyclic annealing VAE beta factor
block_len = 39.05e3;
n_blocks = 40;
e = ones(block_len * n_blocks, 1);

for k = 0:n_blocks-1
    idx = k * block_len + 1 : (k + 1) * block_len;
    if mod(k, 2) == 0
        e(idx) = 1:block_len;       % ramp
    else
        e(idx) = block_len;         % constant
    end
end

disp('finished training initialization');

%%%%%%%%%% TRAINING %%%%%%%%%%
while epoch < numEpochs && ~monitor.Stop
    tic;
    epoch = epoch + 1;
    lr = 0.0010023*exp(-0.002305*epoch); %exponentially decaying learning rate

    % Random order of data for Mini-Batch-Training
    idx = randperm(cv.TrainSize);
    trainData = trainData(idx, :);

    for i = 1:numIterations
        iteration = iteration + 1;
            
        beta = beta_max./(1 + exp(-3e-4*(e(iteration)-15e3)));
        idx = (i-1) * miniBatchSize + 1:i * miniBatchSize;
        XBatch = trainData(idx, :);
        XBatch = dlarray(single(XBatch)', 'CB');

        % Anomaly-Pairing
        posPairsIdx = randperm(length(normalData), miniBatchSize);
        negPairsIdx = randperm(length(anomalies), miniBatchSize);
        posBatch = normalData(posPairsIdx, :);
        negBatch = anomalies(negPairsIdx, :);

        XPosBatch = dlarray(single(posBatch)', 'CB');
        XNegBatch = dlarray(single(negBatch)', 'CB');


        % transfer to GPU if possible
        if (executionEnvironment == "auto" && canUseGPU) || executionEnvironment == "gpu"
            XBatch = gpuArray(XBatch);
            XPosBatch = gpuArray(XPosBatch);
            XNegBatch = gpuArray(XNegBatch);
        end

        %% Calculate gradients and update parameters
        
        [infGrad, genGrad, contrastiveGrad] = dlfeval(@FD_clVAE_combinedGradients, encoderBeta1_fgs, decoderBeta1_fgs, XBatch, XPosBatch, XNegBatch, beta, margin);
        
        % Update the decoder
        [decoderBeta1_fgs.Learnables, avgGradientsDecoder, avgGradientsSquaredDecoder] = ...
            adamupdate(decoderBeta1_fgs.Learnables, genGrad, avgGradientsDecoder, avgGradientsSquaredDecoder, i, lr);
        
        combinedGradients = infGrad;  

        for j = 1:height(infGrad)
            combinedGradients.Value{j} = infGrad.Value{j} + contrastiveGrad.Value{j};
        end

        % Update des Encoders
        [encoderBeta1_fgs.Learnables, avgGradientsEncoder, avgGradientsSquaredEncoder] = ...
            adamupdate(encoderBeta1_fgs.Learnables, combinedGradients, avgGradientsEncoder, avgGradientsSquaredEncoder, i, lr);
    end
    elapsedTime = toc;

    % Test after each epoch
    XTest = dlarray(single(testData'), 'CB');
    if (executionEnvironment == "auto" && canUseGPU) || executionEnvironment == "gpu"
        XTest = gpuArray(XTest);
    end
    [z, zMean, zLogvar] = FD_clVAE_sampling(encoderBeta1_fgs, XTest);
    xPred = sigmoid(forward(decoderBeta1_fgs, z));
    [el, rc, kl] = FD_clVAE_ELBOloss(XTest, xPred, zMean, zLogvar, beta);
    rec_loss(epoch) = rc;
    elbo(epoch) = el;
    kl_loss(epoch) = kl;
    recordMetrics(monitor,epoch,ELBO=el,Reconstruction=rc, KL=kl);
    updateInfo(monitor,Epoch=epoch + " of " + numEpochs);
    monitor.Progress = iteration/(numIterations*numEpochs)*100;
end

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%  TEST  %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

%% Decision Threshold Calculation

latentTest_all = dlarray(single(vals_all)', 'CB');
[z, zMean, zLogvar] = FD_clVAE_sampling(encoderBeta1_fgs, latentTest_all);

klDivergence = -0.5 * sum(1 + zLogvar - zMean.^2 - exp(zLogvar), 1);
klDivergence = gather(extractdata(klDivergence));  
% figure, scatter (1:length(klDivergence), klDivergence)

% Calculate the maximum and standard deviation of the training data normal 
% class to use it for decision threshold determination
diver4th = klDivergence(length(ALA_visible) + 1:end);
diver4th = rmoutliers(diver4th, "mean");
div_std = std(diver4th);
div_max = max(diver4th);
upper_th = 1;
lower_th = div_max + 1*div_std;

%% Application of Model on Test Images

% Specificity test on non-FGS images
% load test image 
test_img = imread(fullfile(raw_data_path, 'P008 (3).png'));
test_img = FD_clVAE_ImagePreprocess(test_img);
test_feat = FD_clVAE_extractFeatures(test_img);
test_feat = reshape(test_feat, [], 10);
latentTest_all = dlarray(single(test_feat)', 'CB');
[z, zMean, zLogvar] = FD_clVAE_sampling(encoderBeta1_fgs, latentTest_all);

% Calculate test image Kullback-Leiber-Divergence per Pixel
klDivergence = -0.5 * sum(1 + zLogvar - zMean.^2 - exp(zLogvar), 1);
klDivergence = gather(extractdata(klDivergence)); 
% Identify False Positives in non-FGS-images 
test_img_div = klDivergence/upper_th;
test_img_div(test_img_div <= lower_th) = 0;
fp = numel(find(test_img_div));
% Overall Specificity
spec_oa = (numel(test_img_div)-fp)/numel(test_img_div);
% Specificity excluding black [0 0 0] pixels
non_black_pixel = numel(find(rgb2gray(test_img)> 0));
spec_bn = (non_black_pixel-fp)/non_black_pixel;

% Reconstruct KL-divergence based image
[m, n, ~] = size(test_img);
div_image = reshape(test_img_div, m, n);
figure;
imshow(div_image, []);
colormap(gray);
colorbar;
title('Reconstructed Test Image based on KL-Divergence Scores');

% Binarisieren des Bildes und Umrandung finden
BW = imbinarize(div_image, 0);
boundaries = bwboundaries(BW);

figure, imshow(test_img)
hold on
for k=1:length(boundaries)
   b = boundaries{k};
   plot(b(:,2),b(:,1),"g",LineWidth=2);
end


%% Intensity Quantification Visualization

% Load and process FGS image
test_img = imread(fullfile(raw_data_path, 'P004 (1).png'));
test_img = FD_clVAE_ImagePreprocess(test_img);
test_feat = FD_clVAE_extractFeatures(test_img);
test_feat = reshape(test_feat, [], 10);
latentTest_all = dlarray(single(test_feat)', 'CB');
[z, zMean, zLogvar] = FD_clVAE_sampling(encoderBeta1_fgs, latentTest_all);

% Calculate test image Kullback-Leiber-Divergence per Pixel
klDivergence = -0.5 * sum(1 + zLogvar - zMean.^2 - exp(zLogvar), 1);
klDivergence = gather(extractdata(klDivergence));

% Visualization initialization
x = test_feat(:,1); % Red channel value
y = test_feat(:,6); % proportional green channel value
color_val = 0.0327 + -1.009.*x + 0.6343.*y + 46.99.*x.^2 + -93.46.*x.*y;

data_color_val = reshape(color_val, m, n); 
data_color_val(data_color_val < 0) = 0; 
data_color_val(data_color_val > 10) = 11; 
% figure, imshow(data_color_val);

% Jet Colormap 
cmap = hot(256); 
cmap = cmap(25:end, :);
% Define Edge Values
value_edges = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, Inf];
    ... 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, Inf]
    %...21, 22, 23, 24, 25, 26, 27, 28, 29, 30, Inf];
% Custom Colormap
num_colors = length(value_edges) - 1; % Number of Colors
custom_cmap = zeros(num_colors, 3); 
for i = 1:num_colors
    if value_edges(i) < 10 
        color_index = round((i - 1) * (size(cmap, 1) / num_colors)) + 1;
        custom_cmap(i, :) = cmap(color_index, :);
    elseif value_edges(i) >= 10
        custom_cmap(i, :) = [1.0000    1.0000    0.7500]; % dark red for every value > 10
    elseif value_edges(i) <= 1
        custom_cmap(i, :) = [0.2500         0         0];
    end
end

figure('Position', [1, 1, 1920, 1080]);
axes('Position', [0, 0, 1, 1]);
imsc = imagesc(data_color_val); 
axis off; 
colormap(custom_cmap);           % Apply Custom Colormap
frame = getframe(gca);           
rgbImage = imresize(frame.cdata, [1080, 1920]);
% colorbar;                        
clim([0 10]);                   

% Reconstruct KL-divergence based image
test_img_div = klDivergence/upper_th;
test_img_div(test_img_div <= lower_th) = 0;
[m, n, ~] = size(test_img);
div_image = reshape(test_img_div, m, n);
figure;
imshow(div_image, []);
colormap(gray);
colorbar;
title('Reconstructed Test Image based on KL-Divergence Scores');
% Binarisieren des Bildes und Umrandung finden
BW = imbinarize(div_image, 0);
boundaries = bwboundaries(BW);

color_ml = [197 198 170]/255;
img_proc = test_img;
for c = 1:3
    channel = im2double(rgbImage(:,:,c));
    temp = img_proc(:,:,c);
    temp(BW) = channel(BW);
    img_proc(:,:,c) = temp;
end
figure, imshow(img_proc)
hold on
for k=1:length(boundaries)
   b = boundaries{k};
   plot(b(:,2),b(:,1),"Color",color_ml,LineWidth=2);
end

figure, imshowpair(test_img, img_proc, 'montage')

