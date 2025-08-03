%% Contrastive Variational Autoencoder for Synthetic Smaples Fluorescence 
%% Detection - Main

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Script Name : synthFluorescenceDetection_clVAE_main.m
%
% Description : A training sequence for a contarstive learning Variational 
%               Autoencoder for the detection of fluorescence in images of 
%               synthetic fluorescent samples is performed within this script. 
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
%               change the variable beta_max in the section “Initialize VAE net.”
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

%% Load and Prepare Data
script_fullpath = mfilename('fullpath');
[script_dir, ~, ~] = fileparts(script_fullpath);
raw_data_path = fullfile(script_dir, 'Images and Masks');

% preprocess images and generate pixelwise input data
[vals_PPIX_all, vals_PPIX_visible, vals_background_all, vals_all, vals_PPIX_all_test,...
    vals_background_all_test, vals_all_test, labs_test, features_region3]...
    = FD_clVAE_CreateDataset(raw_data_path);

idx = randperm(length(features_region3(:,1)), 60e3); %60k datapoints for from image background for the normal class 
normalData = features_region3(idx(1:10e3),:); %10k of the normal class data for pairing with anomalies
data = features_region3(idx(10e3+1:end),:); %50k of the normal class data for unsupervised learning VAE input
% split training and validation data
cv = cvpartition(length(data), 'HoldOut', 0.2);
trainData = data(training(cv), :);
testData = data(test(cv), :);

% anomaly data for contrastive learning
idx = randperm(length(vals_PPIX_visible(:,1)), 10e3);
anomalies = vals_PPIX_visible(idx,:);

% generate normal-anomaly pairs
pairs = [normalData; anomalies];
labels = [zeros(size(normalData,1), 1); ones(size(anomalies,1), 1)]; % 0 fot normal, 1 for anomaly

disp('data initialization done');
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
encoderBeta1_synth = dlnetwork(encoderLG);
decoderBeta1_synth = dlnetwork(decoderLG);

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

% Initialize factor for cyclic annealing beta
block_len = 3900;
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

disp('training initialization done');

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
            
        beta = beta_max./(1 + exp(-2e-3*(e(iteration)-1.5e3)));
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
        
        [infGrad, genGrad, contrastiveGrad] = dlfeval(@FD_clVAE_combinedGradients, encoderBeta1_synth, decoderBeta1_synth, XBatch, XPosBatch, XNegBatch, beta, margin);
        
        % Update the decoder
        [decoderBeta1_synth.Learnables, avgGradientsDecoder, avgGradientsSquaredDecoder] = ...
            adamupdate(decoderBeta1_synth.Learnables, genGrad, avgGradientsDecoder, avgGradientsSquaredDecoder, i, lr);
        
        combinedGradients = infGrad;  

        for j = 1:height(infGrad)
            combinedGradients.Value{j} = infGrad.Value{j} + contrastiveGrad.Value{j};
        end

        % Update des Encoders
        [encoderBeta1_synth.Learnables, avgGradientsEncoder, avgGradientsSquaredEncoder] = ...
            adamupdate(encoderBeta1_synth.Learnables, combinedGradients, avgGradientsEncoder, avgGradientsSquaredEncoder, i, lr);
    end
    elapsedTime = toc;

    % Test after each epoch
    XTest = dlarray(single(testData'), 'CB');
    if (executionEnvironment == "auto" && canUseGPU) || executionEnvironment == "gpu"
        XTest = gpuArray(XTest);
    end
    [z, zMean, zLogvar] = FD_clVAE_sampling(encoderBeta1_synth, XTest);
    xPred = sigmoid(forward(decoderBeta1_synth, z));
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

% Using encoder network to generate Kullback-Leiber-Divergence per data
% point
test_data = dlarray(single(vals_all_test)', 'CB');
[z, zMean, zLogvar] = FD_clVAE_sampling(encoderBeta1_synth, test_data);

klDivergence = -0.5 * sum(1 + zLogvar - zMean.^2 - exp(zLogvar), 1);
klDivergence = gather(extractdata(klDivergence));

% Plot ROC curve
labs_test_bin = [ones(size(vals_PPIX_all_test, 1), 1); zeros(size(vals_background_all_test, 1), 1)];
[X_beta1,Y_beta1,T_beta1,AUC_beta1,OPT_beta1] = ...
    perfcurve(labs_test_bin, klDivergence, 1);

clr_1 = [198 1 86]/255; 
clr_2 = [0 51 108]/255;
clr_3 = [0 8 49]/255;

figure, plot (X_beta1, Y_beta1, 'Color', clr_1, 'LineWidth', 1.5);
hold on
plot(OPT_beta1(1),OPT_beta1(2),'o', "MarkerFaceColor", clr_1, "MarkerEdgeColor", "none")
% plot (X_beta2, Y_beta2, "Color", clr_2, 'LineWidth', 1.5);
% plot(OPT_beta2(1),OPT_beta2(2),'square', "MarkerFaceColor", clr_2, "MarkerEdgeColor", "none")
% plot (X_beta3, Y_beta3, "Color", clr_3, 'LineWidth', 1.5);
% plot(OPT_beta3(1),OPT_beta3(2),'<', "MarkerFaceColor", clr_3, "MarkerEdgeColor", "none")
line([0 1], [0 1], "LineStyle", "--", "Color", [0.5 0.5 0.5])
xlabel('False positive rate') 
ylabel('True positive rate')