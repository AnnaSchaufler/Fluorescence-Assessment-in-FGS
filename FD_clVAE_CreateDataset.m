%% Create Dataset

function [vals_PPIX_all, vals_PPIX_visible, vals_background_all, vals_all, ...
    vals_PPIX_all_test, vals_background_all_test, vals_all_test, labs_test, ...
    features_region3, tsne_data, tsne_labs] = FD_clVAE_CreateDataset(data_path)

%% Load Data
liquid1 = imread(fullfile(data_path, 'liquid1.jpg'));
liquid2 = imread(fullfile(data_path, 'liquid2.jpg'));
liquid3 = imread(fullfile(data_path, 'liquid3.jpg'));
masks_files = dir(strcat(data_path, '\masks liquid\*.mat'));
test_masks_files = dir(strcat(data_path, '\masks test\*.mat')); 

for q = 1:length(masks_files) 
    load(fullfile(data_path, 'masks liquid', masks_files(q).name)); 
end
for q = 1:length(test_masks_files) 
    load(fullfile(data_path, 'masks test', test_masks_files(q).name)); 
end

%% Extract Features
img_1 = FD_clVAE_ImagePreprocess(liquid1);
img_2 = FD_clVAE_ImagePreprocess(liquid2);
img_3 = FD_clVAE_ImagePreprocess(liquid3);

features_1 = FD_clVAE_extractFeatures(img_1);
features_2 = FD_clVAE_extractFeatures(img_2);
features_3 = FD_clVAE_extractFeatures(img_3);

for k = 1:size(features_1, 3)
    feature_k = features_1(:, :, k);
    features_5ug(:, k) = feature_k(mask_5ug);
    features_2ug(:, k) = feature_k(mask_2ug);
    features_1ug(:, k) = feature_k(mask_1ug);
    features_DMSO1(:,k) = feature_k(mask_DMSO1);
    features_BG1(:,k) = feature_k(mask_BG1);
    features_stripes1(:,k) = feature_k(mask_stripes1);
    features_5ug_test(:, k) = feature_k(mask_5ug_test);
    features_2ug_test(:, k) = feature_k(mask_2ug_test);
    features_1ug_test(:, k) = feature_k(mask_1ug_test);
    features_DMSO1_test(:,k) = feature_k(mask_DMSO1_test);
    features_BG1_test(:,k) = feature_k(mask_BG1_test);
    features_stripes1_test(:,k) = feature_k(mask_stripes1_test);
end

for k = 1:size(features_2, 3)
    feature_k = features_2(:, :, k);
    features_05ug(:, k) = feature_k(mask_05ug);
    features_02ug(:, k) = feature_k(mask_02ug);
    features_01ug(:, k) = feature_k(mask_01ug);
    features_DMSO2(:,k) = feature_k(mask_DMSO2);
    features_BG2(:,k) = feature_k(mask_BG2);
    features_stripes2(:,k) = feature_k(mask_stripes2);
    features_05ug_test(:, k) = feature_k(mask_05ug_test);
    features_02ug_test(:, k) = feature_k(mask_02ug_test);
    features_01ug_test(:, k) = feature_k(mask_01ug_test);
    features_DMSO2_test(:,k) = feature_k(mask_DMSO2_test);
    features_BG2_test(:,k) = feature_k(mask_BG2_test);
    features_stripes2_test(:,k) = feature_k(mask_stripes2_test);
end

for k = 1:size(features_3, 3)
    feature_k = features_3(:, :, k);
    features_005ug(:, k) = feature_k(mask_005ug);
    features_0025ug(:, k) = feature_k(mask_0025ug);
    features_001ug(:, k) = feature_k(mask_001ug);
    features_DMSO3(:,k) = feature_k(mask_DMSO3);
    features_BG3(:,k) = feature_k(mask_BG3);
    features_stripes3(:,k) = feature_k(mask_stripes3);
    features_005ug_test(:, k) = feature_k(mask_005ug_test);
    features_0025ug_test(:, k) = feature_k(mask_0025ug_test);
    features_001ug_test(:, k) = feature_k(mask_001ug_test);
    features_DMSO3_test(:,k) = feature_k(mask_DMSO3_test);
    features_BG3_test(:,k) = feature_k(mask_BG3_test);
    features_stripes3_test(:,k) = feature_k(mask_stripes3_test);
    features_region3(:,k) = feature_k(mask_region3);
end

%% TSNE set
tsne_data = [features_5ug_test; features_2ug_test; features_1ug_test;...
    features_05ug_test; features_02ug_test; features_01ug_test;...
    features_005ug_test; features_0025ug_test; features_001ug_test;...
    features_DMSO1_test; features_DMSO2_test; features_DMSO3_test];
l1 = length(features_5ug_test);
l2 = l1 + length(features_2ug_test);
l3 = l2 + length(features_1ug_test);
l4 = l3 + length(features_05ug_test);
l5 = l4 + length(features_02ug_test); 
l6 = l5 + length(features_01ug_test);
l7 = l6 + length(features_005ug_test);
l8 = l7 + length(features_0025ug_test);
l9 = l8 + length(features_001ug_test);
tsne_labs = zeros(length(tsne_data), 1);
tsne_labs(1:l1) = 1;
tsne_labs(l1+1:l2) = 2;
tsne_labs(l2+1:l3) = 3;
tsne_labs(l3+1:l4) = 4;
tsne_labs(l4+1:l5) = 5;
tsne_labs(l5+1:l6) = 6;
tsne_labs(l6+1:l7) = 7;
tsne_labs(l7+1:l8) = 8;
tsne_labs(l8+1:l9) = 9;

%% Traning Set
vals_PPIX_all = [features_5ug; features_2ug; features_1ug;...
    features_05ug; features_02ug; features_01ug;...
    features_005ug; features_0025ug; features_001ug];
vals_PPIX_visible = [features_5ug; features_2ug; features_1ug];
vals_background_all = [features_DMSO1; features_DMSO2; features_DMSO3;...
    features_BG1; features_BG2; features_BG3; ...
    features_stripes1; features_stripes2; features_stripes3];
vals_all = [vals_PPIX_all; vals_background_all];

% % labels vector
% labs_all = zeros(length(vals_all), 1);
% labs_all(1:length(vals_PPIX_all)) = 1;

%% Test Set
vals_PPIX_all_test = [features_5ug_test; features_2ug_test; features_1ug_test;...
    features_05ug_test; features_02ug_test; features_01ug_test;...
    features_005ug_test; features_0025ug_test; features_001ug_test];
vals_background_all_test = [features_DMSO1_test; features_DMSO2_test; features_DMSO3_test;...
    features_BG1_test; features_BG2_test; features_BG3_test; ...
    features_stripes1_test; features_stripes2_test; features_stripes3_test];
vals_all_test = [vals_PPIX_all_test; vals_background_all_test];

% test data labels vector
labs_test = zeros(length(vals_all_test), 1);
labs_test(1:l1) = 1;
labs_test(l1+1:l2) = 2;
labs_test(l2+1:l3) = 3;
labs_test(l3+1:l4) = 4;
labs_test(l4+1:l5) = 5;
labs_test(l5+1:l6) = 6;
labs_test(l6+1:l7) = 7;
labs_test(l7+1:l8) = 8;
labs_test(l8+1:l9) = 9;

end