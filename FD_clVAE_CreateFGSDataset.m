%% Create Dataset

function [ALA_visible, ref_all, vals_all] = FD_clVAE_CreateFGSDataset(data_path)


%% Load Data, ref 4,5,6,7,9
ref1 = imread(fullfile(data_path, 'P007 (1).png'));
ref2 = imread(fullfile(data_path, 'P007 (2).png'));
ref3 = imread(fullfile(data_path, 'P008 (2).png'));
ref4 = imread(fullfile(data_path, 'P008 (3).png'));
ref5 = imread(fullfile(data_path, 'P008 (1).png'));

ala1 = imread(fullfile(data_path, 'P001 (1).png'));
ala2 = imread(fullfile(data_path, 'P002 (2).png'));
ala3 = imread(fullfile(data_path, 'P002 (1).png'));
ala4 = imread(fullfile(data_path, 'P003 (1).png'));
ala5 = imread(fullfile(data_path, 'P003 (4).png'));

masks_files = dir(strcat(data_path, '\masks ala\*.mat')); 

for q = 1:length(masks_files) 
    mask_name = masks_files(q).name;
    load(fullfile(data_path, 'masks ala', mask_name)); 
end

%% Extract Features
ref_1 = FD_clVAE_ImagePreprocess(ref1);
ref_2 = FD_clVAE_ImagePreprocess(ref2);
ref_3 = FD_clVAE_ImagePreprocess(ref3);
ref_4 = FD_clVAE_ImagePreprocess(ref4);
ref_5 = FD_clVAE_ImagePreprocess(ref5);

ala_1 = FD_clVAE_ImagePreprocess(ala1);
ala_2 = FD_clVAE_ImagePreprocess(ala2);
ala_3 = FD_clVAE_ImagePreprocess(ala3);
ala_4 = FD_clVAE_ImagePreprocess(ala4);
ala_5 = FD_clVAE_ImagePreprocess(ala5);

features_ref1 = FD_clVAE_extractFeatures(ref_1);
features_ref2 = FD_clVAE_extractFeatures(ref_2);
features_ref3 = FD_clVAE_extractFeatures(ref_3);
features_ref4 = FD_clVAE_extractFeatures(ref_4);
features_ref5 = FD_clVAE_extractFeatures(ref_5);

features_ala1 = FD_clVAE_extractFeatures(ala_1);
features_ala2 = FD_clVAE_extractFeatures(ala_2);
features_ala3 = FD_clVAE_extractFeatures(ala_3);
features_ala4 = FD_clVAE_extractFeatures(ala_4);
features_ala5 = FD_clVAE_extractFeatures(ala_5);

for k = 1:size(features_ala1, 3)
    feature_k = features_ala1(:, :, k);
    features_a1(:, k) = feature_k(P001_1_mask);
end

for k = 1:size(features_ala2, 3)
    feature_k = features_ala2(:, :, k);
    features_a2(:, k) = feature_k(P002_2_mask);
end

for k = 1:size(features_ala3, 3)
    feature_k = features_ala3(:, :, k);
    features_a3(:, k) = feature_k(P002_1_mask);
end

for k = 1:size(features_ala4, 3)
    feature_k = features_ala4(:, :, k);
    features_a4(:, k) = feature_k(P003_1_mask);
end

for k = 1:size(features_ala5, 3)
    feature_k = features_ala5(:, :, k);
    features_a5(:, k) = feature_k(P003_4_mask);
end

%% Traning Set
ALA_visible = [features_a1; features_a2; features_a2; features_a4; features_a5];
ref_all = [reshape(features_ref1, [], 10); reshape(features_ref2, [], 10); reshape(features_ref3, [], 10);...
    reshape(features_ref4, [], 10); reshape(features_ref5, [], 10)];
vals_all = [ALA_visible; ref_all];

end