function features = FD_clVAE_extractFeatures(img)

newMin = 0;
newMax = 1;

redChannel = double(img(:, :, 1));
greenChannel = double(img(:, :, 2));
blueChannel = double(img(:, :, 3));

gray = double(rgb2gray(img));

% relative channel proportions
rd = redChannel./(redChannel + greenChannel + blueChannel);
rd(isnan(rd)) = 0;
gr = greenChannel./(redChannel + greenChannel + blueChannel);
gr(isnan(gr)) = 0;
bl = blueChannel./(redChannel + greenChannel + blueChannel);
bl(isnan(bl)) = 0;

% proportional channel differences
BR = (blueChannel - redChannel)./(blueChannel + redChannel);
BR(isnan(BR)) = 0;
originalMin = -1;
originalMax = 1;
nrmlz = @(x) ((x - originalMin) / (originalMax - originalMin)) * (newMax - newMin) + newMin;
BR = nrmlz(BR);
BG = (blueChannel - greenChannel)./(blueChannel + greenChannel);
BG(isnan(BG)) = 0;
originalMin = -1;
originalMax = 1;
nrmlz = @(x) ((x - originalMin) / (originalMax - originalMin)) * (newMax - newMin) + newMin;
BG = nrmlz(BG);
RG = (redChannel - greenChannel)./(redChannel + greenChannel);
RG(isnan(RG)) = 0;
originalMin = -1;
originalMax = 1;
nrmlz = @(x) ((x - originalMin) / (originalMax - originalMin)) * (newMax - newMin) + newMin;
RG = nrmlz(RG);

[rows, cols, ~] = size(img);
features = zeros(rows, cols, 10);

for i = 1:rows
    for j = 1:cols
        features(i,j,1) = redChannel(i,j);
        features(i,j,2) = blueChannel(i,j);
        features(i,j,3) = greenChannel(i,j);
        features(i,j,4) = gray(i,j);
        features(i,j,5) = rd(i,j);
        features(i,j,6) = gr(i,j);
        features(i,j,7) = bl(i,j);
        features(i,j,8) = RG(i,j);
        features(i,j,9) = BR(i,j);
        features(i,j,10) = BG(i,j);
    end
end

end