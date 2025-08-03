function [zSampled, zMean, zLogvar] = FD_clVAE_sampling(encoderNet, x)
    compressed = forward(encoderNet, x);
    d = size(compressed, 1) / 2;
    zMean = compressed(1:d, :);
    zLogvar = compressed(d+1:end, :);
    
    epsilon = randn(size(zMean), 'like', zMean);
    sigma = exp(0.5 * zLogvar);
    z = epsilon .* sigma + zMean;
    zSampled = dlarray(z, 'CB');
end