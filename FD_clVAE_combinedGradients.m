function [infGrad, genGrad, contrastiveGrad] = FD_clVAE_combinedGradients(encoderNet, decoderNet, x, xPos, xNeg, beta, margin)
    % VAE Loss
    [z, zMean, zLogvar] = FD_clVAE_sampling(encoderNet, x);
    xPred = sigmoid(forward(decoderNet, z));
    vaeLoss = FD_clVAE_ELBOloss(x, xPred, zMean, zLogvar, beta);
    
    % Contrastive Loss for positive pair
    contrastiveLossPos = FD_contrastiveLossFunc(encoderNet, x, xPos, zeros(size(x,2), 1), margin);
    
    % Contrastive Loss for negative pair
    contrastiveLossNeg = FD_contrastiveLossFunc(encoderNet, x, xNeg, ones(size(x,2), 1), margin);
    
    % Combine the losses
    totalContrastiveLoss = contrastiveLossPos + contrastiveLossNeg;
    
    % Gradients
    [genGrad, infGrad] = dlgradient(vaeLoss, decoderNet.Learnables, encoderNet.Learnables);
    contrastiveGrad = dlgradient(totalContrastiveLoss, encoderNet.Learnables);
end