
function [elbo, rcLoss, klLoss]  = FD_clVAE_ELBOloss(x, xPred, zMean, zLogvar, beta)
    reconstructionLoss = sum((xPred - x).^2, 1);
    KL = -0.5 * sum(1 + zLogvar - zMean.^2 - exp(zLogvar), 1);
    rcLoss = mean(reconstructionLoss);
    klLoss = mean(beta*KL);
    elbo = rcLoss + klLoss;
end