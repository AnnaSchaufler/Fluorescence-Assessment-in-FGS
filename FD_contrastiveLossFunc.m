function contrastiveLoss = FD_contrastiveLossFunc(encoderNet, x1, x2, y, margin)
    % x1 und x2 are the data pairs, y is the label, (0 for same class, 1 for different classes)
    z1 = FD_clVAE_sampling(encoderNet, x1);
    z2 = FD_clVAE_sampling(encoderNet, x2);
    
    % Euclidean distance between the latent representations
    dist = sum((z1 - z2).^2, 1);

    % Contrastive Loss (Hinge Loss)
    contrastiveLossBatch = mean((1 - y) .* dist + y .* max(0, margin - sqrt(dist)).^2);
    % Mean over the batch size to obtain a scalar
    contrastiveLoss = mean(contrastiveLossBatch, 'all');
end