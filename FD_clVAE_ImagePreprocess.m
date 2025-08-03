function [img_pp] = FD_clVAE_ImagePreprocess(img)
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Identification and interpolation of block artifacts caused by image
% compression.
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

r_channel = double(img(:,:,1))./255;
g_channel = double(img(:,:,2))./255;
b_channel = double(img(:,:,3))./255;

hsv = rgb2hsv(img);
channel1Min = 0.994;
channel1Max = 0.436;
hue_mask = ( (hsv(:,:,1) >= channel1Min) | (hsv(:,:,1) <= channel1Max) );

s = hsv(:,:,2);
num_bins = 256;
hist_s = imhist(s, num_bins);
bin_edges = linspace(0, 1, num_bins);

%figure, bar(bin_edges, hist_s)

smoothed_hist_s = smooth(hist_s, 10);
% figure, bar(bin_edges, smoothed_hist_s)
% hold on
% bar(bin_edges, hist_s)

% Identification of unusual spikes within the saturation histogram
threshold = 1.5;  % treshold factor for histogram spikes
spike_indices = find(hist_s > threshold * smoothed_hist_s);
% Conversion of indices to saturation values
spike_values = bin_edges(spike_indices);

mask = false(size(s));
tolerance = 0.00;  % Tolerance range for peak values ( adjustable)

for i = 1:length(spike_values)
    mask = mask | (s >= (spike_values(i) - tolerance) & ...
                   s <= (spike_values(i) + tolerance));
end
% figure, imshow(mask)

masked_r_channel = r_channel;
masked_g_channel = g_channel;
masked_b_channel = b_channel;
masked_r_channel(mask) = 0; 
masked_g_channel(mask) = 0;
masked_b_channel(mask) = 0;
masked_r_channel(hue_mask) = 0; 
masked_g_channel(hue_mask) = 0;
masked_b_channel(hue_mask) = 0;

% Interpolation to replace 0s - 9x9 median filter
corrected_r_channel = medfilt2(masked_r_channel, [9 9]);
corrected_g_channel = medfilt2(masked_g_channel, [9 9]);
corrected_b_channel = medfilt2(masked_b_channel, [9 9]);
% Replacement of the identified artifact pixels with corrected pixels
final_rgb_image = cat(3, corrected_r_channel, corrected_g_channel, corrected_b_channel);
% Conversion back to RGB 
img_pp = final_rgb_image;

% % Anzeigen des korrigierten Bildes
% figure, imshow(img_pp);
% title('Corrected Image');

end