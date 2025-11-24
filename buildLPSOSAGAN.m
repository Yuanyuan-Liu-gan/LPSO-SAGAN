function [gen, disc] = buildLPSOSAGAN(imgSize, config)
% Builds network using layerGraph and Custom Multiplication Layer

%% Generator Construction
lgraph = layerGraph();

% 1. Core Layers (Sequential)
tempLayers = [
    imageInputLayer([1 1 100], 'Name', 'in', 'Normalization', 'none')
    transposedConv2dLayer(4, 512, 'Name', 'tconv1', 'Stride', 1, 'Cropping', 0)
    batchNormalizationLayer('Name', 'bn1')
    reluLayer('Name', 'relu1')
    ];
lgraph = addLayers(lgraph, tempLayers);
lastLayer = 'relu1';

% SA5 (512 Ch)
if config(5) == 1
    [lgraph, lastLayer] = addAttentionBlock(lgraph, lastLayer, 'SA5', 512);
end

% 4x4 -> 8x8
tempLayers = [
    transposedConv2dLayer(4, 256, 'Name', 'tconv2', 'Stride', 2, 'Cropping', 1)
    batchNormalizationLayer('Name', 'bn2')
    reluLayer('Name', 'relu2')
    ];
lgraph = addLayers(lgraph, tempLayers);
lgraph = connectLayers(lgraph, lastLayer, 'tconv2');
lastLayer = 'relu2';

if config(4) == 1
    [lgraph, lastLayer] = addAttentionBlock(lgraph, lastLayer, 'SA4', 256); 
end

% 8x8 -> 16x16
tempLayers = [
    transposedConv2dLayer(4, 128, 'Name', 'tconv3', 'Stride', 2, 'Cropping', 1)
    batchNormalizationLayer('Name', 'bn3')
    reluLayer('Name', 'relu3')
    ];
lgraph = addLayers(lgraph, tempLayers);
lgraph = connectLayers(lgraph, lastLayer, 'tconv3');
lastLayer = 'relu3';

if config(3) == 1
    [lgraph, lastLayer] = addAttentionBlock(lgraph, lastLayer, 'SA3', 128); 
end

% 16x16 -> 32x32
tempLayers = [
    transposedConv2dLayer(4, 64, 'Name', 'tconv4', 'Stride', 2, 'Cropping', 1)
    batchNormalizationLayer('Name', 'bn4')
    reluLayer('Name', 'relu4')
    ];
lgraph = addLayers(lgraph, tempLayers);
lgraph = connectLayers(lgraph, lastLayer, 'tconv4');
lastLayer = 'relu4';

if config(2) == 1
    [lgraph, lastLayer] = addAttentionBlock(lgraph, lastLayer, 'SA2', 64); 
end

% 32x32 -> 64x64
tempLayers = [
    transposedConv2dLayer(4, 32, 'Name', 'tconv5', 'Stride', 2, 'Cropping', 1)
    batchNormalizationLayer('Name', 'bn5')
    reluLayer('Name', 'relu5')
    ];
lgraph = addLayers(lgraph, tempLayers);
lgraph = connectLayers(lgraph, lastLayer, 'tconv5');
lastLayer = 'relu5';

if config(1) == 1
    [lgraph, lastLayer] = addAttentionBlock(lgraph, lastLayer, 'SA1', 32); 
end

% Output
tempLayers = [
    transposedConv2dLayer(4, 3, 'Name', 'out_conv', 'Stride', 2, 'Cropping', 1)
    tanhLayer('Name', 'out_tanh')
    ];
lgraph = addLayers(lgraph, tempLayers);
lgraph = connectLayers(lgraph, lastLayer, 'out_conv');

gen = dlnetwork(lgraph);

%% Discriminator Construction
lgraphD = layerGraph();
tempLayers = [
    imageInputLayer(imgSize, 'Name', 'in', 'Normalization', 'none')
    convolution2dLayer(4, 64, 'Name', 'conv1', 'Stride', 2, 'Padding', 1)
    leakyReluLayer(0.2, 'Name', 'lrelu1')
    ];
lgraphD = addLayers(lgraphD, tempLayers);
lastLayer = 'lrelu1';

tempLayers = [
    convolution2dLayer(4, 128, 'Name', 'conv2', 'Stride', 2, 'Padding', 1)
    batchNormalizationLayer('Name', 'bn2')
    leakyReluLayer(0.2, 'Name', 'lrelu2')
    ];
lgraphD = addLayers(lgraphD, tempLayers);
lgraphD = connectLayers(lgraphD, lastLayer, 'conv2');
lastLayer = 'lrelu2';

if config(6) == 1
    [lgraphD, lastLayer] = addAttentionBlock(lgraphD, lastLayer, 'SA6', 128); 
end

tempLayers = [
    convolution2dLayer(4, 256, 'Name', 'conv3', 'Stride', 2, 'Padding', 1)
    batchNormalizationLayer('Name', 'bn3')
    leakyReluLayer(0.2, 'Name', 'lrelu3')
    ];
lgraphD = addLayers(lgraphD, tempLayers);
lgraphD = connectLayers(lgraphD, lastLayer, 'conv3');
lastLayer = 'lrelu3';

if config(7) == 1
    [lgraphD, lastLayer] = addAttentionBlock(lgraphD, lastLayer, 'SA7', 256); 
end

tempLayers = [
    convolution2dLayer(4, 512, 'Name', 'conv4', 'Stride', 2, 'Padding', 1)
    batchNormalizationLayer('Name', 'bn4')
    leakyReluLayer(0.2, 'Name', 'lrelu4')
    convolution2dLayer(4, 1, 'Name', 'conv_out', 'Stride', 1, 'Padding', 0)
    ];
lgraphD = addLayers(lgraphD, tempLayers);
lgraphD = connectLayers(lgraphD, lastLayer, 'conv4');

disc = dlnetwork(lgraphD);
end

