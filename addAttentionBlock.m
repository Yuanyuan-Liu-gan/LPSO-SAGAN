function [lg, outputName] = addAttentionBlock(lg, inputName, blockName, numChannels)
convName = [blockName '_conv'];
sigName  = [blockName '_sig'];
multName = [blockName '_mult'];
scaleName = [blockName '_scale'];
addName  = [blockName '_add'];

% Branch: Attention Map Generation
layers = [
    convolution2dLayer(1, numChannels, 'Name', convName, 'Padding', 'same', 'WeightsInitializer', 'he')
    SigmoidLayer(sigName)
    ];
lg = addLayers(lg, layers);

% Connect Input to Branch
lg = connectLayers(lg, inputName, convName);

% Element-wise Multiplication (Attn Map * Input)
% CustomMultiplicationLayer takes 2 inputs
lg = addLayers(lg, CustomMultiplicationLayer(multName));
lg = connectLayers(lg, inputName, [multName '/in1']);
lg = connectLayers(lg, sigName,   [multName '/in2']);

% Scale and Add (Residual)
lg = addLayers(lg, ScaleLayer(scaleName));
lg = connectLayers(lg, multName, scaleName);

lg = addLayers(lg, additionLayer(2, 'Name', addName));
lg = connectLayers(lg, inputName, [addName '/in1']);
lg = connectLayers(lg, scaleName, [addName '/in2']);

outputName = addName;
end
