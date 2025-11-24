classdef CustomMultiplicationLayer < nnet.layer.Layer
    methods
        function layer = CustomMultiplicationLayer(name)
            layer.Name = name;
            layer.NumInputs = 2; % <Important>: Accepts 2 inputs
        end
        function Z = predict(~, X, Y)
            Z = X .* Y;
        end
    end
end