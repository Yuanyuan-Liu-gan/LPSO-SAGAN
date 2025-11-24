classdef ScaleLayer < nnet.layer.Layer
    properties (Learnable)
        Gamma
    end
    methods
        function layer = ScaleLayer(name)
            layer.Name = name;
            layer.Gamma = dlarray(single(0));
        end
        function Z = predict(layer, X)
            Z = layer.Gamma .* X;
        end
    end
end