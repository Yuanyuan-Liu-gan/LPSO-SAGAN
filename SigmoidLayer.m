classdef SigmoidLayer < nnet.layer.Layer
    methods
        function layer = SigmoidLayer(name)
            layer.Name = name;
        end
        function Z = predict(~, X)
            Z = sigmoid(X);
        end
    end
end