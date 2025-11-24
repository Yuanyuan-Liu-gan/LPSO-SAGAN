function [finalLoss, netG, netD] = trainGAN_Subroutine(netG, netD, data, params)
iteration = 0; 
lossVal = 100;
avgG = []; 
avgSqG = []; 
avgD = []; 
avgSqD = [];
max_steps = 20;

reset(data);
while hasdata(data) && iteration < max_steps
    iteration = iteration + 1;
    X_tbl = read(data);
    realImg = single(cat(4, X_tbl.input{:,:}));
    realImg = (realImg - 127.5) / 127.5;
    realImg = dlarray(realImg, 'SSCB');
    
    if canUseGPU
        realImg = gpuArray(realImg); 
    end
    Z = dlarray(randn(1, 1, 100, size(realImg, 4), 'single'), 'SSCB');
    if canUseGPU
        Z = gpuArray(Z); 
    end
    
    [gradG, gradD, lossG, lossD] = dlfeval(@modelGradients, netG, netD, realImg, Z);
    
    [netD, avgD, avgSqD] = adamupdate(netD, gradD, avgD, avgSqD, iteration, params.lr, params.beta1, params.beta2);
    [netG, avgG, avgSqG] = adamupdate(netG, gradG, avgG, avgSqG, iteration, params.lr, params.beta1, params.beta2);
    lossVal = extractdata(lossG) + extractdata(lossD);
end
%Use gather() to ensure scalar is on CPU
finalLoss = double(gather(lossVal));
end
