function [gradG, gradD, lossG, lossD] = modelGradients(netG, netD, realX, Z)
fakeX = forward(netG, Z);
predReal = forward(netD, realX);
predFake = forward(netD, fakeX);
lossD = mean(relu(1 - predReal), 'all') + mean(relu(1 + predFake), 'all');
lossG = -mean(predFake, 'all');
gradG = dlgradient(lossG, netG.Learnables);
gradD = dlgradient(lossD, netD.Learnables);
end
