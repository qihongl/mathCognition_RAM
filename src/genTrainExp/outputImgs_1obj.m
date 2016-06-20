%% generate some training examples
% the image has only one object, the ram agent needs to learn to fixate at 
% that object
clear; 

saveDir = '../../plots/oneObj';
imgName = 'oneObj';

for n = 1 : 20
    thisImgName = sprintf('%s%.3d', imgName,n);
    genTrainExp(1,saveDir,thisImgName);
end