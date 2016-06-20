clear; 

saveDir = '../../plots/genTrainExp';
imgName = 'distorted';


for n = 1 : 7 
    thisImgName = sprintf('%s%d', imgName,n);
    genTrainExp(n,saveDir);
end