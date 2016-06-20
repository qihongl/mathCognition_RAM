%% generate some training examples
% the image has only one object, the ram agent needs to learn to fixate at 
% that object
clear; 

% parameters
param.saveDir = '../../plots/oneObj';
imgName = 'oneObj';

% set parameters for objects and frame
param.showImg = 0;
param.saveImg = 1;
param.saveStruct = 0; 
param.getPrototype = 0;


param.obj_num = 1;
param.obj_radius = 4;

% the dimension of the image
param.frame_ver = 28;
param.frame_hor = 28;
% empty spaces
param.frame_boundary = param.obj_radius * 3;
param.frame_space = param.obj_radius * 3;
%  
param.frame_distortion = param.obj_radius * 1.5; 


%% generate images 
for n = 1 : 20
    param.imgName = sprintf('%s%.3d', imgName, n);
    genTrainExp(param);
end

%% write a parameter file 
filename = fullfile(param.saveDir,'paramRecord.txt');
fileID = fopen(filename,'w');
writeParam(fileID, param)
fclose(fileID);