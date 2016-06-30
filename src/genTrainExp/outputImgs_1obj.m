%% generate some training examples
% the image has only one object, the ram agent needs to learn to fixate at 
% that object
clear; 

% the number of images that you want to generate
numImages = 5;

% parameters
param.saveDir = '../../plots/oneObj';
imgName = 'oneObj';
allPatterns = {'prototype', 'randomVec', 'allPoss'};

% set parameters for objects and frame
param.pattern = allPatterns{2};
param.showImg = 1;
param.saveImg = 0;
param.saveStruct = 0; 


% img parameter 
param.obj_num = 5;          % number of objects
param.obj_radius = 3;       % size of the object 
param.varySize = 1;         % random radius for object 

param.frame_ver = 28;       % the length of the image
param.frame_hor = 80;       % the height of the image
param.frame_boundary = param.obj_radius * 3;    % space to the boundary of img
param.frame_space = param.obj_radius * 3;       % space in between objects
param.frame_distortion = param.obj_radius * 1.5; % magnitude of distortion


%% generate images 
for n = 1 : numImages
    param.imgName = sprintf('%s%.3d', imgName, n);
    genTrainExp(param);
end

%% write a parameter file 
filename = fullfile(param.saveDir,'paramRecord.txt');
fileID = fopen(filename,'w');
writeParam(fileID, param)
fclose(fileID);