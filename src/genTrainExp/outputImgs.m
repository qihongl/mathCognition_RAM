%% generate some training examples
% the image has only one object, the ram agent needs to learn to fixate at 
% that object
clear variables; close all; 

% the number of images that you want to generate
numImages = 20;

% parameters
imgName = 'multiObj';
param.saveDir = strcat('../../plots/', imgName);

% prototype: all objects are "centered" and aligned
% randomVec: each object is translated by a random vector
% allPoss: UNDER CONSTRUCTION
allPatterns = {'prototype', 'randomVec', 'allPoss'};
param.pattern = allPatterns{2};

% set parameters for objects and frame
param.showImg = 1;          % display a sample image 
param.saveImg = 0;          % same the image 
param.saveStruct = 0;       % save the matrix representation 

% img parameter 
param.obj_num = 5;          % number of objects
param.obj_radius = 4;       % size of the object 
param.varySize = 0;         % random radius for object 

param.frame_ver = 28;       % the length of the image
param.frame_hor = 80;       % the height of the image
param.frame_boundary = param.obj_radius * 3;    % space to the boundary of img
param.frame_space = param.obj_radius * 3;       % space in between objects

param.distortion_x = param.obj_radius * 1; % magnitude of distortion
param.distortion_y = param.obj_radius * 2;
allDistributions = {'elliptical', 'rectangular'};
param.frame_randVecDistribution = allDistributions{1};


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