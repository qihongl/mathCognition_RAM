%% make a demo img for the training set 
clear; clf 

allDataSets = {'multiObj', 'oneObj', 'oneObj_big'};
dataSetName = allDataSets{2};
format = '.mat';

% find the path 
path = strcat('../../datasets/', dataSetName);

% get the images 
% randsample(500-9,1)
numImg = 9;
startingIdx = randsample(500-numImg,1)
for i = 1: numImg
    filename = sprintf(strcat(dataSetName,'%.3d'),startingIdx+i);
    filename = strcat(filename, format);
    img = load(fullfile(path,filename));
    % show the image 
    subplot(3,3,i)
    imagesc(img.img)
    axis equal tight
end
colormap gray
suptitle_text = sprintf('Some examples for the %s data set',dataSetName);
suptitle(suptitle_text)