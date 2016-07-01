%% make a demo img for the training set 
clear; clf 

allDataSets = {'multiObj', 'OneObjs'};
dataSetName = allDataSets{1};           % which dataset you want to see
format = '.mat';

% find the path 
path = strcat('../plots/', dataSetName);
% get the images 
for i = 1 : 9 
    filename = sprintf(strcat(dataSetName,'%.3d'),i);
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