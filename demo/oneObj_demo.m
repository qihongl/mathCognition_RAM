% make a demo img for the oneObj training set 
clear; 

path = '../plots/oneObj';
format = '.mat';

for i = 1 : 9 
    filename = sprintf('oneObj%.3d',i);
    filename = strcat(filename, format);
    img = load(fullfile(path,filename));
    
    subplot(3,3,i)
    imagesc(img.img)
    axis square
end
colormap gray
suptitle('Some training examples for the oneObj data set')