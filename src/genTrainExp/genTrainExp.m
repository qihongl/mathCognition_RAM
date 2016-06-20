%% generate training examples for the counting task
% space in between is fixed, objects are left aligned
function [ ] = genTrainExp(param)

%% set parameters for objects and frame
showImg = param.showImg;
saveImg = param.saveImg;
saveStruct = param.saveStruct; 
getPrototype = param.getPrototype;


obj.num = param.obj_num;
obj.radius = param.obj_radius;

% the dimension of the image
frame.ver = param.frame_ver;
frame.hor = param.frame_hor;
% empty spaces
frame.boundary = param.frame_boundary;
frame.space = param.frame_space;
% 
frame.distortion = param.frame_distortion; 

%% generate images
% blank slate
img = zeros(frame.ver, frame.hor);
% generated the coordinates for all pixels on the image
[x,y] = meshgrid(0: frame.hor, 0: frame.ver);
coords = horzcat(x(:), y(:));

% generated the coordinates for all objects
obj.coords = getObjCoords(obj, frame);
if ~getPrototype
    obj.coords = distortObjLocation(obj.coords, frame.distortion);
end

% put objects on the frame
img = placeObjs( obj, coords, img);

% pixel value in {1, 0}
if sum(sum((img == true) | (img == false))) ~= frame.ver * frame.hor
    warning('WOW: The pixel values are strange')
end

%% plot
if showImg
    imagesc(img);
    colorbar
    axis equal
end

%% save the image
if saveImg
    imgFormat = '.jpg';
    imgName = strcat(param.imgName,imgFormat);
    imwrite(img, fullfile(param.saveDir, imgName));
end

%% save as structure (numerical form)
if saveStruct
    % TODO: to be implemented
    imgFileFormat = '.mat';
    imgFileName = strcat(param.imgName,imgFileFormat);
    fullImgFileName = fullfile(param.saveDir,imgFileName);
    save(fullImgFileName, 'img');
end

end