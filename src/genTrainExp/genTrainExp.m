%% generate training examples for the counting task
% space in between is fixed, objects are left aligned
function [ ] = genTrainExp(param)
%% read parameters for objects and frame
showImg = param.showImg;
saveImg = param.saveImg;
saveStruct = param.saveStruct; 
pattern = param.pattern;
obj.maxNum = param.max_obj_num;
obj.num = param.obj_num;
obj.varySize = param.varySize; 
obj.radius = param.obj_radius;
frame.alignment = param.alignment;
frame.ver = param.frame_ver;
frame.hor = param.frame_hor;
frame.boundary = param.frame_boundary;
frame.space = param.frame_space;
frame.distortion_x = param.distortion_x; 
frame.distortion_y = param.distortion_y; 
frame.randVecDistribution = param.frame_randVecDistribution;

%% generate images
% blank slate
img = zeros(frame.ver, frame.hor);
% generated the coordinates for all pixels on the image
[x,y] = meshgrid(0: frame.hor, 0: frame.ver);
coords = horzcat(x(:), y(:));

% generated the coordinates for all objects
obj.coords = getObjCoords(obj, frame);

if strcmp(pattern,'prototype')
    % do nothing
elseif strcmp(pattern,'randomVec')
    obj.coords = distortObjLocation(obj.coords, frame);
elseif strcmp(pattern,'allPoss')
    % TODO
else
    error('Unrecognizable pattern parameter! ')
end

% put objects on the frame
img = drawObjs(obj, coords, img);

% pixel value in {1, 0}
if sum(sum((img == true) | (img == false))) ~= frame.ver * frame.hor
    warning('WOW: The pixel values are strange')
end


%% additional options 
% plot
if showImg
    imagesc(img);
    colorbar
    axis equal tight
    colormap gray
end

% save the image
if saveImg
    imgFormat = '.jpg';
    imgName = strcat(param.imgName,imgFormat);
    imwrite(img, fullfile(param.saveDir, imgName));
end

% save as structure (numerical form)
if saveStruct
    imgFileFormat = '.mat';
    imgFileName = strcat(param.imgName,imgFileFormat);
    fullImgFileName = fullfile(param.saveDir,imgFileName);
    vectorImg = reshape(img, [size(img,1) * size(img,2), 1]);
    if param.supervised 
        vectorImg = vertcat(vectorImg, param.obj_num, reshape(obj.coords, [param.obj_num * 2,1]));
    end
    save(fullImgFileName, 'vectorImg');
end

end