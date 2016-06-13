%% generate training examples for the counting task
% space in between is fixed, objects are left aligned
function [ ] = genTrainExp(numObj)

% set parameters for objects and frame
showImg = 1;
saveImg = 0;
saveStruct = 0; 
getPrototype = 0;


obj.num = numObj;
obj.radius = 5;
% the dimension of the image
frame.ver = 30;
frame.hor = 160;
% empty spaces
frame.boundary = obj.radius * 4;
frame.space = obj.radius * 4;


%% generate images
% blank slate
img = zeros(frame.ver, frame.hor);
% generated the coordinates for all pixels on the image
[x,y] = meshgrid(0: frame.hor, 0: frame.ver);
coords = horzcat(x(:), y(:));

% generated the coordinates for all objects
obj.coords = getObjCoords(obj, frame);
if ~getPrototype
    obj.coords = distortObjLocation(obj.coords, obj.radius);
end

% put objects on the frame
img = placeObjs( obj, coords, img);


%% plot
if showImg
    imagesc(img)
    colorbar
    axis equal
end

%% save the image
if saveImg
    imgFormat = '.jpg';
    if getPrototype
        imgName = sprintf('proto%d', obj.num);
    else
        imgName = sprintf('distorted%d', obj.num);
    end
    
    imgName = strcat(imgName,imgFormat);
    imwrite(img, fullfile('../plots/genTrainExp', imgName))
end

%% save as structure (numerical form)
if saveStruct
    
end

end