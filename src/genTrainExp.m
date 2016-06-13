%% generate training examples for the counting task
% space in between is fixed, objects are left aligned
clear all;

% set parameters
obj.radius = 5;
obj.num = 4;
frame.ver = 30;
frame.hor = 160;
frame.boundary = obj.radius * 4;
frame.space = obj.radius * 4;

% template for the image
img = zeros(frame.ver, frame.hor);

%% gen prototype
% generated the coordinates for all objects
obj.coords = getObjCoords(obj, frame);

% generated the coordinates for all pixels on the image
[x,y] = meshgrid(0: frame.hor, 0: frame.ver);
coords = horzcat(x(:), y(:));

% pairwise distance matrix for all pixels on the image
distmat = squareform(pdist(coords));

% generate objects! 
for o = 1 : obj.num
    obj.coords(o,:)
    [~,idx] = ismember(obj.coords(o,:), coords,'rows');
    
    % paint the object on the image
    fill = coords(distmat(:,idx) < obj.radius,:);
    % fill all coords that close to the center
    for i = 1 : size(fill,1)
        tempCoord = fill(i,:);
        img(tempCoord(2), tempCoord(1)) = 1;
    end
end

%% plot 
imagesc(img)
colorbar
axis equal