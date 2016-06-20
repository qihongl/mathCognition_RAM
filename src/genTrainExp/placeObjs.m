%% place objects onto the training image
% needs to know the radius and the coordinates of the objects
function [ img ] = placeObjs(obj, frameCoords, img)

% pairwise distance matrix for all pixels on the image
distmat = squareform(pdist(frameCoords));
    
for o = 1 : obj.num
    obj.coords(o,:)
    [~,idx] = ismember(obj.coords(o,:), frameCoords,'rows');
        
    % paint the object on the image
    fill = frameCoords(distmat(:,idx) < obj.radius,:);
    % fill all frameCoords that close to the center
    for i = 1 : size(fill,1)
        tempCoord = fill(i,:);
        img(tempCoord(2), tempCoord(1)) = 1;
    end
end


end

