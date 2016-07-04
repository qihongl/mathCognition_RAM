%% place objects onto the training image
% needs to know the radius and the coordinates of the objects
function [ img ] = drawObjs(obj, frameCoords, img)

% check input
minRadiusAllowed = 1; 
if obj.radius <= minRadiusAllowed
    error('The input object radius parameter is too small!')
end
    
% pairwise distance matrix for all pixels on the image
distmat = squareform(pdist(frameCoords));

for o = 1 : obj.num
    obj.coords(o,:)
    [~,idx] = ismember(obj.coords(o,:), frameCoords,'rows');
    
    if ~obj.varySize
        % fixed object radius 
        actualRadius = obj.radius;
    else
        % iid random scale the radius of each object 
        radiusIsValid = false; 
        while ~radiusIsValid
            % re-roll the radius until the radius become valid
            actualRadius = round(unifrnd(0,1) * obj.radius);
            if actualRadius > minRadiusAllowed
                radiusIsValid = true;
            end
        end
    end
    
    % paint the object on the image!
    fill = frameCoords(distmat(:,idx) < actualRadius,:);
    
    % fill all frameCoords that is close to the center
    for i = 1 : size(fill,1)
        tempCoord = fill(i,:);
        img(tempCoord(2), tempCoord(1)) = 1;
    end
end


end

