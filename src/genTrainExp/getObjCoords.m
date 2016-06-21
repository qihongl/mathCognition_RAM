function [ coords ] = getObjCoords( obj, frame )
% center along the y dim
pt_y = round(frame.ver / 2);
% all objects are aligned 
pt_ys = repmat(pt_y, [obj.num,1]);
% horizontally, objects are equally spaced 
pt_xs = frame.boundary + (0 : obj.num - 1)' * frame.space;

% check if objects out of bound
if any(pt_xs > (frame.hor - frame.boundary))
    error('ERROR: The max number of objects is %d!!!', ...
        find(pt_xs > (frame.hor - frame.boundary), 1) - 1)
end

% concatenate x and y 
coords = horzcat(pt_xs, pt_ys);

end

