function [ coords ] = getObjCoords( obj, frame )
% check input arguments 
possible_pt_xs = frame.boundary + (0:obj.maxNum-1) * frame.space;
if any(possible_pt_xs > (frame.hor - frame.boundary))
    error('ERROR: The max number of objects is %d!!!', ...
        find(possible_pt_xs > (frame.hor - frame.boundary), 1) - 1)
end

% center along the y dim
pt_y = round(frame.ver / 2);
% all objects are aligned 
pt_ys = repmat(pt_y, [obj.num,1]);

% the x coordinate of the objects, left aligned at the origin (0)
pt_xs_fromOrigin = (0 : obj.num - 1)' * frame.space;

if strcmp(frame.alignment, 'left')
    % horizontally, objects are equally spaced 
    pt_xs = frame.boundary + pt_xs_fromOrigin;
elseif strcmp(frame.alignment, 'center')
    pt_xs = pt_xs_fromOrigin + (frame.hor/2-median(pt_xs_fromOrigin)); 
else
    error('ERROR: unrecognizable alignment pattern.')
end

% concatenate x and y 
coords = horzcat(pt_xs, pt_ys);

end

