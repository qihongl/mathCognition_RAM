function [ coords ] = getObjCoords( obj, frame )

pt_y = round(frame.ver / 2);
pt_ys = repmat(pt_y, [obj.num,1]);
pt_xs = frame.boundary + (0 : obj.num - 1)' * frame.space;

% check if objects out of bound
if any(pt_xs > (frame.hor - frame.boundary))
    error('ERROR: The max number of objects is %d!!!', ...
        find(pt_xs > (frame.hor - frame.boundary), 1) - 1)
end

coords = horzcat(pt_xs, pt_ys);

end

