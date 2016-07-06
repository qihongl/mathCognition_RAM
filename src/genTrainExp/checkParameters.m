function [ ] = checkParameters(p)
% check if the board is large enough (if there might be too many objs)
possible_pt_xs = p.frame_boundary + (0:p.max_obj_num-1) * p.frame_space;
if any(possible_pt_xs > (p.frame_hor - p.frame_boundary))
    error(['ERROR: The max number of objects is %d!!!\n', ...
        'please decrease the number of objects or make the board larger'], ...
        find(possible_pt_xs > (p.frame_hor - p.frame_boundary), 1) - 1)
end

% check if object might collide 
if p.obj_radius * 3 >= p.frame_space
    warning(['WARNING: Objects might collide. ',...
        'Recommend param.frame_space >= obj_radius * 3'])
end

end