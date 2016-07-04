%% generate random vector s.t. norm(vec) < l, where l is pre-sepcified
function [ vector ] = genRandVector(maxLength_x, maxLength_y, typeDistribution)
% check input argument
lowerBound_raduis = 0; 
if maxLength_x <= lowerBound_raduis || maxLength_y <= lowerBound_raduis 
    error('maxLength has to be positive')
end

if strcmp(typeDistribution, 'rectangular')
    % Uniformly sample from a rectangle
    [x,y] = getRectPts (maxLength_x, maxLength_y);
    
elseif strcmp(typeDistribution, 'elliptical')
    while true
        % Uniformly sample from a rectangle
        [x,y] = getRectPts (maxLength_x, maxLength_y);
        % break if in the ellipse
        if (x.*x/(maxLength_x^2)+y.*y/(maxLength_y^2)) < 1
            break;
        end
    end
    
else
    error('Unrecognizable distribution type for the distortion.')
end

% concatenate to get a vector
vector = horzcat(x,y);

end

%% helper function 
function [x,y] = getRectPts (maxLength_x, maxLength_y)
% Uniformly sample from a rectangle
x = maxLength_x * (2*rand-1);
y = maxLength_y * (2*rand-1);
end
