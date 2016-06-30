%% generate random vector s.t. norm(vec) < l, where l is pre-sepcified
function [ vector ] = genRandVector(maxLength, typeDistribution)
if maxLength <= 0 
    error('maxLength has to be positive')
end

if strcmp(typeDistribution, 'circle')
    % generate r and theta, uniformly 
    r = sqrt(rand);
    theta = rand * 2 * pi;
    % compute the coordinates 
    x = r * cos(theta);
    y = r * sin(theta);
    % scale by the max length
    vector = [x,y] .* maxLength;
    
elseif strcmp(typeDistribution, 'rectangle')
    r1 = maxLength;     % x dim 
    r2 = maxLength*2;   % y dim
    % Uniform rectangle 
    x = r1 * (2*rand-1);
    y = r2 * (2*rand-1);
    vector = [x, y];
    
elseif strcmp(typeDistribution, 'ellipse')
    r1 = maxLength;     % x dim 
    r2 = maxLength*2;   % y dim    
    while true
        % Uniform rectangle 
        x = r1 * (2*rand-1);
        y = r2 * (2*rand-1);
        % break if in the ellipse
        if (x.*x/(r1^2)+y.*y/(r2^2)) < 1
            break;
        end
    end
    vector = [x, y];
else
    error('Unrecognizable distribution type for the distortion.')
end


end
