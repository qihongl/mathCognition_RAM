%% generate random vector s.t. norm(vec) < l, where l is pre-sepcified
function [ vector ] = genRandVector(maxLength)

% generate r and theta, uniformly 
r = sqrt(rand);
theta = rand * 2 * pi;

% compute the coordinates 
x = r * cos(theta);
y = r * sin(theta);

% scale by the max length
vector = [x,y] .* maxLength;
end
