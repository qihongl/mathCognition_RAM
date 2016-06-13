%% generate random vector s.t. norm(vec) < l, where l is pre-sepcified
function [ vector ] = genRandVector(maxLength)

% generate r and theta, uniformly 
r = rand(1);
theta = rand(1) * 2 * pi;

% compute the coordinates 
x = sqrt(r) * cos(theta);
y = sqrt(r) * sin(theta);
vector = [x,y];

% multiply by the max length
vector = vector * maxLength;
end
