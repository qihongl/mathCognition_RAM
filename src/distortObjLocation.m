function [ objCoords ] = distortObjLocation( objCoords, distortion )
%UNTITLED6 Summary of this function goes here
%   Detailed explanation goes here
for o = 1 : size(objCoords,1)
    randomMove = genRandVector(distortion);
    objCoords(o,:) = objCoords(o,:) + round(randomMove);
end

end

