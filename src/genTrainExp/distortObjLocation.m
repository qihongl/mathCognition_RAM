%% add a random vector to the object location (objCoords)
function [ objCoords ] = distortObjLocation( objCoords, frameParam )

for o = 1 : size(objCoords,1)
    % define the random vector 
    randVec = genRandVector(frameParam.distortion, frameParam.randVecDistribution);
    % translate the original coord by that random vector 
    objCoords(o,:) = objCoords(o,:) + round(randVec);
end

end

