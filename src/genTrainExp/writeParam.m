function [] = writeParam(fileID, param)
%UNTITLED2 Summary of this function goes here
%   Detailed explanation goes here

fields = fieldnames(param);

for i=1:numel(fields)
    fieldName = fields{i};
    fieldVal = param.(fields{i});
    if isnumeric(fieldVal)
        fieldVal = num2str(fieldVal);
    end
    % write parameter info to txt 
    fprintf(fileID,'%.20s %4s\n',fieldName, fieldVal);
end

end

