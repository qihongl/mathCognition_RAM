%% understand the sampling distribution of the randNumGenerator
clear; clf

maxNum = 7;
sampleSize = 2000;
alpha = 1; 

x = nan(sampleSize,1);
for i = 1 : sampleSize
    x(i) = generateNum(maxNum, alpha);
end

histogram(x)

fs = 14; 
title_text = sprintf(['Distribution from generateNum\n', ...
    'maxNum = %d, alpha = %d, sample size = %d'], maxNum, alpha, sampleSize);
title(title_text, 'fontsize', fs)
xlabel('Number', 'fontsize', fs)
xlabel('Frequency', 'fontsize', fs)
