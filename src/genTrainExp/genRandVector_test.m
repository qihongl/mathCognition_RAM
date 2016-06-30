%% understand the sampling distribution of the random vectors
clear; clf;
radius = 6; 
sampleSize = 2000; 
allDistributions = {'circle', 'ellipse', 'rectangle'};
distributionType = allDistributions{3};

% visualize the location of the random vectors
subplot(1,2,1)
hold on 
nm = nan(sampleSize,1);
for i= 1:sampleSize
    v = genRandVector(radius, distributionType);
    plot(v(1),v(2),'bx')
    nm(i) = norm(v);
end
plot(0,0,'r+', 'linewidth', 2)
axis equal tight

fs = 14; 

title('the sampling distribution of the coordinates', 'fontsize', fs)
hold off

% visualize the norms of the random vectors
subplot(1,2,2)
histogram(nm)
ylabel('frequency', 'fontsize', fs); 
xlabel('norm', 'fontsize', fs)
title_text = sprintf('the sampling distribution of the norm \n mean norm: %f', mean(nm));
title(title_text,'fontsize', fs)


