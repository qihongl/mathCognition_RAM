%% understand the sampling distribution of the random vectors
clear; clf;
radius_x = 6; 
radius_y = 10; 
sampleSize = 2000; 
allDistributions = {'elliptical', 'rectangular'};
distributionType = allDistributions{1};

% visualize the location of the random vectors
subplot(1,2,1)
hold on 
nm = nan(sampleSize,1);
for i= 1:sampleSize
    v = genRandVector(radius_x, radius_y, distributionType);
    plot(v(1),v(2),'bx')
    nm(i) = norm(v);
end
plot(0,0,'r+', 'linewidth', 2)

% aspect ratio, axis, ticks 
axis equal 
ax = gca;
boundary = round(min(radius_x, radius_y) / 3);
ax.XTick = -radius_x: radius_x;
ax.YTick = -radius_y: radius_y;
xlim([-radius_x-boundary,radius_x+boundary])
ylim([-radius_y-boundary,radius_y+boundary])

fs = 14; 
title_text = sprintf(['The sampling distribution of the coordinates\n', ...
    'R_x = %d, R_y = %d'], radius_x,radius_y);
title(title_text, 'fontsize', fs)
hold off

% visualize the norms of the random vectors
subplot(1,2,2)
histogram(nm)
ylabel('frequency', 'fontsize', fs); 
xlabel('norm', 'fontsize', fs)
title_text = sprintf(['the sampling distribution of the norm\n' ...
    'mean norm: %f'], mean(nm));
title(title_text,'fontsize', fs)


