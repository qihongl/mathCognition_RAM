%% understand the sampling distribution of the random vectors
clear; close; 
radius = 5; 
sampleSize = 2000; 

% visualize the location of the random vectors
subplot(1,2,1)
hold on 
nm = nan(sampleSize,1);
for i= 1:sampleSize
    v = genRandVector(radius);
    plot(v(1),v(2),'x')
    nm(i) = norm(v);
end
axis square
title('the sampling distribution of the coordinates')
hold off

% visualize the norms of the random vectors
subplot(1,2,2)
histogram(nm)
xlabel('frequency'); ylabel('norm')
title_text = sprintf('the sampling distribution of the norm \n mean norm: %f', mean(nm));
title(title_text)


