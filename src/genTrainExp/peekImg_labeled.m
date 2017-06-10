load ../../datasets/fiveObj_centered/fiveObj_centered002.mat
% vectorImg
label = vectorImg(1);
img_size = 80;
max_num_obj = 5; 

figure(2)
imagesc(reshape(vectorImg(1:img_size*img_size),img_size,img_size))
colormap gray
axis equal tight
colorbar

numObj = vectorImg(img_size*img_size+1);

title_text = sprintf('Number of obj = %d', numObj);
title(title_text, 'fontsize', 14)

coords = reshape(vectorImg(img_size*img_size+2:end), [max_num_obj,2])



