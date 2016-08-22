load ../../datasets/oneObj_centered/oneObj_centered002.mat
% vectorImg
label = vectorImg(1);
img_size = 28;

figure(2)
imagesc(reshape(vectorImg(1:img_size*img_size),img_size,img_size))
colorbar

numObj = vectorImg(img_size*img_size+1)
coords = reshape(vectorImg(img_size*img_size+2:end), [7,2])

