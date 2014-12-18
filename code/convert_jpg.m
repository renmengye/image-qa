%%
output_folder = '../data/nyu-depth-jpg/';
images_dim = size(images);
for i = 1:images_dim(4)
    imwrite(images(:,:,:,i), strcat(output_folder, sprintf('image%d.jpg', i)));
    disp(i);
end

%%
output_list = '../data/image_list.txt';
flist = fopen(output_list, 'w');
for i = 1:images_dim(4)
    fprintf(flist, 'nyu-depth-jpg/image%i.jpg\n', i);
end
fclose(flist);