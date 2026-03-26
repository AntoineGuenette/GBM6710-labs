storedStructure = load("robotBimages23.mat");
% View the names of variables inside the loaded structure
disp(fieldnames(storedStructure));

% Extract the image data into a new variable
imageL = storedStructure.imgL; 
imageR = storedStructure.imgR;
imwrite(imageL, "imageL.png");
imwrite(imageR, "imageR.png");
% Display the image, automatically scaling the contrast
figure; % Opens a new figure window
imshow(imageArray, []);
% or
% imagesc(imageArray); axis image; % 'axis image' ensures correct aspect ratio
