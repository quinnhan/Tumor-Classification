%% Brain Tumor Classification
% initialize work space
clear all, close all, clc


% First task is to import the image data 
yesfolder = 'TumorDataset/yes';
nofolder = 'TumorDataset/no';

yesfiles = dir(fullfile(yesfolder));
nofiles = dir(fullfile(nofolder));

yesfiles(1:2) = []; % deleting directory elements
[val, idx] = min([yesfiles.bytes]); % get min value and it's index
smallestimageyes = fullfile(yesfolder,yesfiles(idx).name); % find name of smallest image
[rows, columns, colorchannels] = size(imread(smallestimageyes)); % get size of smallest image
% loop through each folder and add images to matrix

% yes dataset
YesData = [];
for i = 1:length(yesfiles)
    basefilename = yesfiles(i).name;
    fullfilename = fullfile(yesfolder,basefilename);
    im = imread(fullfilename);
    if size(im,3) > 1 % checks if image is not gray
        im = rgb2gray(im); % converting to gray scale for easier analysis
    end
    imr = imresize(im,[rows,columns]);
    YesData(:,i) = double(imr(:));
end

nofiles(1:2) = [];
[val,idx] = min([nofiles.bytes]);
smallestimageno = fullfile(nofolder,nofiles(idx).name);
[rows2, columns2, colorchannels2] = size(imread(smallestimageno));

% no dataset
NoData = []; 
for j = 1:length(nofiles)
    basefilename = nofiles(j).name;
    fullfilename = fullfile(nofolder,basefilename);
    im = imread(fullfilename);
    if size(im,3) > 1 % checks if image is not gray
        im = rgb2gray(im); % converting to gray scale for easier analysis
    end
    imr = imresize(im,[rows,columns]);
    NoData(:,j) = double(imr(:));
end


%% Split into training test groups
q1 = randperm(size(YesData,2));
q2 = randperm(size(NoData,2));

split_yes = floor(0.8*length(q1)); % split the training/test data by this amount
split_no = floor(0.8*length(q2));

YesData_train = YesData(:,q1(1:split_yes));
YesData_test = YesData(:,q1((split_yes + 1):end));

NoData_train = NoData(:,q2(1:split_no));
NoData_test = NoData(:,q2((split_no + 1):end));

