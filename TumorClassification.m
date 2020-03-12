%% Brain Tumor Classification
% initialize work space
clear all, close all, clc


% First task is to import the image data 
yesfolder = 'TumorDataset/yes';
nofolder = 'TumorDataset/no';

yesfiles = dir(fullfile(yesfolder));
nofiles = dir(fullfile(nofolder));

yesfiles(1:2) = []; % deleting directory elements

% find the smallest image to resize to
% when combining them into one large matrix, they need to be the same size
[val, idx] = min([yesfiles.bytes]); % get min value and it's index

% repeat for the no cases
nofiles(1:2) = [];
[val2,idx2] = min([nofiles.bytes]);

% have to check for which is smaller then will resize images to all that
% size
if val < val2 
    smallestimage = fullfile(yesfolder,yesfiles(idx).name); % find name of smallest image
else 
    smallestimage = fullfile(nofolder,nofiles(idx2).name);
end
[rows, columns, colorchannels] = size(imread(smallestimage)); % get size of smallest image

% loop through each folder and add images to matrix

% yes dataset
YesData = [];
for i = 1:length(yesfiles)
    basefilename = yesfiles(i).name;
    fullfilename = fullfile(yesfolder,basefilename);
    im = imread(fullfilename);
    if size(im,3) > 1 % checks if image is not gray, rgb2gray can't work on already gray scale images
        im = rgb2gray(im); % converting to gray scale for easier analysis
    end
    imr = imresize(im,[rows,columns]); 
    YesData(:,i) = double(imr(:));
end

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

%% Look at FFT of some images

% comparing the yes and the no images along with their respective FFT's
for i = 1:3
    ys = YesData(:,i);
    ns = NoData(:,i);

    im_y = reshape(ys,rows,columns);
    im_n = reshape(ns,rows,columns);

    L = length(ys)/8400; % i didn't know what to put for here but I don't think it matters?

    n = length(ys);
    k=(2*pi/L)*[0:n/2 -n/2:-1];  
    ks=fftshift(k(1:end-1));

    n2 = length(ns);
    kn = (2*pi/L)*[0:n2/2 -n2/2:-1];
    kns = fftshift(kn);

    ys_f = fft(ys);
    ns_f = fft(ns);

    figure (i)
    subplot(2,2,1)
    imshow(uint8(im_y))

    subplot(2,2,2)
    imshow(uint8(im_n))

    subplot(2,2,3)
    plot(ks,abs(fftshift(ys_f))/max(ys_f))

    subplot(2,2,4)
    plot(kns(1:end-1),abs(fftshift(ns_f))/max(ns_f))
end

%% Split into training test groups
testruns = 20;
percentage = zeros(1,testruns);
for p = 1:testruns
    
q1 = randperm(size(YesData,2));
q2 = randperm(size(NoData,2));

% split the training/test data by a set amount
split_yes = floor(0.8*length(q1)); 
split_no = floor(0.8*length(q2));

YesData_train = YesData(:,q1(1:split_yes));
YesData_test = YesData(:,q1((split_yes + 1):end));

NoData_train = NoData(:,q2(1:split_no));
NoData_test = NoData(:,q2((split_no + 1):end));


X_test = [YesData_test, NoData_test];

%% FFT the images and take SVD
X = [YesData_train, NoData_train];

% X_fft = zeros(size(X));
% for l = 1:length(YesData_train(1,:))
%     X_fft(:,l) = abs(fft(YesData_train(:,l)));
% end

% try wavelet transform instead
X_fft = tc_wavelet(X,rows,columns);
X_test_wav = tc_wavelet(X_test,rows,columns);

[U,S,V] = svd(X_fft,'econ');
%% Graph some stuff

%figure()
sig = diag(S);
[M,N] = size(X);

%subplot(1,2,1), plot(sig(1:50),'ko','Linewidth',[1.5])
ylabel('Singular Values')
xlabel('Singular Value Along Diagonal')

%subplot(1,2,2), semilogy(sig(1:50),'ko','Linewidth',[1.5])
ylabel('Log of Singular Values')
xlabel('Singular Value Along Diagonal')

%% Run through matlab classify

numFeat = 30;

xtrain = V(:,1:numFeat);
xtest = U'*X_test_wav;

ctrain = [repmat({'Tumor'},[size(YesData_train,2),1]);repmat({'NoTumor'},[size(NoData_train,2),1])];
truth = [repmat({'Tumor'},[size(YesData_test,2),1]);repmat({'NoTumor'},[size(NoData_test,2),1])];

svm.mod = fitcsvm(xtrain,ctrain);
pre = predict(svm.mod,xtest(:,1:numFeat));

num_correct = 0;
for k = 1:length(truth)
   if strcmp(pre{k},truth{k})
        num_correct = num_correct + 1;
   end
end
percentage(p) = (num_correct/length(truth))*100;
end
mean(percentage)