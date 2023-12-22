%%% Y. Zhang, et al,%  "Contour Structural Profiles: An Edge-Aware Feature Extractor for Hyperspectral Image Classification"
% IEEE Transactions on Geoscience and Remote Sensing, 2022
tic
clc
clear
close all
addpath ('functions')
addpath (genpath('libsvm-3.22'))
addpath (genpath('KPCA1'))
%% load original image
path='.\Datasets\';
inputs = 'Salinas';
location = [path,inputs];
load (location);
inputs2 = 'SA_E';path2='.\Edge\';
location2 = [path2,inputs2];
load (location2);
%% size of image 
[no_lines, no_rows, no_bands] = size(img);
GroundT=GroundT';
load (['.\training_indexes\sa_p1.mat'])
%% Spectral dimension Reduction
 img2=average_fusion(img,20);
 OA=[];AA=[];kappa=[];CA=[];
indexes=XX;
%% Normalization
no_bands=size(img2,3);
fimg=reshape(img2,[no_lines*no_rows no_bands]);
[fimg] = scale_new(fimg);
fimg=reshape(fimg,[no_lines no_rows no_bands]);
%% Structure extraction
 fimg1 = csp(fimg,0.006,3,0.02,4,E,0.16);
 fimg2 = csp(fimg,0.08,1,0.02,4,E,0.16);
 fimg3 = csp(fimg,0.2,2,0.02,4,E,0.16);
 f_fimg=cat(3,fimg1,fimg2,fimg3);
%% Feature fusion with the Kpca
 fimg =kpca(f_fimg, 1000,30, 'Gaussian',20);%'Gaussian'
%% SVM classification
    fimg = ToVector(fimg);
    fimg = fimg';
    fimg=double(fimg);
%%
train_SL = GroundT(:,indexes);
train_samples = fimg(:,train_SL(1,:))';
train_labels= train_SL(2,:)';
%%
test_SL = GroundT;
test_SL(:,indexes) = [];
test_samples = fimg(:,test_SL(1,:))';
test_labels = test_SL(2,:)';
%% Normalizing Training and original img 
[train_samples,M,m] = scale_func(train_samples);
[fimg ] = scale_func(fimg',M,m);
%% Selecting the paramter for SVM
[Ccv Gcv cv cv_t]=cross_validation_svm(train_labels,train_samples);
%% Training using a Gaussian RBF kernel
parameter=sprintf('-c %f -g %f -m 500 -t 2 -q',Ccv,Gcv);
model=svmtrain(train_labels,train_samples,parameter);
%% Testing
Result = svmpredict(ones(no_lines*no_rows,1),fimg,model); %%%
%% Evaluation
GroudTest = double(test_labels(:,1));
ResultTest = Result(test_SL(1,:),:);
[OA,AA,kappa,CA]=confusion(GroudTest,ResultTest);
Result = reshape(Result,no_lines,no_rows);
VClassMap=label2colord(Result,'india');
figure,imshow(VClassMap);
%% 
disp('%%%%%%%%%%%%%%%%%%% Classification Results of MSSP Method %%%%%%%%%%%%%%%%')
disp(['OA',' = ',num2str(OA),' ||  ','AA',' = ',num2str(AA),'  ||  ','Kappa',' = ',num2str(kappa)])