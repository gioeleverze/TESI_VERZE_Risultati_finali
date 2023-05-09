% This is a demo example for ELMMI with DKS, 
% train/test dataset are based on material posted on: https://github.com/chickenbestlover/RNN-Time-series-Anomaly-Detection
% Accessed: Oct, 2021.
clear; close all; clc;
rng(0); % fix the random seed

%% load the dataset 


Data1=load('./data/train');
train=Data1.data;


%Data1=load('./data/test1');

test=Data1.data;
test=train

%% Parameters to initialize 
winSize = 1;      % window size 
numNodes = 100;    % number of kernel
tau = 1;           % time difference
lambda_chosen = 0.01;    % regularization parameter
neighbor =15;       % parameter for CLA
%%
traindatawin=zeros(winSize*size(train,1),(size(train,2)-winSize+1));
testdatawin=zeros(winSize*size(train,1),size(train,2)-winSize+1);

for m=1:size(train,1)   
traindatawin(winSize*(m-1)+1:winSize*m,:) = day_sliding_window(train(m, :), winSize, tau);    
testdatawin(winSize*(m-1)+1:winSize*m,:) = day_sliding_window(train(m, :), winSize, tau);
end

traindatalength = size(traindatawin,2); 

%% Initial the obtained results %%
trainData = pagetranspose(train);
testData = pagetranspose(test);
testCount= size(testData,2);
trainCount =size(trainData,2);
measuredMI_ELMMImulti = zeros(1, testCount - 1);

%traindatalength=size(trainData,2);
%% 
%% PCA if necessary
Q = 19;  % dimension reduction parameter
trainpPCAinput = trainData';
testPCAinput = testData';
[coeff,~,~,~,~] = pca(trainpPCAinput);
PCAvector = coeff(:,1:Q);
trainPCAoutput = trainpPCAinput*PCAvector;
testPCAoutput = testPCAinput*PCAvector;

%% Start CLA clustering (Can change to other clustering methods)
daynum = 1;  % number of day in training data  
cluster_mat = zeros(daynum,1); % number of cluster for each day
%center_multiday = zeros(daynum,19*40); % mean center for each day
center_multiday = zeros(daynum,2); % mean center for each day

idx_multiday = zeros(daynum,traindatalength); % clusters allowcation for each sample each day 
center_all = cell(1,daynum); % clusters center allocation for all day

%%%%%%%%%%%% PCA training data to start CLA
%data = trainData';
%[coeff2,~,~,~,~] = pca(trainData');
data = trainData';
[coeff2,~,~,~,~] = pca(trainData');
PCAvector2 = coeff2(:,1:2);
trainPCAout2 = data*PCAvector2;
%trainPCAout2 = data;

%%%%%%%%%%%%%%%  clustering for diffirent days
for m = 1:daynum
 trainPCAoutput2 = trainPCAout2(traindatalength*(m-1)+1:traindatalength*m,:); 
 [idx,clusterNums]=CLA(trainPCAoutput2,neighbor);
 %center_perday = zeros(clusterNums,19*40);
 center_perday = zeros(clusterNums,2);
 cluster_mat(m,:) = clusterNums;
for i = 1 : clusterNums
 center_perday(i,:) = mean(trainPCAoutput2(idx == i, :),1);
end
 center_all{m} = center_perday;
 center_multiday(m,:) = mean(center_perday,1);
 idx_multiday(m,:) =idx;
end    
feats_1 = zeros(1,size(testData,1),traindatalength);   % candidate sampled training data in 3D form
for m = 1:daynum
 feats_1(m,:,:) = trainData(:,traindatalength*(m-1)+1:traindatalength*m); 
end
feats = permute(feats_1,[1 3 2]);

%% Start ELM-MI with DKS
%test_fest_pca = testData'; % Project test data to calculate distance
test_fest_pca = testData'*PCAvector2;

fprintf('Starting ELM-MI with DKS\n');
if trainCount < numNodes
   numNodes = trainCount;
end
sigmas_RBs = 1 * rand(numNodes, 1);
for j = 1 : testCount-1
 %%  DKS
 centers_RBs = select_elm_Nodes_fast(daynum, center_multiday, test_fest_pca(j,:), numNodes,center_all, feats, idx_multiday,trainData );
 %% estimate H, k, and beta
 Phi = zeros(numNodes, 1);
for ni = 1 : numNodes
 dist2_x = sum((testData(:, j) - centers_RBs(:, ni)).^2);
 Phix = exp(-dist2_x ./(2 * sigmas_RBs(ni).^2));
 dist2_y = sum((testData(:, j + 1) - centers_RBs(:, ni)).^2);
 Phiy = exp(-dist2_y ./(2 * sigmas_RBs(ni).^2));  
 Phi(ni, 1)= Phix.* Phiy;
end
 H = Phi*Phi';
 k = Phi;
 beta = pinv(H + lambda_chosen * eye(numNodes)) * k;
 %% calculate the MI as anomaly score
 I = (beta' * H * beta)/2 - (k' * beta) + 1/2; 
 measuredMI_ELMMImulti(1,j) = I;
end

%% Display anomaly scores
figure(1)
measuredMI_ELMMImulti = smooth(measuredMI_ELMMImulti,10)';
measuredMI_ELMMImulti =  [measuredMI_ELMMImulti zeros(1,winSize)];
subplot(2,1,1)
plot(test', 'r-','LineWidth',1,'MarkerSize',8);
ylabel('Value');
title('Demo Test Data  '); 
subplot(2,1,2)
plot(measuredMI_ELMMImulti, 'r-','LineWidth',1,'MarkerSize',8);
ylabel('Anomaly Score');
title('Demo Test Result '); 



writematrix(measuredMI_ELMMImulti','Result/ELM.csv')




d=dir('./data/mat/*.mat');  %lunghezza 15
fprintf(d.name)
iterator=0;
for file=1:length(d)
   % Open the FIle
%% load the dataset 
    % Data1=load('./data/app_tot');
    % test=Data1.data.X_test;
    % test=test(:,1,:);
    % test=reshape(test,[13579,19]);

    Data1=load(strcat(strcat(d(file).folder,'\'),d(file).name));
    test=Data1.data;

    %% Parameters to initialize 
    % winSize = 40;      % window size 
    % numNodes = 1000;    % number of kernel
    % tau = 1;           % time difference
    % lambda_chosen = 0.1;    % regularization parameter
    % neighbor =5;       % parameter for CLA


    %% Initial the obtained results %%

    testData = pagetranspose(test);
    testCount= size(testData,2);

    measuredMI_ELMMImulti = zeros(1, testCount - 1);

    %% 
    %% PCA if necessary
    

    testPCAinput = testData';
    [coeff,~,~,~,~] = pca(trainpPCAinput);
    PCAvector = coeff(:,1:Q);

    testPCAoutput = testPCAinput*PCAvector;

    %% Start ELM-MI with DKS
    %test_fest_pca = testData'; % Project test data to calculate distance
    test_fest_pca = testData'*PCAvector2;

    fprintf('Starting ELM-MI with DKS\n');
    if trainCount < numNodes
       numNodes = trainCount;
    end
    sigmas_RBs = 1 * rand(numNodes, 1);
    for j = 1 : testCount-1
     %%  DKS
     centers_RBs = select_elm_Nodes_fast(daynum, center_multiday, test_fest_pca(j,:), numNodes,center_all, feats, idx_multiday,trainData );
     %% estimate H, k, and beta
     Phi = zeros(numNodes, 1);
    for ni = 1 : numNodes
     dist2_x = sum((testData(:, j) - centers_RBs(:, ni)).^2);
     Phix = exp(-dist2_x ./(2 * sigmas_RBs(ni).^2));
     dist2_y = sum((testData(:, j + 1) - centers_RBs(:, ni)).^2);
     Phiy = exp(-dist2_y ./(2 * sigmas_RBs(ni).^2));  
     Phi(ni, 1)= Phix.* Phiy;
    end
     H = Phi*Phi';
     k = Phi;
     beta = pinv(H + lambda_chosen * eye(numNodes)) * k;
     %% calculate the MI as anomaly score
     I = (beta' * H * beta)/2 - (k' * beta) + 1/2; 
     measuredMI_ELMMImulti(1,j) = I;
    end

    %% Display anomaly scores
    figure(1)
    measuredMI_ELMMImulti = smooth(measuredMI_ELMMImulti,10)';
    measuredMI_ELMMImulti =  [measuredMI_ELMMImulti zeros(1,winSize)];
    subplot(2,1,1)
    plot(test', 'r-','LineWidth',1,'MarkerSize',8);
    ylabel('Value');
    title('Demo Test Data  '); 
    subplot(2,1,2)
    plot(measuredMI_ELMMImulti, 'r-','LineWidth',1,'MarkerSize',8);
    ylabel('Anomaly Score');
    title('Demo Test Result '); 


    stringa=strcat(int2str(iterator),'.csv');
    stringa=strcat('Result/',stringa);
    writematrix(measuredMI_ELMMImulti',stringa);
    iterator=iterator+1;

end


