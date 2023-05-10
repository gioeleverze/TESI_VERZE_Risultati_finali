function [measuredMI_ELMMImulti] = algorithm(train,test)

%% Parameters to initialize 
winSize = 2;      % window size 
numNodes = 100;    % number of kernel
tau = 1;           % time difference
lambda_chosen = 0.01;    % regularization parameter
neighbor =5;       % parameter for CLA
    %% Minmax & Windowing the training / test data
traindataori = zeros(size(train,1),size(train,2));
testdataori = zeros(size(test,1),size(test,2));
for i=1:size(train,1)
    min_x = min(train(i,:));
    max_x = max(train(i,:));
    max_x(max_x == 0) = 1;
    traindataori(i,:) = (train(i,:) - min_x) ./ (max_x - min_x);    
    testdataori(i,:) = (test(i,:) - min_x) ./ (max_x - min_x);    
end

traindatawin=zeros(winSize*size(train,1),(size(traindataori,2)-winSize+1));
testdatawin=zeros(winSize*size(train,1),size(testdataori,2)-winSize+1);

for m=1:size(train,1)   
traindatawin(winSize*(m-1)+1:winSize*m,:) = day_sliding_window(traindataori(m, :), winSize, tau);    
testdatawin(winSize*(m-1)+1:winSize*m,:) = day_sliding_window(testdataori(m, :), winSize, tau);
end
traindatalength = size(traindatawin,2); 

%% PCA if necessary
Q = 10;  % dimension reduction parameter
trainpPCAinput = traindatawin';
testPCAinput = testdatawin';
[coeff,~,~,~,~] = pca(trainpPCAinput);
PCAvector = coeff(:,1:Q);
trainPCAoutput = trainpPCAinput*PCAvector;
testPCAoutput = testPCAinput*PCAvector;
%% Initial the obtained results %%
trainData = trainPCAoutput';
testData = testPCAoutput';
testCount= size(testData,2);
trainCount =size(trainData,2);
measuredMI_ELMMImulti = zeros(1, testCount - 1);
    
%% Start CLA clustering (Can change to other clustering methods)
daynum = 1;  % number of day in training data  
cluster_mat = zeros(daynum,1); % number of cluster for each day
center_multiday = zeros(daynum,2); % mean center for each day
idx_multiday = zeros(daynum,traindatalength); % clusters allowcation for each sample each day 
center_all = cell(1,daynum); % clusters center allocation for all day

%%%%%%%%%%%% PCA training data to start CLA
data = trainData';
[coeff2,~,~,~,~] = pca(trainData');
PCAvector2 = coeff2(:,1:2);
trainPCAout2 = data*PCAvector2;

%%%%%%%%%%%%%%%  clustering for diffirent days
for m = 1:daynum
 trainPCAoutput2 = trainPCAout2(traindatalength*(m-1)+1:traindatalength*m,:); 
 [idx,clusterNums]=CLA(trainPCAoutput2,neighbor);
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
test_fest_pca = testData'*PCAvector2; % Project test data to calculate distance
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

measuredMI_ELMMImulti = smooth(measuredMI_ELMMImulti,50)';
measuredMI_ELMMImulti =  [measuredMI_ELMMImulti zeros(1,winSize)];








end

