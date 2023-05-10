function node_feats = select_elm_Nodes_fast(daynum, center_multiday, test_fest_pca, numNodes, center_all, feats, cluster_idxs,trainData )
%  This is the demo example for DKS used in ELMMI with DKS
%  Inputs:
%           daynum: number of day in training data
%           center_multiday: mean center for each day
%           test_fest_pca: test input
%           numNodes: number of kernel
%           center_all: clusters center allocation for all day 
%           feats: candidate sampled training data in 3D form: D x N x C, D is the number of days, N is the number of samples in each day, 
%           C is number of dimension
%           cluster_idxs: clusters allowcation for each sample each day 
%           trainData: sampled training data
%
%  Output:
%            node_feats: slectected kernel centeres under DSK
%% Determine the number of selected kernel for each day
D = daynum;
N = size(feats, 2);
C = size(feats, 3);
%Calculate the number of selected kernels in each day
node_per_day = cal_num_by_dist(center_multiday, test_fest_pca, numNodes);
node_feats = zeros(C,numNodes);
sidx = 1;
for di = 1 : D
 tnum_node = node_per_day(di); %% number of node perday
 if tnum_node <= 0
    continue;
 end
%Calculate the number of selected kernel center in each cluster
 node_per_cluster = cal_num_by_dist(center_all{di}, test_fest_pca, node_per_day(di));
 assert(sum(node_per_cluster) == tnum_node);
%Select kernel center from sampled training data 
for ci = 1 : numel(node_per_cluster)
 if node_per_cluster(ci,:)>0
    nidxs = find(cluster_idxs(di,:) == ci);
    nidxs = nidxs(randi(size(nidxs, 2), node_per_cluster(ci), 1));
    nidxs = N * (di - 1) + nidxs;
    node_feats(:,sidx : sidx + node_per_cluster(ci) - 1) = trainData(: , nidxs);  % assign centers
    sidx = sidx + node_per_cluster(ci);
 end
end
    
end

end
%% Determine the number of kernel center by distance
function num_nodes = cal_num_by_dist(centers, target, quota)
 dists = sqrt(sum((centers - target).^2, 2)).^3;
 dists = 1.0 ./ dists;
 num_nodes = zeros(size(centers, 1), 1);
 num_nodes(1 : end - 1) = floor(quota * dists (1 : end - 1) ./ sum(dists));
 num_nodes(end) = quota - sum(num_nodes);
 num_nodes(num_nodes < 0) = 0;
end


