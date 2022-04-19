import pandas as pd
import numpy as np
import os
from scipy.spatial import distance
import networkx as nx
import math
import scipy.sparse as sp
from glob import glob
import time
import shutil

####KNN
# knn_graph_edgelist
def calculateKNNgraphDistanceWeighted(featureMatrix, distanceType, k):
    edgeListWeighted = []
    for i in np.arange(featureMatrix.shape[0]):
        tmp = featureMatrix[i, :].reshape(1, -1)
        distMat = distance.cdist(tmp, featureMatrix, distanceType)
        res = distMat.argsort()[:k + 1]
        tmpdist = distMat[0, res[0][1:k + 1]]
        boundary = np.mean(tmpdist) + np.std(tmpdist)
        for j in np.arange(1, k + 1):
            if distMat[0, res[0][j]] <= boundary and i != res[0][j]:
                edgeListWeighted.append((i, res[0][j], 1))
    return edgeListWeighted

# generate_adj_nx_matirx
def generate_adj_nx_weighted_adj(featureMatrix, distanceType, k):
    edgeList = calculateKNNgraphDistanceWeighted(featureMatrix, distanceType, k)
    nodes = range(0,featureMatrix.shape[0])
    Gtmp = nx.Graph()
    Gtmp.add_nodes_from(nodes)
    Gtmp.add_weighted_edges_from(edgeList)
    adj = nx.adjacency_matrix(Gtmp)
    adj_knn_by_feature = np.array(adj.todense())
    return adj_knn_by_feature

# generate_self_loop_adj
def preprocess_graph_self_loop(adj):
    adj = sp.coo_matrix(adj)
    adj_ = adj + sp.eye(adj.shape[0])
    adj_ = adj_.A
    return adj_

####MP
# attention_ave
def gat_forward_att_ave(adj, Wh):
    attention_ave = adj
    attention_ave_par = attention_ave.sum(axis=1, keepdims=True)
    attention_ave_final = attention_ave/attention_ave_par
    h_prime = np.dot(attention_ave_final, Wh)
    return h_prime

##layer_loop_att_ave
def forward_gat_multi_layer(adj, Wh, layer_num):
    hidden = Wh
    for num in range(layer_num):
        h = gat_forward_att_ave(adj , hidden)
        hidden = h
        #print(num)
    return hidden

####MI_GC
# MI
def Moran_I(multi_hop_weight_mat, feature, MI_type='normal'):
    
    if MI_type == 'normal':
        w = multi_hop_weight_mat
        y = feature
        n = len(y)
        z = y - y.mean()
        z2ss = (z * z).sum()
        s0 = np.sum(w)
        zl = np.dot(w , z)
        inum = (z * zl).sum()
        MI = n / s0 * inum / z2ss
    
    if MI_type == 'row_normalizaiton':
        WR_temp = multi_hop_weight_mat
        WR = np.zeros((WR_temp.shape[0],WR_temp.shape[1]))
        each_row_sum_list=[]
        for i in range(WR_temp.shape[0]):
            each_row_sum_list.append(np.sum(WR_temp[i,:]))
        for i in range(WR_temp.shape[0]):
            for j in range(WR_temp.shape[1]):
                if WR_temp[i,j] != 0:
                    WR[i,j] = WR_temp[i,j]/each_row_sum_list[i]
        w = WR
        y = feature
        n = len(y)
        z = y - y.mean()
        z2ss = (z * z).sum()
        s0 = np.sum(w)
        zl = np.dot(w , z)
        inum = (z * zl).sum()
        MI = n / s0 * inum / z2ss
    return MI

# GC
def GC_related(multi_hop_weight_mat, feature, GC_type='normal'):
    
    if GC_type == 'normal':
        w = multi_hop_weight_mat
        y = np.asarray(feature).flatten()
        n = len(y)
        s0 = np.sum(w)
        yd = y - y.mean()
        yss = sum(yd * yd)
        den = yss * s0 * 2.0
        _focal_ix, _neighbor_ix = w.nonzero()
        _weights = csr_matrix(w).data
        num = (_weights * ((y[_focal_ix] - y[_neighbor_ix])**2)).sum()
        a = (n - 1) * num
        GC = a / den
        if GC > 1:
            GC_related = GC - 1
        if GC < 1:
            GC_related = 1 - GC
        if GC == 1:
            GC_related = 0
    
    if GC_type == 'row_normalizaiton':
        WR_temp = multi_hop_weight_mat
        WR = np.zeros((WR_temp.shape[0],WR_temp.shape[1]))
        each_row_sum_list=[]
        for i in range(WR_temp.shape[0]):
            each_row_sum_list.append(np.sum(WR_temp[i,:]))
        for i in range(WR_temp.shape[0]):
            for j in range(WR_temp.shape[1]):
                if WR_temp[i,j] != 0:
                    WR[i,j] = WR_temp[i,j]/each_row_sum_list[i]
        w = WR
        y = np.asarray(feature).flatten()
        n = len(y)
        s0 = np.sum(w)
        yd = y - y.mean()
        yss = sum(yd * yd)
        den = yss * s0 * 2.0
        _focal_ix, _neighbor_ix = w.nonzero()
        _weights = csr_matrix(w).data
        num = (_weights * ((y[_focal_ix] - y[_neighbor_ix])**2)).sum()
        a = (n - 1) * num
        GC = a / den
        if GC > 1:
            GC_related = GC - 1
        if GC < 1:
            GC_related = 1 - GC
        if GC == 1:
            GC_related = 0
    return GC_related

# spatial_adj_knn
def calculateKNNDistanceWeighted_spatial_autocor(featureMatrix, distanceType, k):
    edgeListWeighted = []
    for i in np.arange(featureMatrix.shape[0]):
        tmp = featureMatrix[i, :].reshape(1, -1)
        distMat = distance.cdist(tmp, featureMatrix, distanceType)
        res = distMat.argsort()[:k + 1]
        for j in np.arange(1, k + 1):
            edgeListWeighted.append((i, res[0][j], 1))
    return edgeListWeighted

# generate_adj_nx_matirx
def generate_spatial_adj_nx_weighted_based_on_coordinate(featureMatrix, distanceType, k):
    edgeList = calculateKNNDistanceWeighted_spatial_autocor(featureMatrix, distanceType, k)
    nodes = range(0,featureMatrix.shape[0])
    Gtmp = nx.Graph()
    Gtmp.add_nodes_from(nodes)
    Gtmp.add_weighted_edges_from(edgeList)
    adj = nx.adjacency_matrix(Gtmp)
    adj_knn_by_coordinate = np.array(adj.todense())
    return adj_knn_by_coordinate


####SpaGCN
sample = '151507'      ##sample init
# current_os
meta_folder_path = os.path.abspath('./meta_data_folder/metaData_brain_16_coords')
embedding_in_SpaGCN_folder = "./SpaGCN_MP_embedding_"+sample+"/"
if not os.path.exists(embedding_in_SpaGCN_folder):
    os.makedirs(embedding_in_SpaGCN_folder)

start_time = time.time()
print("MP_MIM_SpaGCN. Start Time: %s seconds" %
        (start_time))

####MP_parameter_set
MP_k_num_list = [1,2,3,4,5,6,7,8,9]
MP_l_num_list = [1,2,3,4,5,6,7,8,9,10,11,12,13,14,15]

####loop_part
features_np = embedding_input    ##embedding generated in SpaGCN
barcode_list = adata_barcode_list    ##barcode_list of current sample

emblist = []
for i in range(features_np.shape[1]):
    emblist.append('embedding'+str(i)) 
embedding_MIrow_max_list = []
count=0
count_list = []
for MP_k_num_in in MP_k_num_list:
    for MP_l_num_in in MP_l_num_list:
        count=count+1
        count_list.append(count)
        adj = generate_adj_nx_weighted_adj(features_np, distanceType='euclidean', k=MP_k_num_in)
        adj_self_loop = preprocess_graph_self_loop(adj)
        features_graph_np = forward_gat_multi_layer(adj_self_loop, features_np, layer_num=MP_l_num_in)
        metadata_features_graph_df = pd.read_csv(meta_folder_path+'/'+sample+'_humanBrain_metaData.csv',index_col=0)
        if sample == '151507' or sample == '151508' or sample == '151509' or sample == '151510' or sample == '151669' or sample == '151670' or sample == '151671' or sample == '151672' or sample == '151673' or sample == '151674' or sample == '151675' or sample == '151676':
            features_graph_df = pd.DataFrame(features_graph_np, columns=emblist, index = metadata_features_graph_df.index.tolist())
        if sample == '2-5' or sample == '2-8' or sample == '18-64' or sample == 'T4857':
            features_graph_df = pd.DataFrame(features_graph_np, columns=emblist, index = barcode_list)
        features_graph_df = pd.DataFrame(features_graph_np, columns=emblist, index = metadata_features_graph_df.index.tolist())
        features_graph_remove_zero_df = features_graph_df.loc[~(features_graph_df==0).all(axis=1)]
        features_graph_remove_zero_np = features_graph_remove_zero_df[emblist].values
        features_graph_remove_zero_df.to_csv(embedding_in_SpaGCN_folder+sample+'_'+str(count)+'_spagcn_default_raw_gat_ave_MP_k'+str(MP_k_num_in)+'_l'+str(MP_l_num_in)+'_embedding.csv')
        ####MI_max
        coordinate_features_graph_remove_zero_df = metadata_features_graph_df.loc[features_graph_remove_zero_df.index]
        coordinate_features_graph_remove_zero_np = coordinate_features_graph_remove_zero_df[['array_row','array_col']].values
        MI_spatial_adj_knn = generate_spatial_adj_nx_weighted_based_on_coordinate(coordinate_features_graph_remove_zero_np, distanceType='euclidean', k=4)
        embedding_MIrow_list = []
        for i in range(features_graph_remove_zero_np.shape[1]):
            embedding_current_MIrow = Moran_I(MI_spatial_adj_knn, features_graph_remove_zero_np[:,i], 'row_normalizaiton')
            embedding_MIrow_list.append(embedding_current_MIrow)
        embedding_MIrow_list_np = np.array(embedding_MIrow_list)
        embedding_MIrow_max_list.append(np.max(embedding_MIrow_list_np))
embedding_MIrow_max_np = np.array(embedding_MIrow_max_list).reshape(len(embedding_MIrow_max_list),-1)
count_np = np.array(count_list).reshape(len(count_list),-1)
embedding_MIrow_max_with_count_np = np.hstack((embedding_MIrow_max_np,count_np))
count_set = embedding_MIrow_max_with_count_np[np.argsort(embedding_MIrow_max_with_count_np[:,0])][len(count_list)-1,1]
embedding_name = glob(os.path.join(embedding_in_SpaGCN_folder,sample+'_'+str(int(count_set))+'_*.csv'))
graph_embedding_df = pd.read_csv(embedding_name[0])
graph_embedding_np = graph_embedding_df[emblist].values    ####transformed embedding for downstream analysis
if os.path.exists(embedding_in_SpaGCN_folder):
    shutil.rmtree(embedding_in_SpaGCN_folder)

end_time = time.time()
print("MP_MIM_SpaGCN. End Time: %s seconds" %
        (end_time))
print("MP_MIM_SpaGCN Done. Total Running Time: %s seconds" %
        (end_time - start_time))
