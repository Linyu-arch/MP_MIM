import pandas as pd
import numpy as np
import os
from scipy.spatial import distance
import networkx as nx
import math
import scipy.sparse as sp
from glob import glob
import argparse
import time
from scipy.sparse import csr_matrix

parser = argparse.ArgumentParser(description='Extend MP-MIM to General Case')
parser.add_argument('--MP-k-num', type=int, default=90, help='number of k_num in KNN graph of message passing (default: 90)')
parser.add_argument('--MP-l-num', type=int, default=15, help='number of layer_num in message passing (default: 15)')
parser.add_argument('--MP-type', type=str, default='Distance_based_GCN', help='the type of message passing (default: Distance_based_GCN)')
args = parser.parse_args()

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
            if distMat[0, res[0][j]] <= boundary and i != res[0][j] :
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

# attention_dis
def softmax(X):
    X_exp = np.exp(X)
    partition = X_exp.sum(axis=1, keepdims=True)
    return X_exp/partition

def _prepare_euclidean_attentional_mechanism_input(Wh):
    distMat = distance.cdist(Wh, Wh, 'euclidean')
    return distMat

def gat_forward_euclidean(adj, Wh):
    e = _prepare_euclidean_attentional_mechanism_input(Wh)
    zero_vec = -9e15*np.ones_like(e)
    attention = np.where(adj > 0, e, zero_vec)
    attention = softmax(attention)
    h_prime = np.dot(attention, Wh)
    return h_prime

# layer_loop_att_ave
def forward_basic_gcn_multi_layer(adj, Wh, layer_num):
    hidden = Wh
    for num in range(layer_num):
        h = gat_forward_att_ave(adj , hidden)
        hidden = h
        #print(num)
    return hidden

# layer_loop_att_euc
def forward_dis_gcn_multi_layer(adj, Wh, layer_num):
    hidden = Wh
    for num in range(layer_num):
        h = gat_forward_euclidean(adj , hidden)
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

# spatial_adj_distance
def MI_spatial_adj_matrix(coordinateMatrix, hop_num=1, distanceType='cityblock'):
    distMat = distance.cdist(coordinateMatrix, coordinateMatrix, distanceType)
    multi_hop_weight_mat = np.zeros((distMat.shape[0] , distMat.shape[1]))
    if distanceType == 'euclidean':
        if hop_num == 1:
            for i in range(distMat.shape[0]):
                for j in range(distMat.shape[1]):
                    if distMat[i][j] <= math.sqrt(2) and distMat[i][j] > 0:
                        multi_hop_weight_mat[i][j] = 1
    return multi_hop_weight_mat


if __name__ == '__main__':
    ####time_computing
    start_time = time.time()
    print("Start Time: %s seconds" %
          (start_time))
    ####hyperparameter_set_initial
    k_num_distance_att = args.MP_k_num
    layer_num_distance_att = args.MP_l_num
    gcn_type_att = args.MP_type
    
    embedding_MP_MIM_list = []
    embedding_name_list = []
    emb0_MI_list = []
    emb1_MI_list = []
    emb2_MI_list = []
    emb0_GC_list = []
    emb1_GC_list = []
    emb2_GC_list = []
    
    ####current_os
    embedding_folder_path = os.path.abspath('./embedding_metadata_folder')
    MP_MIM_output_folder_path = os.path.abspath('./graph_embedding_folder')
    if not os.path.exists(MP_MIM_output_folder_path):
        os.makedirs(MP_MIM_output_folder_path)
    for embedding_metadata_file in os.listdir(embedding_folder_path):
        if embedding_metadata_file.split('_')[-1]=='metaData.csv':
            coordinate_embedding_whole_df = pd.read_csv(embedding_folder_path+'/'+embedding_metadata_file,index_col=0)
            print(coordinate_embedding_whole_df)
    for embedding_file in os.listdir(embedding_folder_path):
        if embedding_file.split('_')[-1]=='embedding.csv':
            embedding_name = embedding_file.split('.csv')[0]
            embedding_df = pd.read_csv(embedding_folder_path+'/'+embedding_name+'.csv',index_col=0)
            embedding_dim = embedding_df.shape[1]
            emblist = []
            for i in range(embedding_dim):
                emblist.append('embedding'+str(i))
            embedding_celllist = embedding_df.index.tolist()
            embedding_name_list.append(embedding_name)
            native_embedding_remove_zero_df = embedding_df.loc[~(embedding_df==0).all(axis=1)]
            coordinate_native_embedding_remove_zero_df = coordinate_embedding_whole_df.loc[native_embedding_remove_zero_df.index]
            coordinate_native_embedding_remove_zero_np = coordinate_native_embedding_remove_zero_df[['array_row','array_col']].values
            native_embedding_remove_zero = native_embedding_remove_zero_df[emblist].values
            native_embedding_whole = embedding_df[emblist].values
            ####embedding_knn_graph
            knn_graph_k_num = k_num_distance_att
            l_num = layer_num_distance_att
            adj = generate_adj_nx_weighted_adj(native_embedding_whole, distanceType='euclidean', k=knn_graph_k_num)
            adj_self_loop = preprocess_graph_self_loop(adj)
            if gcn_type_att == 'Distance_based_GCN':
                graph_embedding_whole = forward_dis_gcn_multi_layer(adj_self_loop, native_embedding_whole, l_num)
            if gcn_type_att == 'Basic_GCN':
                graph_embedding_whole = forward_basic_gcn_multi_layer(adj_self_loop, native_embedding_whole, l_num)
            graph_embedding_add_barcode_df = pd.DataFrame(graph_embedding_whole, index=embedding_celllist, columns=emblist)
            graph_embedding_add_barcode_df.to_csv(MP_MIM_output_folder_path+'/'+embedding_name+'_graph_embedding.csv')
            graph_embedding_remove_zero_df = graph_embedding_add_barcode_df.loc[~(graph_embedding_add_barcode_df==0).all(axis=1)]
            graph_embedding_remove_zero_np = graph_embedding_remove_zero_df[emblist].values
            coordinate_graph_embedding_remove_zero_df = coordinate_embedding_whole_df.loc[graph_embedding_remove_zero_df.index]
            coordinate_graph_embedding_remove_zero_np = coordinate_graph_embedding_remove_zero_df[['array_row','array_col']].values
            ####MI_spatial_adj
            if gcn_type_att == 'Distance_based_GCN':
                MI_native_embedding_spatial_adj = MI_spatial_adj_matrix(coordinate_native_embedding_remove_zero_np, hop_num=1, distanceType='euclidean')
                MI_graph_embedding_spatial_adj = MI_spatial_adj_matrix(coordinate_graph_embedding_remove_zero_np, hop_num=1, distanceType='euclidean')
            if gcn_type_att == 'Basic_GCN':
                MI_native_embedding_spatial_adj = generate_spatial_adj_nx_weighted_based_on_coordinate(coordinate_native_embedding_remove_zero_np, distanceType='euclidean', k=4)
                MI_graph_embedding_spatial_adj = generate_spatial_adj_nx_weighted_based_on_coordinate(coordinate_graph_embedding_remove_zero_np, distanceType='euclidean', k=4)
            ####MI
            MIrow_emb0 = Moran_I(MI_native_embedding_spatial_adj, native_embedding_remove_zero[:,0], 'row_normalizaiton')
            MIrow_emb1 = Moran_I(MI_native_embedding_spatial_adj, native_embedding_remove_zero[:,1], 'row_normalizaiton')
            MIrow_emb2 = Moran_I(MI_native_embedding_spatial_adj, native_embedding_remove_zero[:,2], 'row_normalizaiton')
            emb0_MI_list.append(MIrow_emb0)
            emb1_MI_list.append(MIrow_emb1)
            emb2_MI_list.append(MIrow_emb2)
            ####GC
            GCrow_emb0 = GC_related(MI_native_embedding_spatial_adj, native_embedding_remove_zero[:,0], 'row_normalizaiton')
            GCrow_emb1 = GC_related(MI_native_embedding_spatial_adj, native_embedding_remove_zero[:,1], 'row_normalizaiton')
            GCrow_emb2 = GC_related(MI_native_embedding_spatial_adj, native_embedding_remove_zero[:,2], 'row_normalizaiton')
            emb0_GC_list.append(GCrow_emb0)
            emb1_GC_list.append(GCrow_emb1)
            emb2_GC_list.append(GCrow_emb2)
            ####MP_MIM
            graph_embedding_MIrow_list = []
            for dim_num in range(graph_embedding_remove_zero_np.shape[1]):
                embedding_current_MIrow = Moran_I(MI_graph_embedding_spatial_adj, graph_embedding_remove_zero_np[:,dim_num], 'row_normalizaiton')
                graph_embedding_MIrow_list.append(embedding_current_MIrow)
            embedding_MIrow_list_np = np.array(graph_embedding_MIrow_list)
            embedding_MP_MIM_list.append(np.max(embedding_MIrow_list_np))
    ####save_result
    Spatial_result_df = pd.DataFrame({'embedding_name':embedding_name_list,'MI_emb0':emb0_MI_list,'MI_emb1':emb1_MI_list,'MI_emb2':emb2_MI_list, \
                                                                           'GC_emb0':emb0_GC_list,'GC_emb1':emb1_GC_list,'GC_emb2':emb2_GC_list, \
                                                                           'MP_MIM':embedding_MP_MIM_list})
    Spatial_result_df.to_csv(MP_MIM_output_folder_path+'/MI_GC_and_MP_MIM.csv')

    end_time = time.time()
    print("End Time: %s seconds" %
            (end_time))
    print("Done. Total Running Time: %s seconds" %
            (end_time - start_time))


