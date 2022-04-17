
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

parser = argparse.ArgumentParser(description='Main Entrance of MP_MIM_RESEPT')
parser.add_argument('--sampleName', type=str, default='151507')
parser.add_argument('--MP-k-num', type=int, default=90, help='number of k_num in KNN graph of message passing (default: 90)')
parser.add_argument('--MP-l-num', type=int, default=15, help='number of layer_num in message passing (default: 15)')
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
    ########RESEPT
    ####time_computing
    start_time = time.time()
    print("MP_MIM_RESEPT. Start Time: %s seconds" %
          (start_time))
    ####parameter_set_initial
    PEalphaList = ['0.1','0.2','0.3', '0.5', '1.0', '1.2', '1.5','2.0']
    zdimList = ['3','10', '16','32', '64', '128', '256']
    sample = args.sampleName
    k_num_distance_att = args.MP_k_num
    layer_num_distance_att = args.MP_l_num
    
    ####sample_list
    sample_list = [ '151507','151508', '151509', '151510', '151669', '151670', '151671', '151672', '151673', '151674', '151675', '151676','18-64','2-5', '2-8', 'T4857']
    letter_list = [ 'a','b', 'c', 'd', 'e', 'f', 'g', 'h', 'i', 'j', 'k', 'l','m', 'n', 'o', 'p']
    count_init = sample_list.index(sample)
    count = 56*count_init
    letter = letter_list[count_init]
    embedding_MIrow_max_list = []
    embedding_name_list = []
    
    ####current_os
    meta_folder_path = os.path.abspath('./meta_data_folder/metaData_brain_16_coords')
    embedding_folder_path = os.path.abspath('./RESEPT_embedding_folder')
    embedding_in_RESEPT_folder = "RESEPT_MP_embedding_"+sample+"/"
    if not os.path.exists(embedding_in_RESEPT_folder):
        os.makedirs(embedding_in_RESEPT_folder)
    
    ####MP_parameter_set
    k_num_distance_att_list = [10,20,30,40,50,60,70,80,90]
    layer_num_distance_att_list = [1,2,3,4,5,6,7,8,9,10,11,12,13,14,15]
    
    ####loop_part
    for i in range(len(PEalphaList)):
        for j in range((len(zdimList))):
            ####read_embedding
            count = count + 1
            embedding_root_path = '/'+sample+'_embedding_raw/'+letter+'_'+str(count)+'_outputdir-3S-'+sample+'_raw_EM1_resolution0.3_euclidean_dummy_add_PEalpha'+str(PEalphaList[i])+'_k6_NA_zdim'+str(zdimList[j])+'/'+sample+'_raw_6_euclidean_NA_dummy_add_'+str(PEalphaList[i])+'_intersect_160_GridEx19_embedding.csv'
            embedding_df = pd.read_csv(embedding_folder_path+embedding_root_path,index_col=0)
            embedding_celllist = embedding_df.index.tolist()
            graph_embedding_name = sample+'_raw_res0.3_euclidean_NA_dummy_add_PEalpha'+str(PEalphaList[i])+'_k6_NA_zdim'+str(zdimList[j])+'_gat_self_loop_euc_graphK'+str(k_num_distance_att)+'_layer'+str(layer_num_distance_att)
            embedding_name_list.append(graph_embedding_name)
            native_embedding_whole = embedding_df[['embedding0','embedding1','embedding2']].values
            ####embedding_knn_graph
            knn_graph_k_num = k_num_distance_att
            l_num = layer_num_distance_att
            adj = generate_adj_nx_weighted_adj(native_embedding_whole, distanceType='euclidean', k=knn_graph_k_num)
            adj_self_loop = preprocess_graph_self_loop(adj)
            graph_embedding_whole = forward_dis_gcn_multi_layer(adj_self_loop, native_embedding_whole, l_num)
            graph_embedding_add_barcode_df = pd.DataFrame(graph_embedding_whole, index=embedding_celllist, columns=['embedding0','embedding1','embedding2'])
            graph_embedding_add_barcode_df.to_csv(embedding_in_RESEPT_folder+sample+'_'+str(count)+'_raw_res0.3_euclidean_NA_dummy_add_PEalpha'+str(PEalphaList[i])+'_k6_NA_zdim'+str(zdimList[j])+'_gat_self_loop_euc_graphK'+str(knn_graph_k_num)+'_layer'+str(l_num)+'_graph_embedding.csv')
            graph_embedding_remove_zero_df = graph_embedding_add_barcode_df.loc[~(graph_embedding_add_barcode_df==0).all(axis=1)]
            #print(graph_embedding_remove_zero_df)
            graph_embedding_remove_zero_whole = graph_embedding_remove_zero_df[['embedding0','embedding1','embedding2']].values
            coordinate_graph_embedding_whole_df = pd.read_csv(meta_folder_path+'/'+sample+'_humanBrain_metaData.csv',index_col=0)
            coordinate_graph_embedding_remove_zero_df = coordinate_graph_embedding_whole_df.loc[graph_embedding_remove_zero_df.index]
            coordinate_graph_embedding_remove_zero_np = coordinate_graph_embedding_remove_zero_df[['array_row','array_col']].values
            ####MI_spatial_adj
            MI_graph_embedding_spatial_adj = MI_spatial_adj_matrix(coordinate_graph_embedding_remove_zero_np, hop_num=1, distanceType='euclidean')
            ####MI_max
            embedding_MIrow_list = []
            for dim_num in range(graph_embedding_remove_zero_whole.shape[1]):
                embedding_current_MIrow = Moran_I(MI_graph_embedding_spatial_adj, graph_embedding_remove_zero_whole[:,dim_num], 'row_normalizaiton')
                embedding_MIrow_list.append(embedding_current_MIrow)
            embedding_MIrow_list_np = np.array(embedding_MIrow_list)
            embedding_MIrow_max_list.append(np.max(embedding_MIrow_list_np))
    ####save_result
    MIrow_result_gat_euc_df = pd.DataFrame({'embedding_name':embedding_name_list,'embedding_MP_MIM':embedding_MIrow_max_list})
    MIrow_result_gat_euc_df.to_csv(sample+'_gat_self_loop_euc_knn_graph_K'+str(k_num_distance_att_list[8])+'_layer'+str(layer_num_distance_att_list[14])+'_MP_MIM.csv')

    end_time = time.time()
    print("MP_MIM_RESEPT. End Time: %s seconds" %
            (end_time))
    print("MP_MIM_RESEPT Done. Total Running Time: %s seconds" %
            (end_time - start_time))


