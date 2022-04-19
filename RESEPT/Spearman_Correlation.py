import numpy as np
import pandas as pd
import os
from scipy.stats import spearmanr
import argparse

parser = argparse.ArgumentParser(description='Main Entrance of MP_MIM_RESEPT')
parser.add_argument('--sampleName', type=str, default='151507')
parser.add_argument('--MP-k-num', type=int, default=90, help='number of k_num in KNN graph of message passing (default: 90)')
parser.add_argument('--MP-l-num', type=int, default=15, help='number of layer_num in message passing (default: 15)')
args = parser.parse_args()


if __name__ == '__main__':
    # sample init
    sample = args.sampleName
    k_num_distance_att = args.MP_k_num
    layer_num_distance_att = args.MP_l_num
    ground_truth_folder_path = os.path.abspath('./Embedding_Ground_Truth_Quality_Rank_'+sample+'/')
    embedding_in_RESEPT_folder = "RESEPT_MP_embedding_"+sample+"/"
    ####sample list
    sample_list = [ '151507','151508', '151509', '151510', '151669', '151670', '151671', '151672', '151673', '151674', '151675', '151676','18-64','2-5', '2-8', 'T4857']
    letter_list = [ 'a','b', 'c', 'd', 'e', 'f', 'g', 'h', 'i', 'j', 'k', 'l','m', 'n', 'o', 'p']
    PEalphaList = ['0.1','0.2','0.3', '0.5', '1.0', '1.2', '1.5','2.0']
    zdimList = ['3','10', '16','32', '64', '128', '256']
    count_init = sample_list.index(sample)
    count = 56*count_init
    letter = letter_list[count_init]
    legend_name_list = []
    
    for i in range(len(PEalphaList)):
        for j in range((len(zdimList))):
            count = count + 1
            embedding_name = sample+'_'+letter+'_'+str(count)+'_raw_PEalpha'+str(PEalphaList[i])+'_zdim'+str(zdimList[j])
            legend_name_list.append(embedding_name)
    
    # Ground Truth
    raw_embedding_kmeans_ari_result_df = pd.read_csv(ground_truth_folder_path+'/'+sample+'_raw_embedding_ground_truth_rank.csv', index_col = 0)
    raw_embedding_kmeans_ari_result_df_T = raw_embedding_kmeans_ari_result_df.T
    raw_embedding_kmeans_ari_result_name_list = []
    for i in range(56):
        raw_embedding_kmeans_ari_result_name_list.append(raw_embedding_kmeans_ari_result_df_T.iloc[0].values[i].split('_')[0]+'_'+raw_embedding_kmeans_ari_result_df_T.iloc[0].values[i].split('_')[1]+'_'+raw_embedding_kmeans_ari_result_df_T.iloc[0].values[i].split('_')[2]+'_'+raw_embedding_kmeans_ari_result_df_T.iloc[0].values[i].split('_')[3]+'_'+raw_embedding_kmeans_ari_result_df_T.iloc[0].values[i].split('_')[9]+'_'+raw_embedding_kmeans_ari_result_df_T.iloc[0].values[i].split('_')[12])
    raw_embedding_kmeans_ari_result_name_np = np.array(raw_embedding_kmeans_ari_result_name_list)
    raw_embedding_kmeans_ari_result_name_init_np = np.zeros((raw_embedding_kmeans_ari_result_name_np.shape[0],2))
    for i in range(raw_embedding_kmeans_ari_result_name_np.shape[0]):
        raw_embedding_kmeans_ari_result_name_init_np[i,0] = raw_embedding_kmeans_ari_result_name_np[i].split('_')[2]
        raw_embedding_kmeans_ari_result_name_init_np[i,1] = i+1
    raw_embedding_kmeans_ari_result_name_init_np_int = raw_embedding_kmeans_ari_result_name_init_np.astype(int)
    raw_embedding_kmeans_ari_result_order = raw_embedding_kmeans_ari_result_name_init_np_int[np.argsort(raw_embedding_kmeans_ari_result_name_init_np_int[:,0])][:,1]
    
    # Spearman Correlation
    MP_MIM_csv = sample+'_gat_self_loop_euc_knn_graph_K'+str(k_num_distance_att)+'_layer'+str(layer_num_distance_att)+'_MP_MIM.csv'
    MP_MIM_result_df = pd.read_csv('./'+MP_MIM_csv,index_col=0)
    MP_MIM_result = MP_MIM_result_df.T.values[1,:]
    MP_MIM_result_sort_descending = pd.DataFrame(MP_MIM_result.reshape(-1,len(MP_MIM_result)),columns=legend_name_list).sort_values(by=0,axis=1,ascending=False)
    MP_MIM_result_sort_descending_np = np.array(list(MP_MIM_result_sort_descending))
    MI_MIM_init_np = np.zeros((MP_MIM_result_sort_descending_np.shape[0],2))
    for k in range(MP_MIM_result_sort_descending_np.shape[0]):
        MI_MIM_init_np[k,0] = MP_MIM_result_sort_descending_np[k].split('_')[2]
        MI_MIM_init_np[k,1] = k+1
    MI_MIM_init_np_int = MI_MIM_init_np.astype(int)
    MI_MIM_order = MI_MIM_init_np_int[np.argsort(MI_MIM_init_np_int[:,0])][:,1]
    Spearman_correlation,pvalue=spearmanr(raw_embedding_kmeans_ari_result_order,MI_MIM_order)
    print(sample+' Spearman correlation is '+str(Spearman_correlation))
