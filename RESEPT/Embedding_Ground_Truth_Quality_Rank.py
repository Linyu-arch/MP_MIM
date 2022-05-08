import numpy as np
import pandas as pd
import os
from sklearn.cluster import KMeans
from sklearn.metrics.cluster import adjusted_rand_score
import argparse


parser = argparse.ArgumentParser(description='Embedding_Ground_Truth_Quality_Rank_')
parser.add_argument('--sampleName', type=str, default='151507')
args = parser.parse_args()


if __name__ == '__main__':
    sample = args.sampleName
    meta_folder_path = os.path.abspath('./meta_data_folder/metaData_brain_16_coords') 
    embedding_folder_path = os.path.abspath('./RESEPT_embedding_folder')
    output_folder_path = os.path.abspath('./Embedding_Ground_Truth_Quality_Rank_'+sample+'/')
    if not os.path.exists(output_folder_path):
        os.makedirs(output_folder_path)
    
    knn_distanceList=['euclidean']
    PEalphaList = ['0.1','0.2','0.3', '0.5', '1.0', '1.2', '1.5','2.0']
    zdimList = ['3','10', '16','32', '64', '128', '256']
    ####sample_list
    sample_list = [ '151507','151508', '151509', '151510', '151669', '151670', '151671', '151672', '151673', '151674', '151675', '151676','18-64','2-5', '2-8', 'T4857']
    letter_list = [ 'a','b', 'c', 'd', 'e', 'f', 'g', 'h', 'i', 'j', 'k', 'l','m', 'n', 'o', 'p']
    count_init = sample_list.index(sample)
    count = 56*count_init
    letter = letter_list[count_init]
    
    embedding_name_list = []
    ari_result_list = []
    n_clusters_num = 7
    
    if sample=='151669' or sample=='151670' or sample=='151671' or sample=='151672':
        n_clusters_num = 5
    if sample=='2-8':
        n_clusters_num = 6
    
    for i in range(len(PEalphaList)):
        for j in range((len(zdimList))):
            count = count + 1
            embedding_root_path = '/'+sample+'_embedding_raw/'+letter+'_'+str(count)+'_outputdir-3S-'+sample+'_raw_EM1_resolution0.3_euclidean_dummy_add_PEalpha'+str(PEalphaList[i])+'_k6_NA_zdim'+str(zdimList[j])+'/'+sample+'_raw_6_euclidean_NA_dummy_add_'+str(PEalphaList[i])+'_intersect_160_GridEx19_embedding.csv'
            meta_df = pd.read_csv(meta_folder_path+'/'+sample+'_humanBrain_metaData.csv')
            embedding_df = pd.read_csv(embedding_folder_path+embedding_root_path)
            embedding_name = sample+'_'+letter+'_'+str(count)+'_raw_res0.3_euclidean_NA_dummy_add_PEalpha'+str(PEalphaList[i])+'_k6_NA_zdim'+str(zdimList[j])
            embedding_name_list.append(embedding_name)
            
            X = embedding_df[['embedding0','embedding1','embedding2']].values
            #X = embedding_df.iloc[:,1:4].values
            print(X.shape)
            kmeans = KMeans(n_clusters=n_clusters_num, random_state=0).fit(X)
            kmeans_label = kmeans.labels_
            print(kmeans_label)
            
            ground_truth_init_np = np.array(meta_df['benmarklabel'])
            ground_truth_label_np = np.zeros((len(ground_truth_init_np),))
    
            if sample == '2-5' or sample == '2-8' or sample == '18-64' or sample == 'T4857':
                for k in range(len(ground_truth_init_np)):
                    if ground_truth_init_np[k] == 'Layer 1':
                        ground_truth_label_np[k] = 1
                    if ground_truth_init_np[k] == 'Layer 2':
                        ground_truth_label_np[k] = 2
                    if ground_truth_init_np[k] == 'Layer 3':
                        ground_truth_label_np[k] = 3
                    if ground_truth_init_np[k] == 'Layer 4':
                        ground_truth_label_np[k] = 4
                    if ground_truth_init_np[k] == 'Layer 5':
                        ground_truth_label_np[k] = 5
                    if ground_truth_init_np[k] == 'Layer 6':
                        ground_truth_label_np[k] = 6
                    if ground_truth_init_np[k] == 'White matter' or ground_truth_init_np[k] == 'Noise' or ground_truth_init_np[k] is np.NAN:
                        ground_truth_label_np[k] = 0
            if sample == '151507' or sample == '151508' or sample == '151509' or sample == '151510' or sample == '151669' or sample == '151670' or sample == '151671' or sample == '151672' or sample == '151673' or sample == '151674' or sample == '151675' or sample == '151676':
                for k in range(len(ground_truth_init_np)):
                    if ground_truth_init_np[k] == 'Layer1':
                        ground_truth_label_np[k] = 1
                    if ground_truth_init_np[k] == 'Layer2':
                        ground_truth_label_np[k] = 2
                    if ground_truth_init_np[k] == 'Layer3':
                        ground_truth_label_np[k] = 3
                    if ground_truth_init_np[k] == 'Layer4':
                        ground_truth_label_np[k] = 4
                    if ground_truth_init_np[k] == 'Layer5':
                        ground_truth_label_np[k] = 5
                    if ground_truth_init_np[k] == 'Layer6':
                        ground_truth_label_np[k] = 6
                    if ground_truth_init_np[k] == 'WM' or ground_truth_init_np[k] is np.NAN:
                        ground_truth_label_np[k] = 0
            print(ground_truth_label_np)
            ari = adjusted_rand_score(kmeans_label , ground_truth_label_np)
            ari_result_list.append(ari)
    order_num_list = []
    for l in range(len(ari_result_list)):
        order_num_list.append(l+1)
    order_num_pd = pd.DataFrame({'Order_num':order_num_list})
    ARI_k_means_result = pd.DataFrame({'Name':embedding_name_list,'ARI_k_means':ari_result_list})
    ARI_k_means_result_sort = ARI_k_means_result.sort_values(by=['ARI_k_means'], ascending=False)
    ARI_k_means_result_sort.index = order_num_list
    ARI_k_means_result_sort.to_csv(output_folder_path+'/'+sample+'_raw_embedding_ground_truth_rank.csv')

