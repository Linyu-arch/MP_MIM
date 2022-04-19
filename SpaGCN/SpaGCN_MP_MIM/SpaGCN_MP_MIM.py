import os,json
import pandas as pd
import numpy as np
import scanpy as sc
from SpaGCN2.util import prefilter_specialgenes, prefilter_genes
from generate_embedding import generate_spagcn_output
from util import filter_panelgenes
import random, torch
import warnings
warnings.filterwarnings("ignore")
from sklearn.metrics.cluster import adjusted_rand_score
import argparse
import time

random.seed(200)
torch.manual_seed(200)
np.random.seed(200)

parser = argparse.ArgumentParser(description='SpaGCN Integrated MP-MIM')
parser.add_argument('--sampleName', type=str, default='151507')
args = parser.parse_args()


#sample_list = [ '151507', '151508','151509', '151510', '151669', '151670', '151671', '151672', '151673', '151674',
#               '151675', '151676', '2-5', '2-8', '18-64', 'T4857']


if __name__ == '__main__':

    sample = args.sampleName
    data_path = "./original/"+sample+"/"
    h5_path = "filtered_feature_bc_matrix.h5"
    scale_factor_path = "spatial/scalefactors_json.json"
    spatial_path = "spatial/tissue_positions_list.csv"
    
    # --------------------------------------------------------------------------------------------------------#
    # -------------------------------load data--------------------------------------------------#
    # Read in gene expression and spatial location
    adata = sc.read_10x_h5(os.path.join(data_path,h5_path))
    spatial_all=pd.read_csv(os.path.join(data_path,spatial_path),sep=",",header=None,na_filter=False,index_col=0)
    spatial = spatial_all[spatial_all[1] == 1]
    spatial = spatial.sort_values(by=0)
    assert all(adata.obs.index == spatial.index)
    adata.obs["in_tissue"]=spatial[1]
    adata.obs["array_row"]=spatial[2]
    adata.obs["array_col"]=spatial[3]
    adata.obs["pxl_col_in_fullres"]=spatial[4]
    adata.obs["pxl_row_in_fullres"]=spatial[5]
    adata.obs.index.name = 'barcode'
    adata.var_names_make_unique()
    
    # Read scale_factor_file
    with open(os.path.join(data_path,scale_factor_path)) as fp_scaler:
        scaler = json.load(fp_scaler)
    adata.uns["spot_diameter_fullres"] = scaler["spot_diameter_fullres"]
    adata.uns["tissue_hires_scalef"] = scaler["tissue_hires_scalef"]
    adata.uns["fiducial_diameter_fullres"] = scaler["fiducial_diameter_fullres"]
    adata.uns["tissue_lowres_scalef"] = scaler["tissue_lowres_scalef"]
    
    pca_opt = True
    optical_img_path = None
    # -------------------------threshold
    #adata.X[adata.X < threshold]= 0
    
    # filter
    prefilter_genes(adata,min_cells=3) # avoiding all genes are zeros
    prefilter_specialgenes(adata)
    print('load data finish')

    #--------------------------------------------------------------------------------------------------------#
    # -------------------------------calculate_ARI--------------------------------------------------#
    adata_barcode_list = adata.obs.index.tolist()
    embedding, y_pred, prob = generate_spagcn_output(adata, pca=50, res=0.4,dim = 50, img_path=optical_img_path, pca_opt=pca_opt, sample=sample)
    
    if sample == '151507' or sample == '151508' or sample == '151509' or sample == '151510' or sample == '151669' or sample == '151670' or sample == '151671' or sample == '151672' or sample == '151673' or sample == '151674' or sample == '151675' or sample == '151676':
        # ground truth label
        meta_folder_path = os.path.abspath('./meta_data_folder/metaData_brain_16_coords')
        meta_df = pd.read_csv(meta_folder_path+'/'+sample+'_humanBrain_metaData.csv')
        ground_truth_init_np = np.array(meta_df['benmarklabel'])
        ground_truth_label_np = np.zeros((len(ground_truth_init_np),))
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
        ari = adjusted_rand_score(y_pred , ground_truth_label_np)
        print(sample+' ari is '+str(ari))
    
    if sample == '2-5' or sample == '2-8' or sample == '18-64' or sample == 'T4857':
        # ground truth label
        meta_folder_path = os.path.abspath('./meta_data_folder/metaData_brain_16_coords')
        meta_df = pd.read_csv(meta_folder_path+'/'+sample+'_humanBrain_metaData.csv',index_col=0)
        meta_df_barcode_sort = meta_df.loc[adata_barcode_list]
        ground_truth_init_np = np.array(meta_df_barcode_sort['benmarklabel'])
        ground_truth_label_np = np.zeros((len(ground_truth_init_np),))
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
        ari = adjusted_rand_score(y_pred , ground_truth_label_np)
        print(sample+' ari is '+str(ari))

