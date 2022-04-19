import os,json
import pandas as pd
import scanpy as sc
from pipeline_transform_embedding_to_image import transform_embedding_to_image
from inpaint_images import inpaint
import warnings
import argparse
warnings.filterwarnings("ignore")


parser = argparse.ArgumentParser(description='Transform Embedding to RGB Image')
parser.add_argument('--sampleName', type=str, default='151507')
args = parser.parse_args()



if __name__ == '__main__':
    sample = args.sampleName
    embedding_path = "./RESEPT_MP_embedding_"+sample+"/"
    for embedding_file in os.listdir(embedding_path):
        sample = embedding_file.split('_')[0]
        image_name = embedding_file.split('.csv')[0]
        data_path = "./original/"+sample+"/"
        h5_path = "filtered_feature_bc_matrix.h5"
        scale_factor_path = "spatial/scalefactors_json.json"
        spatial_path = "spatial/tissue_positions_list.csv"
        pseudo_image_folder = "Embedding_RGB_Image/"+sample
        if not os.path.exists(pseudo_image_folder):
            os.makedirs(pseudo_image_folder)
    
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
    
        embedding = pd.read_csv(os.path.join(embedding_path,embedding_file),sep=",",na_filter=False,index_col=0)
        embedding.index.name = 'barcode'
        embedding = embedding.sort_values(by='barcode')
        embedding = embedding.loc[:, ['embedding0', 'embedding1', 'embedding2']].values
        adata.obsm["embedding"] = embedding
        print('generate embedding finish')
        #
        # # --------------------------------------------------------------------------------------------------------#
        # # --------------------------------transform_embedding_to_image-------------------------------------------------#
        high_img, low_img = transform_embedding_to_image(adata, image_name, pseudo_image_folder, img_type='lowres',
                                                        scale_factor_file=True)  # img_type:lowres,hires,both
        # calculate_MI(adata,  image_name)
        adata.uns["high_img"] = high_img
        adata.uns["low_img"] = low_img
        print('transform embedding to image finish')
        print('transform embedding to image finish')
    
        # --------------------------------------------------------------------------------------------------------#
        # --------------------------------inpaint image-------------------------------------------------#
        img_path = pseudo_image_folder + "/"
        inpaint_path = inpaint(img_path, adata, spatial_all)
        print('generate pseudo images finish')
