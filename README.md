# MP-MIM

## Sampling and ranking spatial transcriptomics data embeddings to identify tissue architecture

MP-MIM is a novel embedding evaluation method that employs a graph convolution way for message passing and spatial autocorrelation index to evaluate embeddings from deep learning models used for spatial transcriptome analysis. It can accurately evaluate multiple embeddings with different hyperparameter settings and identify high-quality embeddings by producing a high correlation between the predicted spatial architecture and the ground truth. The main workflow of MP-MIM is shown as follows:

![MP-MIM workflow](docs/images/workflow.jpg)
As shown in (A), (B), (C) and (D), the overall workflow of integrating MP-MIM into spatial models includes four parts: (A) data preprocessing to convert the input embedding into graph-structured data, (B) embedding transformation by message passing, (C) embedding evaluation by using Moran’s I with maximum filtering and (D) the graph embeddings and ranking used for the downstream analysis of the respective models. To validate the proposed method, MP-MIM, the panel of method validation also details the baseline comparison methods and evaluation metrics.

--------------------------------------------------------------------------------

## System Requirements:

MP-MIM was implemented on a computing server with 2.2 GHz, 144 cores CPU, and 503 GB RAM under an ubuntu 18.04 operating system.

### OS Requirements: 

MP-MIM can run on both Linux and Windows. The package has been tested on the Linux system.

### Install MP-MIM from Github:

```bash
git clone https://github.com/YuLin-code/MP-MIM.git
cd MP-MIM
```

### Python Dependencies: 

MP-MIM mainly depends on the Python (3.6+) scientific stack and python virutal environment with conda (<https://anaconda.org/>) is recommended.

```shell
conda create -n MP_MIM python=3.8
conda activate MP_MIM
pip install -r requirements.txt
```

### Configure RESEPT Environment:

Download and install [RESEPT](https://github.com/OSU-BMBL/RESEPT). Generate embedding and conduct tissue architecture identification. 

### Configure SpaGCN Environment:

Download and install [SpaGCN](https://github.com/jianhuupenn/SpaGCN). Integrate MP-MIM into SpaGCN and conduct tissue architecture identification. 


## Demo:

### 1. Set Baseline

The deep learning model RESEPT was used as an embedding sampling method to generate embeddings. 56 three-dimensional embeddings with different hyperparameter settings were generated and saved in the source code folder for 16 samples. Run the following command to generate the ground truth ranking and we take '151507' as the example sample.

```bash
cd RESEPT
python Embedding_Ground_Truth_Quality_Rank.py --sampleName 151507
```

### 2. MP-MIM Rank

We take an example of an analysis when k_num is 90 and l_num is 15 in MP-MIM. Here we use hyperparameters to demo purposes:

- **MP-k-num** defines the number of nearest neighbors in the KNN graph.
- **MP-l-num** defines the number of iteration in message passing. 

If you want to reproduce results in the manuscript, please use default hyperparameters.

```bash
python MP_MIM_RESEPT.py --sampleName 151507 --MP-l-num 15 --MP-k-num 90
```

### 3. Spearman Correlation Analysis

MP-MIM can accurately identify the embeddings with different qualities by producing a high correlation between the output ranking and the ground truth. Run the following command line to calculate Spearman correlation: 

```bash
python Spearman_Correlation.py --sampleName 151507 --MP-l-num 15 --MP-k-num 90
```

### 4. Integrate MP-MIM into SpaGCN

The native SpaGCN and the SpaGCN integrated MP-MIM versions are in the source code. Please put the original dataset that contains the spatial and metadata information of each sample into the path of running SpaGCN with supporting for tissue architecture identification. Run the following command line to obtain the ARI of each SpaGCN version: 

```bash
cd ../SpaGCN/Native_SpaGCN
python SpaGCN_Native.py --sampleName 151507
cd ../SpaGCN_MP_MIM
python SpaGCN_MP_MIM.py --sampleName 151507
```

More information can be checked at the [tutorial](https://github.com/YuLin-code/MP-MIM/tree/master/tutorial).

## References:

1. GAT <https://github.com/Diego999/pyGAT>
2. Moran’s I and Geary's C <https://github.com/pysal>
3. RESEPT <https://github.com/OSU-BMBL/RESEPT>
4. SpaGCN <https://github.com/jianhuupenn/SpaGCN>
