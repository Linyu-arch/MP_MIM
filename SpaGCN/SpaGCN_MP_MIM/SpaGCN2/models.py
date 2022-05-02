import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.parameter import Parameter
from torch.nn.modules.module import Module
from sklearn.cluster import KMeans
import torch.optim as optim
import pandas as pd
import numpy as np
import scanpy as sc
from . layers import GraphConvolution
from scipy.spatial import distance
import networkx as nx
import math
import scipy.sparse as sp
import os
import time
from glob import glob
import shutil



class simple_GC_DEC(nn.Module):
    def __init__(self, nfeat, nhid, alpha=0.2, louvain_seed=0):
        super(simple_GC_DEC, self).__init__()
        self.gc = GraphConvolution(nfeat, nhid)
        self.nhid=nhid
        #self.mu determined by the init method
        self.alpha=alpha
        self.loiuvain_seed=louvain_seed

    def forward(self, x, adj):
        x=self.gc(x, adj)
        q = 1.0 / ((1.0 + torch.sum((x.unsqueeze(1) - self.mu)**2, dim=2) / self.alpha) + 1e-8)
        q = q**(self.alpha+1.0)/2.0
        q = q / torch.sum(q, dim=1, keepdim=True)
        return x, q

    def loss_function(self, p, q):
        def kld(target, pred):
            return torch.mean(torch.sum(target*torch.log(target/(pred+1e-6)), dim=1))
        loss = kld(p, q)
        return loss

    def target_distribution(self, q):
        #weight = q ** 2 / q.sum(0)
        #return torch.transpose((torch.transpose(weight,0,1) / weight.sum(1)),0,1)e
        p = q**2 / torch.sum(q, dim=0)
        p = p / torch.sum(p, dim=1, keepdim=True)
        return p

    def fit(self, X,adj, barcode_list,lr=0.001, max_epochs=5000, update_interval=5, trajectory_interval=50,weight_decay=5e-4,opt="sgd",init="louvain",n_neighbors=10,res=0.4,n_clusters=10,init_spa=True,tol=1e-3,sample='151507'):
        self.trajectory=[]
        if opt=="sgd":
            optimizer = optim.SGD(self.parameters(), lr=lr, momentum=0.9)
        elif opt=="admin":
            optimizer = optim.Adam(self.parameters(),lr=lr, weight_decay=weight_decay)

        features= self.gc(torch.FloatTensor(X),torch.FloatTensor(adj))

        #--------------------------------------------------------------------add MP-MIM
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
        
        ##generate_adj_nx_matirx
        def generate_adj_nx_weighted_adj(featureMatrix, distanceType, k):
            edgeList = calculateKNNgraphDistanceWeighted(featureMatrix, distanceType, k)
            nodes = range(0,featureMatrix.shape[0])
            Gtmp = nx.Graph()
            Gtmp.add_nodes_from(nodes)
            Gtmp.add_weighted_edges_from(edgeList)
            adj = nx.adjacency_matrix(Gtmp)
            adj_knn_by_feature = np.array(adj.todense())
            return adj_knn_by_feature
        
        ##generate_self_loop_adj
        def preprocess_graph_self_loop(adj):
            adj = sp.coo_matrix(adj)
            adj_ = adj + sp.eye(adj.shape[0])
            adj_ = adj_.A
            return adj_
        
        ####MP
        ##attention_ave
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
        ##MI
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
        
        ##spatial_adj_knn
        def calculateKNNDistanceWeighted_spatial_autocor(featureMatrix, distanceType, k):
            edgeListWeighted = []
            for i in np.arange(featureMatrix.shape[0]):
                tmp = featureMatrix[i, :].reshape(1, -1)
                distMat = distance.cdist(tmp, featureMatrix, distanceType)
                res = distMat.argsort()[:k + 1]
                for j in np.arange(1, k + 1):
                    edgeListWeighted.append((i, res[0][j], 1))
            return edgeListWeighted
        
        ##generate_adj_nx_matirx
        def generate_spatial_adj_nx_weighted_based_on_coordinate(featureMatrix, distanceType, k):
            edgeList = calculateKNNDistanceWeighted_spatial_autocor(featureMatrix, distanceType, k)
            nodes = range(0,featureMatrix.shape[0])
            Gtmp = nx.Graph()
            Gtmp.add_nodes_from(nodes)
            Gtmp.add_weighted_edges_from(edgeList)
            adj = nx.adjacency_matrix(Gtmp)
            adj_knn_by_coordinate = np.array(adj.todense())
            return adj_knn_by_coordinate
        
        
        features_np = features.detach().numpy()
        start_time = time.time()
        print("MP_MIM_SpaGCN. Start Time: %s seconds" %
          (start_time))
        #MP_parameter_set
        MP_k_num_list = [1,2,3,4,5,6,7,8,9]
        MP_l_num_list = [1,2,3,4,5,6,7,8,9,10,11,12,13,14,15]
        #MP_k_num_list = [1]
        #MP_l_num_list = [1]
        embedding_MIrow_max_list = []
        
        meta_folder_path = os.path.abspath('./meta_data_folder/metaData_brain_16_coords')
        embedding_in_SpaGCN_folder = "./SpaGCN_MP_embedding_"+sample+"/"
        if not os.path.exists(embedding_in_SpaGCN_folder):
            os.makedirs(embedding_in_SpaGCN_folder)

        emblist = []
        for i in range(features.shape[1]):
            emblist.append('embedding'+str(i)) 
        features_df = pd.DataFrame(features.detach().numpy(), columns=emblist)
        features_df.to_csv(embedding_in_SpaGCN_folder+sample+'_spagcn_default_raw_gat_ave_native_embedding.csv')
        
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
        graph_embedding_np = graph_embedding_df[emblist].values    #transformed embedding for downstream analysis
        if os.path.exists(embedding_in_SpaGCN_folder):
            shutil.rmtree(embedding_in_SpaGCN_folder)
        end_time = time.time()
        print("MP_MIM_SpaGCN. End Time: %s seconds" %
                (end_time))
        print("MP_MIM_SpaGCN Done. Total Running Time: %s seconds" %
                (end_time - start_time))
                
        #-------------------------------------------------------------------------------add MP
        if init=="kmeans":
            print("Initializing cluster centers with kmeans, n_clusters known")
            self.n_clusters=n_clusters
            kmeans = KMeans(self.n_clusters, n_init=20)
            if init_spa:
                #------Kmeans use exp and spatial
                #y_pred = kmeans.fit_predict(features.detach().numpy())
                y_pred = kmeans.fit_predict(graph_embedding_np)                             #add MP
            else:
                #------Kmeans only use exp info, no spatial
                #y_pred = kmeans.fit_predict(X)  #Here we use X as numpy
                y_pred = kmeans.fit_predict(graph_embedding_np)                             #add MP
        elif init=="louvain":
            print("Initializing cluster centers with louvain, resolution = ", res)
            if init_spa:
                #adata=sc.AnnData(features.detach().numpy())
                adata=sc.AnnData(graph_embedding_np)                                        #add MP
            else:
                adata=sc.AnnData(X)
            sc.pp.neighbors(adata, n_neighbors=n_neighbors)
            sc.tl.louvain(adata, resolution=res, random_state=self.loiuvain_seed)
            y_pred=adata.obs['louvain'].astype(int).to_numpy()
            self.n_clusters=len(np.unique(y_pred))
        #----------------------------------------------------------------
        y_pred_last = y_pred
        self.mu = Parameter(torch.Tensor(self.n_clusters, self.nhid))
        X=torch.FloatTensor(X)
        adj=torch.FloatTensor(adj)
        self.trajectory.append(y_pred)
        #features=pd.DataFrame(features.detach().numpy(),index=np.arange(0,features.shape[0]))
        features=pd.DataFrame(features_graph_np,index=np.arange(0,features.shape[0]))                                         #add MP
        Group=pd.Series(y_pred,index=np.arange(0,features.shape[0]),name="Group")
        Mergefeature=pd.concat([features,Group],axis=1)
        cluster_centers=np.asarray(Mergefeature.groupby("Group").mean())
        
        self.mu.data.copy_(torch.Tensor(cluster_centers))
        self.train()
        epoch=0
        for epoch in range(max_epochs):
            if epoch%update_interval == 0:
                _, q = self.forward(X,adj)
                p = self.target_distribution(q).data
            if epoch%100==0:
                print("Epoch ", epoch) 
            optimizer.zero_grad()
            z,q = self(X, adj)
            loss = self.loss_function(p, q)
            loss.backward()
            optimizer.step()
            if epoch%trajectory_interval == 0:
                self.trajectory.append(torch.argmax(q, dim=1).data.cpu().numpy())

            #Check stop criterion
            y_pred = torch.argmax(q, dim=1).data.cpu().numpy()
            delta_label = np.sum(y_pred != y_pred_last).astype(np.float32) / X.shape[0]
            y_pred_last = y_pred
            if epoch>0 and (epoch-1)%update_interval == 0 and delta_label < tol:
                print('delta_label ', delta_label, '< tol ', tol)
                print("Reach tolerance threshold. Stopping training.")
                break


    def fit_with_init(self, X,adj, init_y, lr=0.001, max_epochs=5000, update_interval=1, weight_decay=5e-4,opt="sgd"):
        print("Initializing cluster centers with kmeans.")
        if opt=="sgd":
            optimizer = optim.SGD(self.parameters(), lr=lr, momentum=0.9)
        elif opt=="admin":
            optimizer = optim.Adam(self.parameters(),lr=lr, weight_decay=weight_decay)
        X=torch.FloatTensor(X)
        adj=torch.FloatTensor(adj)
        features, _ = self.forward(X,adj)
        features=pd.DataFrame(features.detach().numpy(),index=np.arange(0,features.shape[0]))
        Group=pd.Series(init_y,index=np.arange(0,features.shape[0]),name="Group")
        Mergefeature=pd.concat([features,Group],axis=1)
        cluster_centers=np.asarray(Mergefeature.groupby("Group").mean())
        self.mu.data.copy_(torch.Tensor(cluster_centers))
        self.train()
        for epoch in range(max_epochs):
            if epoch%update_interval == 0:
                _, q = self.forward(torch.FloatTensor(X),torch.FloatTensor(adj))
                p = self.target_distribution(q).data
            X=torch.FloatTensor(X)
            adj=torch.FloatTensor(adj)
            optimizer.zero_grad()
            z,q = self(X, adj)
            loss = self.loss_function(p, q)
            loss.backward()
            optimizer.step()

    def predict(self, X, adj):
        z,q = self(torch.FloatTensor(X),torch.FloatTensor(adj))
        return z, q




class GC_DEC(nn.Module):
    def __init__(self, nfeat, nhid1,nhid2, n_clusters=None, dropout=0.5,alpha=0.2):
        super(GC_DEC, self).__init__()

        self.gc1 = GraphConvolution(nfeat, nhid1)
        self.gc2 = GraphConvolution(nhid1, nhid2)
        self.dropout = dropout
        self.mu = Parameter(torch.Tensor(n_clusters, nhid2))
        self.n_clusters=n_clusters
        self.alpha=alpha

    def forward(self, x, adj):
        x=self.gc1(x, adj)
        x = F.relu(x)
        x = F.dropout(x, self.dropout, training=True)
        x = self.gc2(x, adj)
        q = 1.0 / ((1.0 + torch.sum((x.unsqueeze(1) - self.mu)**2, dim=2) / self.alpha) + 1e-6)
        q = q**(self.alpha+1.0)/2.0
        q = q / torch.sum(q, dim=1, keepdim=True)
        return x, q

    def loss_function(self, p, q):
        def kld(target, pred):
            return torch.mean(torch.sum(target*torch.log(target/(pred+1e-6)), dim=1))
        loss = kld(p, q)
        return loss

    def target_distribution(self, q):
        #weight = q ** 2 / q.sum(0)
        #return torch.transpose((torch.transpose(weight,0,1) / weight.sum(1)),0,1)e
        p = q**2 / torch.sum(q, dim=0)
        p = p / torch.sum(p, dim=1, keepdim=True)
        return p

    def fit(self, X,adj, lr=0.001, max_epochs=10, update_interval=5, weight_decay=5e-4,opt="sgd",init="louvain",n_neighbors=10,res=0.4):
        self.trajectory=[]
        print("Initializing cluster centers with kmeans.")
        if opt=="sgd":
            optimizer = optim.SGD(self.parameters(), lr=lr, momentum=0.9)
        elif opt=="admin":
            optimizer = optim.Adam(self.parameters(),lr=lr, weight_decay=weight_decay)

        features, _ = self.forward(torch.FloatTensor(X),torch.FloatTensor(adj))
        #----------------------------------------------------------------
        
        if init=="kmeans":
            #Kmeans only use exp info, no spatial
            #kmeans = KMeans(self.n_clusters, n_init=20)
            #y_pred = kmeans.fit_predict(X)  #Here we use X as numpy
            #Kmeans use exp and spatial
            kmeans = KMeans(self.n_clusters, n_init=20)
            y_pred = kmeans.fit_predict(features.detach().numpy())
        elif init=="louvain":
            adata=sc.AnnData(features.detach().numpy())
            sc.pp.neighbors(adata, n_neighbors=n_neighbors)
            sc.tl.louvain(adata,resolution=res)
            y_pred=adata.obs['louvain'].astype(int).to_numpy()
        #----------------------------------------------------------------
        X=torch.FloatTensor(X)
        adj=torch.FloatTensor(adj)
        self.trajectory.append(y_pred)
        features=pd.DataFrame(features.detach().numpy(),index=np.arange(0,features.shape[0]))
        Group=pd.Series(y_pred,index=np.arange(0,features.shape[0]),name="Group")
        Mergefeature=pd.concat([features,Group],axis=1)
        cluster_centers=np.asarray(Mergefeature.groupby("Group").mean())
        
        self.mu.data.copy_(torch.Tensor(cluster_centers))
        self.train()
        for epoch in range(max_epochs):
            if epoch%update_interval == 0:
                _, q = self.forward(X,adj)
                p = self.target_distribution(q).data
            if epoch%100==0:
                print("Epoch ", epoch) 
            optimizer.zero_grad()
            z,q = self(X, adj)
            loss = self.loss_function(p, q)
            loss.backward()
            optimizer.step()
            self.trajectory.append(torch.argmax(q, dim=1).data.cpu().numpy())

    def fit_with_init(self, X,adj, init_y, lr=0.001, max_epochs=10, update_interval=1, weight_decay=5e-4,opt="sgd"):
        print("Initializing cluster centers with kmeans.")
        if opt=="sgd":
            optimizer = optim.SGD(self.parameters(), lr=lr, momentum=0.9)
        elif opt=="admin":
            optimizer = optim.Adam(self.parameters(),lr=lr, weight_decay=weight_decay)
        X=torch.FloatTensor(X)
        adj=torch.FloatTensor(adj)
        features, _ = self.forward(X,adj)
        features=pd.DataFrame(features.detach().numpy(),index=np.arange(0,features.shape[0]))
        Group=pd.Series(init_y,index=np.arange(0,features.shape[0]),name="Group")
        Mergefeature=pd.concat([features,Group],axis=1)
        cluster_centers=np.asarray(Mergefeature.groupby("Group").mean())
        self.mu.data.copy_(torch.Tensor(cluster_centers))
        self.train()
        for epoch in range(max_epochs):
            if epoch%update_interval == 0:
                _, q = self.forward(torch.FloatTensor(X),torch.FloatTensor(adj))
                p = self.target_distribution(q).data
            X=torch.FloatTensor(X)
            adj=torch.FloatTensor(adj)
            optimizer.zero_grad()
            z,q = self(X, adj)
            loss = self.loss_function(p, q)
            loss.backward()
            optimizer.step()

    def predict(self, X, adj):
        z,q = self(torch.FloatTensor(X),torch.FloatTensor(adj))
        return z, q


