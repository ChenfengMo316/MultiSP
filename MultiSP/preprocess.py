import torch
import numpy as np
import scanpy as sc
import episcanpy as epi
from scipy.sparse import issparse
from sklearn.decomposition import PCA
from sklearn.neighbors import kneighbors_graph 
from .utils import construct_graph_by_coordinate,preprocess_graph,clr_normalize_each_cell,lsi,tfidf
    
def data_preprocess(adata_omics1, adata_omics2,modality_type='RNA and ATAC', k=3): 
    adata_omics1_high_raw,feature1_raw,feature1,scale_factor1=RNA_preprocess(adata_omics1)
    omic1_data_dict={'adata_omic_raw':adata_omics1_high_raw,
                     'feature_raw':feature1_raw,
                     'feature':feature1,
                     'scale_factor':scale_factor1
                    }
    if modality_type=='RNA and Protein':
       adata_omics2_raw,feature2_raw,feature2,scale_factor2=Protein_preprocess(adata_omics2)
       omic2_data_dict={'adata_omic_raw':adata_omics2_raw,
                        'feature_raw':feature2_raw,
                        'feature':feature2,
                        'scale_factor':scale_factor2
                        }
    
    if modality_type=='RNA and ATAC':
       adata_omics2_raw,feature2_raw,feature2,scale_factor2=ATAC_preprocess(adata_omics2)
       omic2_data_dict={'adata_omic_raw':adata_omics2_raw,
                        'feature_raw':feature2_raw,
                        'feature':feature2,
                        'scale_factor':scale_factor2
                       }
    if modality_type=='RNA and ATAC_P_mouse_brain':
       adata_omics2_raw,feature2=P_mouse_brain_ATAC_preprocess(adata_omics2)
       omic2_data_dict={'adata_omic_raw':adata_omics2_raw,
                        'feature':feature2,
                       }

    adj1,adj2,adj_feature_omic1,adj_feature_omic2=graph_construction(adata_omics1_high_raw,adata_omics2_raw,k)
    omic1_data_dict['adj']=adj1
    omic2_data_dict['adj']=adj2
    omic1_data_dict['adj_feat']=adj_feature_omic1
    omic2_data_dict['adj_feat']=adj_feature_omic2
    return omic1_data_dict,omic2_data_dict

    
def RNA_preprocess(adata):
       sc.pp.filter_genes(adata, min_cells=10)
       sc.pp.highly_variable_genes(adata, flavor="seurat_v3", n_top_genes=3000)
       adata_raw=adata.copy()
       adata_high_raw=adata_raw[:, adata_raw.var['highly_variable']]
       sc.pp.normalize_per_cell(adata)
       adata.obs['size_factors'] = adata.obs.n_counts/ np.median(adata.obs.n_counts)
       scale_factor=torch.tensor(adata.obs['size_factors'], dtype=torch.float32).unsqueeze(1)
    
       sc.pp.log1p(adata)
       sc.pp.scale(adata)
       adata_high = adata[:, adata.var['highly_variable']]
       feature_raw=adata_high_raw.X.toarray()
       feature=adata_high.X.toarray()
       adata_high_raw.obsm['feat'] =PCA(n_components=100, random_state=42).fit_transform(adata_high.X.toarray())
       feature=adata_high_raw.obsm['feat']
       return adata_high_raw,feature_raw,feature,scale_factor
    
def Protein_preprocess(adata):
        adata_raw=adata.copy()
        raw_n_counts=adata.X.sum(axis=1)
        adata.obs['size_factors'] =raw_n_counts
        scale_factor=torch.tensor(adata.obs['size_factors'], dtype=torch.float32).unsqueeze(1)
        adata= clr_normalize_each_cell(adata)
        sc.pp.scale(adata)
        if issparse(adata_raw.X):
           feature_raw=adata_raw.X.toarray()
        else:
           feature_raw=adata_raw.X
        if issparse(adata.X):
           feature=adata.X.toarray()
        else:
           feature=adata.X
        adata_raw.obsm['feat']=feature
        return adata_raw,feature_raw,feature,scale_factor

def ATAC_preprocess(adata):
        epi.pp.filter_features(adata, min_cells=int(adata.shape[0] * 0.03))
        epi.pp.filter_features(adata, min_cells=1)
        adata_raw=adata.copy()
        adata.obs['size_factors'] = np.sum(adata.X,axis=1)
        scale_factor=torch.tensor(adata.obs['size_factors'], dtype=torch.float32).unsqueeze(1)
        lsi(adata, use_highly_variable=False, n_components=100+1)
        adata_raw.obsm['feat']=adata.obsm['X_lsi'].copy()
        adata.X=tfidf(adata.X)
        if issparse(adata_raw.X):
           feature_raw=adata_raw.X.toarray()
        else:
           feature_raw=adata_raw.X
        if issparse(adata.X):
           feature= adata.X.toarray()
        else:
           feature= adata.X
        feature=adata_raw.obsm['feat']
        
        return adata_raw,feature_raw,feature,scale_factor

def P_mouse_brain_ATAC_preprocess(adata):
        adata_raw=adata.copy()
        adata_raw.obsm['feat']=adata.obsm['X_lsi'].copy()
        feature=adata_raw.obsm['feat']
        
        return adata_raw,feature

def graph_construction(adata1,adata2,k):
        adj=construct_graph_by_coordinate(adata1.obsm['spatial'],k)


        adj = adj + adj.T
        adj = np.where(adj>1, 1, adj)
        adj=preprocess_graph(adj)

        adj1=adj
        adj2=adj
        
        adj_feature_omic1=kneighbors_graph(adata1.obsm['feat'], 20, mode="connectivity", metric="cosine", include_self=False)
        adj_feature_omic2=kneighbors_graph(adata2.obsm['feat'], 20, mode="connectivity", metric="cosine", include_self=False)

        adj_feature_omic1=adj_feature_omic1.toarray()
        adj_feature_omic2=adj_feature_omic2.toarray()

        adj_feature_omic1 = adj_feature_omic1 + adj_feature_omic1.T
        adj_feature_omic1 = np.where(adj_feature_omic1>1, 1, adj_feature_omic1)
        adj_feature_omic1=preprocess_graph(adj_feature_omic1)

        adj_feature_omic2 = adj_feature_omic2 + adj_feature_omic2.T
        adj_feature_omic2 = np.where(adj_feature_omic2>1, 1, adj_feature_omic2)
        adj_feature_omic2=preprocess_graph(adj_feature_omic2)
    

        return adj1,adj2,adj_feature_omic1,adj_feature_omic2


   
 