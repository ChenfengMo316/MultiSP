# Accurate spatial domain detection by integrating spatial multi-omics data using MultiSP 

This repository contains MultiSP script and two tutorial jupyter notebooks for reproducing the outcomes shown in the paper. 



## Overview
Recent breakthroughs in spatial multi-omics technologies have enabled the profiling of multi-modal data-such as gene expression, chromatin accessibility and protein abundance - while preserving the spatial architecture of tissue sections, which provides unprecedented opportunities to study cellular diversity. However, deciphering complex tissue structures and functions remains challenges due to the highly sparse and noisy nature of the data. To address this, we present MultiSP, a deep learning framework that leverages cellular neighborhood structures and modality-specific probabilistic modeling to enhance the data representation.MultiSP achieves an accurate cross-modality integration and jointly latent representation by further introducing an adversarial learning component. Applications to spatial multi-omics datasets across different technologies and tissue types, MultiSP exhibits superior performance against existing methods in capturing biologically interpretable spatial domains. We also demonstrate MultiSP’s ability in dissecting epigenomics-induced spatial variations and complex tissue structures in the tumor microenvironment,where MultiSP uncovers two tumor-associated macrophage subsets with distinct prognosis power and the immune evasion mechanisms for tumors. Together, MultiSP serves as a powerful framework for uncovering spatially multimodal heterogeneity and regulations by integrating complementary information from multiple modalities.

## Requirements
You'll need to install the following packages in order to run the codes.
* python==3.8
* torch=2.4.0
* numpy==1.26.4
* pandas=2.22.2
* scanpy==1.10.2
* episcanpy=0.4.0
* anndata=0.10.8
* rpy2==3.5.11
* scipy==1.14.0
* scikit-learn==1.5.1
* tqdm==4.66.5
* matplotlib==3.9.2
* R==4.3.1

## Data
The SPOTS mouse spleen and breast cancer data were obtained from the Gene Expression Omnibus (GEO) repository (accession no. [GSE198353]()). The Visium CytAssist human lymph node data were downloaded from the Zenodo database https://zenodo.org/records/10362607. The MISAR-seq mouse brain data were available from the National Genomics Data Center via the accession number([OEP003285, www. biosino.org/node/project/detail/OEP003285]()).

