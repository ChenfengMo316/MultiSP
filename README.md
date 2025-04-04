# MultiSP deciphers tissue structure and multicellular communication from spatial multi-omics data 

## System Requirements

### Hardware requirements

The  MultiSP package can be run on GPU (recommend) or CPU.

### Software requirements

#### OS Requirements

This package has been tested on Windows, Linux and macOS (Ventura) operating systems, and should work in any valid python environment. 

#### Dependencies

* python==3.11
* torch==2.4.0
* numpy==1.26.4
* pandas==2.22.2
* scanpy==1.10.2
* episcanpy==0.4.0
* anndata==0.10.8
* rpy2==3.5.11
* scipy==1.14.0
* scikit-learn==1.5.1
* tqdm==4.66.5
* matplotlib==3.9.2
* R==4.3.1

## Installation

It's prefered to create a new environment for MultiSP

```
conda create -n MultiSP python==3.11
conda activate MultiSP
```

MultiSP is available on PyPI and can be installed via

```
pip install MultiSP
```

Install all the required packages

```
pip install -r requirements.txt
```

Installation of MultiSP should take less than a minute and it may take several minutes to install the dependencies.

The use of the mclust algorithm requires the rpy2 package (Python) and the mclust package (R). See https://pypi.org/project/rpy2/ and https://cran.r-project.org/web/packages/mclust/index.html for detail.

## Tutorials

The Visium CytAssist human lymph node data were downloaded from the Zenodo database https://zenodo.org/records/10362607. The MISAR-seq mouse brain data were available from the National Genomics Data Center via the accession number([OEP003285, www. biosino.org/node/project/detail/OEP003285]([National Omics Data Encyclopedia](https://www.biosino.org/node/project/detail/OEP003285))).

Two step-by-step tutorials are included in the Tutorial folder to show how to use MultiSP.

- 1.Tutorial for spatial RNA-ADT data (It takes about 20 seconds to run on GeForce RTX 3090 GPU )
- 2.Tutorial for spatial RNA-ATAC data (It takes about 2 minutes to run on GeForce RTX 3090 GPU)

Tutorial for inferring spatially multimodal cell-cell communication is available at the Github repository of [CellChat toolkit](https://github.com/jinworks/CellChat)

