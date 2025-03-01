# MMCAF-Net: Multimodal Multiscale Cross Attention Fusion Network for Multimodal Disease Classification

![](https://img.shields.io/badge/-Github-181717?style=flat-square&logo=Github&logoColor=FFFFFF)
![](https://img.shields.io/badge/-Awesome-FC60A8?style=flat-square&logo=Awesome&logoColor=FFFFFF)
![](https://img.shields.io/badge/-Python-3776AB?style=flat-square&logo=Python&logoColor=FFFFFF)
![](https://img.shields.io/badge/-Pytorch-EE4C2C?style=flat-square&logo=Pytorch&logoColor=FFFFFF)


## Proposed method
We propose a new framework called MMCAF-Net. It consists of three main components. The first component is an image encoder that integrates a feature pyramid with E3D-
MSCA module, designed to capture both local and global features of imaging data. This allows for the effective differentiation of challenging cases. The second component employs Kolmogorov–Arnold Networks (KAN) to encode tabular features. Lastly, the third component utilizes the MSCA module to align and fuse the features from both modalities.

The figure below shows our proposed network.

![image](images/main.png)

The figure below shows our proposed E3D-MSCA.

![image](images/E3D-MSCA.png)

The figure below shows our proposed MSCA.

![image](images/MSCA.png)


## Getting started to evaluate
### Install dependencies
```
pip install -r requirements.txt
```

### Data preprocess
Lung-PET-CT-dx dataset can get from this link https://doi.org/10.7937/TCIA.2020.NNC2-0461
In short, using create_hdf5.py to make an hdf5 file.

### Evaluation
To do the evaluation process, please run the following command :
```
sh test.sh
```

### Train by yourself
If you want to train by yourself, you can run this command :
```
sh train.sh
```
