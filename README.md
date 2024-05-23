# MiDSS

### 1. Introduction

This repository contains the implementation of the paper **Constructing and Exploring Intermediate Domains in Mixed Domain Semi-supervised Medical Image Segmentation**

### 2. Dataset Construction

The dataset needs to be divided into two folders for training and testing. The training and testing data should be in the format of the "data" folder.

### 3. Train

`code/train.py` is the implementation of our method on the Prostate and Fundus dataset.

code/train_MNMS is the implementation of our method on the M&Ms dataset.

Modify the paths in lines 631 to 636 of the code.

```python
if args.dataset == 'fundus':
    train_data_path='../../data/Fundus' # the folder of fundus dataset
elif args.dataset == 'prostate':
    train_data_path="../../data/ProstateSlice" # the folder of prostate dataset
elif args.dataset == 'MNMS':
    train_data_path="../../data/MNMS/mnms" # the folder of mnms dataset
```

