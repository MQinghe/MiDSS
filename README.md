# MiDSS

### 1. Introduction

This repository contains the implementation of the paper **[Constructing and Exploring Intermediate Domains in Mixed Domain Semi-supervised Medical Image Segmentation](https://openaccess.thecvf.com/content/CVPR2024/html/Ma_Constructing_and_Exploring_Intermediate_Domains_in_Mixed_Domain_Semi-supervised_Medical_CVPR_2024_paper.html)**
> *In Conference on Computer Vision and Pattern Recognition (CVPR), 2024*

### 2. Dataset Construction

The dataset needs to be divided into two folders for training and testing. The training and testing data should be in the format of the "data" folder.

### 3. Train

`code/train.py` is the implementation of our method on the Prostate and Fundus dataset.

`code/train_MNMS.py` is the implementation of our method on the M&Ms dataset.

Modify the paths in lines 631 to 636 of the code.

```python
if args.dataset == 'fundus':
    train_data_path='../../data/Fundus' # the folder of fundus dataset
elif args.dataset == 'prostate':
    train_data_path="../../data/ProstateSlice" # the folder of prostate dataset
elif args.dataset == 'MNMS':
    train_data_path="../../data/MNMS/mnms" # the folder of mnms dataset
```

then simply run:

```python
python train.py --dataset ... --lb_domain ... --lb_num ... --save_name ... --gpu 0
```

### 4. Test

To run the evaluation code, please update the path of the dataset in `test.py`:

Modify the paths in lines 249 to 254 of the code.

then simply run:

```
python test.py --dataset ... --save_name ... --gpu 0
```

### 5. Acknowledgement

This project is based on the code from the [SSL4MIS](https://github.com/HiLab-git/SSL4MIS) project.

Thanks a lot for their great works.
