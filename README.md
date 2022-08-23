# RandLA-Net-pytorch
This repository contains the implementation of [RandLA-Net (CVPR 2020 Oral)](https://arxiv.org/abs/1911.11236) in PyTorch.
### updates:
* We extend the model with to train with one synthetic dataset, [SynLiDAR](https://github.com/xiaoaoran/SynLiDAR)
* We replaced the weighted cross entropy in the oroginal implementation with focal loss to alliverate the influence of class imbalance
* fix some bugs in the original implementation
* We improve the mIoU on the validation set from 53.1% on **validation set** to **55.1%**.
* This is a good starting point & backbone choice for those who plan to start their research on point clouds segmentation.



- support SemanticKITTI dataset now. (Welcome everyone to develop together and raise PR)
- We place our pretrain-model in [`pretrain_model/checkpoint.tar`](pretrain_model/checkpoint.tar) directory.

### Performance

> Results on Validation Set (seq 08)

- Compare with original implementation

| Model                      | mIoU  |
| -------------------------- | ----- |
| Original Tensorflow        | 0.531 |
| Our Pytorch Implementation | 0.551 |

- Per class mIoU

| mIoU | car  | bicycle | motorcycle | truck | other-vehicle | person | bicyclist | motorcyclist | road | parking | sidewalk | other-ground | building | fence | vegetation | trunk | terrain | pole | traffic-sign |
| ---- | ------- | ---------- | ----- | ------------- | ------ | --------- | ------------ | ---- | ------- | -------- | ------------ | -------- | ----- | ---------- | ----- | ------- | ---- | ------------ | ---- |
| 55.1 | 0.939|0.092|0.347|0.659|0.453|0.548|0.707|0.000|0.920|0.401|0.784|0.006|0.886|0.520|0.855|0.627|0.747|0.568|0.403|

## A. Environment Setup

0. Click [this webpage](https://pytorch.org/get-started/locally/) and use conda to install pytorch>=1.4 (Be aware of the cuda version when installation)

1. Install python packages

```
pip install -r requirements.txt
```

2. Compile C++ Wrappers

```
sh compile_op.sh
```

## B. Prepare Data

Download the [Semantic KITTI dataset](http://semantic-kitti.org/dataset.html#download), and preprocess the data:

```
python data_prepare_semantickitti.py
```
Note: 
- Please change the dataset path in the `data_prepare_semantickitti.py` with your own path.
- Data preprocessing code will **convert the label to 0-19 index**

## C. Training & Testing

1. Training

```bash
python3 train_SemanticKITTI.py <args>
```

2. Testing

```bash
python3 test_SemanticKITTI.py <args>
```
**Note: if the flag `--index_to_label` is set, output predictions will be ".label" files (label figure) which can be visualized; Otherwise, they will be ".npy" (0-19 index) files which is used to evaluated afterward.**

## D. Visualization & Evaluation

1. Visualization

```bash
python3 visualize_SemanticKITTI.py <args>
```

2. Evaluation

- Example Evaluation code

```bash
python3 evaluate_SemanticKITTI.py --dataset /tmp2/tsunghan/PCL_Seg_data/sequences_0.06/ \
    --predictions runs/supervised/predictions/ --sequences 8
```

## Acknowledgement
- This repo is improved from [RandLA-Net PyTorch](https://github.com/tsunghan-wu/RandLA-Net-pytorch)
- Original Tensorflow implementation [link](https://github.com/QingyongHu/RandLA-Net)

