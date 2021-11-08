## Part Grouping Network (PGN)
Ke Gong, Xiaodan Liang, Yicheng Li, Yimin Chen, Ming Yang and Liang Lin, "Instance-level Human Parsing via Part Grouping Network", ECCV 2018 (Oral).

### Introduction

PGN is a state-of-art deep learning methord for semantic part segmentation, instance-aware edge detection and instance-level human parsing built on top of [Tensorflow](http://www.tensorflow.org).

This distribution provides a publicly available implementation for the key model ingredients reported in our latest [paper](http://openaccess.thecvf.com/content_ECCV_2018/papers/Ke_Gong_Instance-level_Human_Parsing_ECCV_2018_paper.pdf) which is accepted by ECCV 2018.


### Crowd Instance-level Human Parsing (CIHP) Dataset

The PGN is trained and evaluated on our [CIHP dataset](https://lip.sysuhcp.com/) for isntance-level human parsing.  Please check it for more model details. The dataset is also available at [google drive](https://drive.google.com/drive/folders/0BzvH3bSnp3E9QjVYZlhWSjltSWM?resourcekey=0-nkS8bDVjPs3bEw3UZW-omA&usp=sharing) and [baidu drive](http://pan.baidu.com/s/1nvqmZBN).

### Pre-trained models

We have released our trained models of PGN on CIHP dataset at [google drive](https://drive.google.com/open?id=1Mqpse5Gen4V4403wFEpv3w3JAsWw2uhk).

### Inference
1. Download the pre-trained model and store in $HOME/checkpoint.
2. Prepare the images and store in $HOME/datasets.
3. Run test_pgn.py.
4. The results are saved in $HOME/output
5. Evaluation scripts are in $HOME/evaluation. Copy the groundtruth files (in _Instance_ids_ folder) into $HOME/evaluation/Instance_part_val before you run the script.

### Training
1. Download the pre-trained model and store in $HOME/checkpoint.
2. Download CIHP dataset or prepare your own data and store in $HOME/datasets.
3. For CIHP dataset, you need to generate the edge labels and left-right flipping labels (optional). We have provided a script for reference.
4. Run train_pgn.py to train PGN.
5. Use test_pgn.py to generate the results with the trained models.
6. The instance tool is used for instance partition process from semantic part segmentation maps and instance-aware edge maps, which is written in MATLAB.

## Related work
+ Self-supervised Structure-sensitive Learning [SSL](https://github.com/Engineering-Course/LIP_SSL), CVPR2017
+ Joint Body Parsing & Pose Estimation Network  [JPPNet](https://github.com/Engineering-Course/LIP_JPPNet), T-PAMI2018
+ Graphonomy: Universal Human Parsing via Graph Transfer Learning [Graphonomy](https://github.com/Gaoyiminggithub/Graphonomy), CVPR2019

