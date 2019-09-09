# Training and Inference for Integer-based Semantic Segmentation Network
This repository is still under cosntruction, use the code at your own risk!
### Overview

### Table of contents
<!-- 1. [About EfficientNet](#about-efficientnet)
2. [About EfficientNet-PyTorch](#about-efficientnet-pytorch)
3. [Installation](#installation)
4. [Usage](#usage)
    * [Load pretrained models](#loading-pretrained-models)
    * [Example: Classify](#example-classification)
    * [Example: Extract features](#example-feature-extraction)
    * [Example: Export to ONNX](#example-export)
6. [Contributing](#contributing)  -->
1. [Installation](#installation)
2. [Download Data](#download-data)
3. [Usage](#usage)
    * [Training](#training)
    * [Evaluation](#evaluation)
    * [Visualization](#visualization)
3. [Contributing](#contributing)

### Installation
Make sure you have **python3** installed in your environment.  
Type the following commands:
```bash
git clone https://github.com/MarkYangjiayi/NOBN
cd NOBN
pip install -r requirements.txt
```

### Download Data
* Pretrained Weights  
For Deeplab, download pretrained checkpoint [here](https://drive.google.com/open?id=1fAyfLim-fYPCy-DuMeyX2OYODQ1RzKTM). For FCN, download pretrained weights [here](https://drive.google.com/open?id=1r1OviIUH5iwylKEDS9c1jn_4Qx9OZKdN). After download, put them into the pretrain folder of the corresponding network you want to train.
* Datasets  
We use TFrecord to feed data into our network, the code references DeepLabv3+ from google, which you can find [here](https://github.com/tensorflow/models/tree/master/research/deeplab).<br/>
To train the model with PASCAL VOC 2012 dataset, first download [PASCAL VOC 2012](http://host.robots.ox.ac.uk/pascal/VOC/voc2012/VOCtrainval_11-May-2012.tar) and [SegmentationClassAug](https://www.dropbox.com/s/oeu149j8qtbs1x0/SegmentationClassAug.zip?dl=0). Do not unzip and move them into the "./data" folder. Then,
```bash
cd data
sh convert_voc2012_aug.sh
```
If successful, you should have an "./data/pascal_voc_seg/tfrecord" folder with the dataset ready in TFRecords format.
Reference this [blog post](https://www.sun11.me/blog/2018/how-to-use-10582-trainaug-images-on-DeeplabV3-code/) for more information.

### Usage
* Training
```python
mode = "train"
```
* Evaluation
```python
mode = "val"
```
* Visualization  
```python
mode = "vis"
```
Note that the output prediction and ground truth is labeled with single channel. To gain better visualization, you have to convert them into RGB images.

### Contributing
If you find a bug, create a GitHub issue, or even better, submit a pull request. Similarly, if you have questions, simply post them as GitHub issues.   
