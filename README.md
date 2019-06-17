# SiamRPN++_PyTorch 


<div align=center><img src="https://github.com/PengBoXiangShang/SiamRPN_plus_plus_Pytorch/blob/master/illustration/SiamRPN_plus_plus_pipeline.png"/></div>

This is an unofficial PyTorch implementation of [SiamRPN++ (CVPR2019)](https://arxiv.org/pdf/1812.11703.pdf), implemented by **[Peng Xu](http://www.pengxu.net)** and **[Jin Feng](https://github.com/JinDouer)**. Our **training** can be conducted on **multi-GPUs**, and use **LMDB** data format to speed up the data loading.

This project is designed with these goals:
- [x] Training on ILSVRC2015_VID dataset.
- [ ] Training on GOT-10k dataset.
- [ ] Training on YouTube-BoundingBoxes dataset.
- [ ] Evaluate the performance on tracking benchmarks.

## Details of SiamRPN++ Network
As stated in the original paper, SiamRPN++ network has three parts, including Backbone Networks, SiamRPN Blocks, and Weighted Fusion Layers.

**1. Backbone Network (modified ResNet-50)**

As stated in the original paper, SiamRPN++ uses ResNet-50 as backbone by modifying the strides and adding dilated convolutions for *conv4* and *conv5* blocks. Here, we present the detailed comparison between original ResNet-50 and SiamRPN++ ResNet-50 backbone in following table.

<table>
   <tr>
      <td colspan = 2 rowspan=2></td>
      <td colspan = 3 style="text-align: center;">bottleneck in conv4</td>
      <td colspan = 3 style="text-align: center;">bottleneck in conv5</td>
   </tr>
   <tr>
      <td>conv1x1</td>
      <td>conv3x3</td>
      <td>conv1x1</td>
      <td>conv1x1</td>
      <td>conv3x3</td>
      <td>conv1x1</td>
   </tr>
   <tr>
      <td rowspan = 3>original ResNet-50</td>
      <td>stride</td>
      <td>1</td>
      <td>2</td>
      <td>1</td>
      <td>1</td>
      <td>2</td>
      <td>1</td>
   </tr>
   <tr>
      <td>padding</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
   </tr>
   <tr>
      <td>dilation</td>
      <td>1</td>
      <td>1</td>
      <td>1</td>
      <td>1</td>
      <td>1</td>
      <td>1</td>
   </tr>
   <tr>
      <td rowspan=3>ResNet-50 in SiamRPN++</td>
      <td>stride</td>
      <td>1</td>
      <td>1</td>
      <td>1</td>
      <td>1</td>
      <td>1</td>
      <td>1</td>
   </tr>
   <tr>
      <td>padding</td>
      <td>0</td>
      <td>2</td>
      <td>0</td>
      <td>0</td>
      <td>4</td>
      <td>0</td>
   </tr>
   <tr>
      <td>dilation</td>
      <td>1</td>
      <td>2</td>
      <td>1</td>
      <td>1</td>
      <td>4</td>
      <td>1</td>
   </tr>
</table>

**2. SiamRPN Block**

Based on our understanding to the original paper, we plot a architecture illustration to describe the *Siamese RPN* block as shown in following.

<div align=center><img src="https://github.com/PengBoXiangShang/SiamRPN_plus_plus_Pytorch/blob/master/illustration/RPN.png"/></div>

We also present the detailed configurations of each layer of RPN block in following table. Please see more details in [./network/RPN.py](https://github.com/PengBoXiangShang/SiamRPN_plus_plus_Pytorch/blob/master/network/RPN.py).

|component|configuration|
|:---|:---|
|adj_1 / adj_2 / adj_3 / adj_4|conv2d(256, 256, ksize=3, pad=1, stride=1), BN2d(256)|
|fusion_module_1 / fusion_module_2|conv2d(256, 256, ksize=1, pad=0, stride=1), BN2d(256), ReLU|
|box head|conv2d(256, 4*5, ksize=1, pad=0, stride=1)|
|cls head|conv2d(256, 2*5, ksize=1, pad=0, stride=1)|

**3. Weighted Fusion Layer** 

We implemente the *weighted fusion layer* via **group convolution operations**. Please see details in [./network/SiamRPN.py](https://github.com/PengBoXiangShang/SiamRPN_plus_plus_Pytorch/blob/master/network/SiamRPN.py).

## Requirements
Ubuntu 14.04

Python 2.7

PyTorch 0.4.0

Other main requirements can be installed by:

```
# 1. Install cv2 package.
conda install opencv

# 2. Install LMDB package.
conda install lmdb

# 3. Install fire package.
pip install fire -c conda-forge
```


## Training Instructions

```
# 1. Clone this repository to your disk.
git clone https://github.com/PengBoXiangShang/SiamRPN_plus_plus_PyTorch.git

# 2. Change working directory.
cd SiamRPN_plus_plus_PyTorch

# 3. Download training data. In this project, we provide the downloading and preprocessing scripts for ILSVRC2015_VID dataset. Please download ILSVRC2015_VID dataset (86GB). The cripts for other tracking datasets are coming soon.
cd data
wget -c http://bvisionweb1.cs.unc.edu/ilsvrc2015/ILSVRC2015_VID.tar.gz
tar -xvf ILSVRC2015_VID.tar.gz
rm ILSVRC2015_VID.tar.gz
cd ..

# 4. Preprocess data.
chmod u+x ./preprocessing/create_dataset.sh
./preprocessing/create_dataset.sh

# 5. Pack the preprocessed data into LMDB format to accelerate data loading.
chmod u+x ./preprocessing/create_lmdb.sh
./preprocessing/create_lmdb.sh

# 6. Start the training.
chmod u+x ./train.sh
./train.sh
```

## Acknowledgement
Many thanks to [Sisi](https://github.com/noCodegirl) who helps us to download the huge ILSVRC2015_VID dataset.
