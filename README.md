# SiamRPN++_PyTorch 
This is an unofficial PyTorch implementation of [SiamRPN++ (CVPR2019)](https://arxiv.org/pdf/1812.11703.pdf), implemented by **Peng Xu** and **[Jin Feng](https://github.com/JinDouer)**.

## Details of SiamRPN++ Network
As stated in the original paper, SiamRPN++ network has three parts.

1. **Backbone Network (modified ResNet-50).**

2. **SiamRPN Block.**

3. **Weighted Fusion Layer.** 

## Requirements
Ubuntu 14.04.1

Python 2.7.15

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
cd SiamRPN++_PyTorch

# 3. Download training data. In this project, we provide the downloading and preprocessing scripts for ILSVRC2015_VID dataset. Please download ILSVRC2015_VID dataset (86GB). The cripts for other tracking datasets are coming soon.
cd data
wget -c http://bvisionweb1.cs.unc.edu/ilsvrc2015/ILSVRC2015_VID.tar.gz
tar -xvf ILSVRC2015_VID.tar.gz
rm ILSVRC2015_VID.tar.gz
cd ..

# 4. Preprocess data.
chmod +x ./preprocessing/create_dataset.sh
./preprocessing/create_dataset.sh

# 5. Pack the preprocessed data into LMDB format to accelerate data loading.
chmod +x ./preprocessing/create_lmdb.sh
./preprocessing/create_lmdb.sh

# 6. Start the training.
chmod +x ./train.sh
./train.sh
```

## TODO


## Acknowledgement
Many thanks for [Sisi] (https://github.com/noCodegirl) helps us to download the huge ILSVRC2015_VID data.
