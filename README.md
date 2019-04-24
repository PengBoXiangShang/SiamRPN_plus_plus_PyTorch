# SiamRPN++_PyTorch 
This is an unofficial PyTorch implementation of [SiamRPN++ (CVPR2019)](https://arxiv.org/pdf/1812.11703.pdf), implemented by **Peng Xu** and **[Jin Feng](www.baidu.com)**.

If you would like to read the Chinese vesion of this readme document, please click [here]().

## Requirements
Ubuntu 14.04.1

Python 2.7.15

PyTorch 0.4.0

Other main requirements can be installed by:

cv2, xml, lmdb, fire, multiprocessing, 
```
# 1. install 
lmdb

# 2. 
```


## Usage Instructions

```
# 1. Clone this repository to your disk.
git clone 

# 2. Change working directory.
cd SiamRPN++_PyTorch

# 3. Download training data. In this project, we provide the downloading and preprocessing scripts for ILSVRC2015_VID dataset. Please download ILSVRC2015_VID dataset (86GB). The cripts for other tracking datasets are coming soon.
wget -c http://bvisionweb1.cs.unc.edu/ilsvrc2015/ILSVRC2015_VID.tar.gz

# 4. Do some data preprocessings, including cropping, padding, resizing, *etc*. Before , you need to .
chmod +x ./preprocessing/create_dataset.sh
./preprocessing/create_dataset.sh

# 5. Pack the data into [LMDB](http://www.lmdb.tech/doc/) format.
chmod +x ./preprocessing/create_lmdb.sh
./preprocessing/create_lmdb.sh

# 6. Start the training.
chmod +x ./train.sh
./train.sh
```

## TODO


## Acknowledgement
