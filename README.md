# aLRP Loss: A Ranking-based, Balanced Loss Function Unifying Classification and Localisation in Object Detection

## Specification of Dependencies
- Python 3.7
- PyTorch 1.2+
- CUDA 10.0
- NumPy 1.16+
- [mmcv](https://github.com/open-mmlab/mmcv)


## Training Code
You can train a model by running the following code:

```
python train.py --cfg PATH_TO_CONFIG_FILE
```
The configuration files are in the `config` folder. We provide all the configuration files used for comparison with SOTA methods in Table 4:
```
- aLRPLoss500_r101.py
- aLRPLoss500_x101.py
- aLRPLoss800_r101.py
- aLRPLoss800_x101.py
```
Also using our code, the results of AP Loss can be reproduced with the following configuration files:
```
- APLoss500_r101.py
- APLoss800_r101.py
```
Different options including the ablation experiments (and more) can be conducted by modifying configuration files. Further instructions/options are provided in the configuration files.

## Evaluation Code
You can test a model by running the following code:

```
python test.py --cfg PATH_TO_CONFIG_FILE
```

Currently, the configuration files are ready to reproduce test-dev results in Table 4. Replacing the test directory from "test-dev2017" "val2017" will conduct the test on validation set. More information can be found in the configuration files.

## Table of Main Results with Pretrained Models

|    Backbone     |  Scale   | oLRP (minival) | AP (minival) | AP (test-dev) | Download  |
| :-------------: | :-----: | :------------: | :------------: | :----: | :-------: |
|    ResNet-101    |  500  |   66.1   |   41.2   |  41.7  | [model](https://drive.google.com/file/d/1ihhXuh49_PeGkfldo9LoD7GXWTi-ZaNc/view?usp=sharing)|
|    ResNext-101-32x8d    | 500 |   64.8  |   42.8   |  43.4  | [model](https://drive.google.com/file/d/1NKFu0gxjEPbyFvYzTFppYrZHBeo4XIis/view?usp=sharing)|
|    ResNet-101   | 800 |   64.5   |   43.6   |  44.1  | [model](https://drive.google.com/file/d/1vymO5NeUTSHX2ZYWYtiJv-80T4FtSmAp/view?usp=sharing)|
| ResNext-101-32x8d | 800  |   62.7   |   45.4   |  45.9  | [model](https://drive.google.com/file/d/1gCrjqCc9i5-A4y-R6Xxbfpv1DiAH41Fy/view?usp=sharing)|

You can find the result files obtained from COCO Evaluation Server for test-dev under `COCO test-dev results` folder.
## Reproducing Instructions
We provide two different options to reproduce our results:

- You can download our pretrained model and directly test it, or 
- You can train a model first and then test the trained model. 

In any case, you need to prepare the repository using the preparation instructions below.

### Preparation Instructions:

- Install required packages:
```
pip install "git+https://github.com/open-mmlab/cocoapi.git#subdirectory=PythonAPI"
pip install opencv-python
```
- Create necessary directories:
```
mkdir data models results
```
- Prepare Data: You can download the images and their annotations from [the download page of COCO dataset](http://cocodataset.org/#download). Please do not forget to change the name of the downloaded `test` folder name as `test-dev2017`. If you have had the dataset, then just create a symbolic link to yours by:
```
ln -s $YOUR_PATH_TO_coco data/coco
```
Finally, the data directories should be arranged like:
```
├── data
│   ├── coco
│   │   ├── annotations
│   │   ├── images
│   │   │   ├── train2017
│   │   │   ├── val2017
│   │   │   ├── test-dev2017
```
- Prepare the pre-trained backbone models under `models` directory by following the structure below:
```
├── models
│   ├── resnet50-pytorch.pth
|   ├── resnet101-pytorch.pth
|   ├── resnext101_32x8d-8ba56ff5.pth
```
We use ResNet-50 and ResNet-101 pre-trained models provided by the official AP-Loss repository. Accordingly, you can download them from [this link](https://1drv.ms/u/s!AgPNhBALXYVSa1pQCFJNNk6JgaA?e=PqhsWD). For ResNext-101, we use the model provided by Pytorch [in this link](https://download.pytorch.org/models/resnext101_32x8d-8ba56ff5.pth) to download.


### Option 1: Testing Pre-Trained models

- Download the pretrained model that you want to test from the link provided in the Table of Results. 

- Create a folder named with the configuration file name under `models` folder. For example, if you want to test for scale 500 and ResNet-101 backbone, then you need to use aLRPLoss500_r101.py configuration file. Accordingly, create a folder named as `aLRPLoss500_r101` under `models`, and put the downloaded model under this folder. For this example, the model path should be: `models/aLRPLoss500_r101/coco_retinanet_159.pt`.

- Locate the json file, which is generated automatically at the end of the test under the `results` folder, and rename it as `detections_test-dev2017_aLRPLoss_results.json`, and finally compress it as `detections_test-dev2017_aLRPLoss_results.zip`. You can submit this zip file to [COCO evaluation server](https://competitions.codalab.org/competitions/20794). Please check the instructions provided in [this link](http://cocodataset.org/#upload) for more details to submit detection outputs.

### Option 2: Training and Testing

- Train the network using the instructions in Training Code, and you will find a model saved at the end of each epoch in models/MODEL_NAME folder. MODEL_NAME will be automatically created by the training code.

- Follow Option 1 to test and get the results. You can modify the config file to see the performance for a specific epoch. The instructions are provided in the config file.

## The code segments directly related with the implementation of aLRP Loss

This part provides an insight on the code by addressing the most relevant parts to see the interaction of the branches.

- You can see how classification scores affect regression loss in lib/model/model.py lines 278-286. Notice that cumulative sum enforces the examples with larger score involve in the regression loss several times.

- You can see the difference between AP Loss and aLRP Classification component from lib/model/alrploss.py lines 55-73. Notice that the localisation errors contribute to the gradients backpropagated from classification branch.

- You can see the implementation of the self-balance extension in train.py lines 158-173.

- You can check any config file under config folder for a general understanding provided by comments.
