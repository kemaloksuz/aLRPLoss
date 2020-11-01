# aLRP Loss: A Ranking-based, Balanced Loss Function Unifying Classification and Localisation in Object Detection 

The official implementation of aLRP Loss. Our implementation is based on [mmdetection](https://github.com/open-mmlab/mmdetection). You can also find a different implementation based on official AP Loss repository [in this link](https://github.com/kemaloksuz/aLRPLoss-AblationExperiments).

> [**aLRP Loss: A Ranking-based, Balanced Loss Function Unifying Classification and Localisation in Object Detection**](https://arxiv.org/abs/2009.13592),            
> [Kemal Oksuz](https://kemaloksuz.github.io/), Baris Can Cam, [Emre Akbas](http://user.ceng.metu.edu.tr/~emre/), [Sinan Kalkan](http://www.kovan.ceng.metu.edu.tr/~sinan/),
> *NeurIPS 2020. ([arXiv pre-print](https://arxiv.org/abs/2009.13592))*


## Summary

Average Localisation-Recall-Precision (aLRP) Loss is a ranking-based loss function to train object detectors by unifying localisation and classification branches. We define aLRP Loss as the average [Localisation Recall Precision](https://arxiv.org/abs/1807.01696) [1] errors on positive examples. To tackle the nondifferentiable nature of ranking during backpropagation, we combine error-driven update of perceptron learning with backpropogation by generalizing the training approach of AP Loss [2] to ranking-based loss functions (see Section 4 in the paper for details). 

With this formulation, aLRP Loss (i) enforces the predictions with large confidence scores to have better localisation, and correlates the classification and localisation tasks (see Figure below), (ii) has significantly less number of hyperparameters (i.e. only 1 hyperparameter) than the conventional loss formulation (i.e. the combination of classification and regression losses by a scaler weight), and (iii) guarantees balanced training (see Theorem 2 in the paper).

![aLRP Toy Example](assets/Teaser.png)


## How to Cite

Please cite the paper if you benefit from our paper or repository:
```
@inproceedings{aLRPLoss,
       title = {A Ranking-based, Balanced Loss Function Unifying Classification and Localisation in Object Detection},
       author = {Kemal Oksuz and Baris Can Cam and Emre Akbas and Sinan Kalkan},
       booktitle = {Advances in Neural Information Processing Systems (NeurIPS)},
       year = {2020}
}
```
## RetinaNet Results

|    Method     |    Backbone     |  Scale   | AP (test-dev) | AP (minival) | Model  | Log  |
| :-------------: | :-------------: | :-----: | :------------: | :------------: | :-------: |:-------: |
| AP Loss* |    ResNet-50    |  500  |   35.7   |   35.4    | [model](https://drive.google.com/file/d/17T2TqSA_mexGx1CSsTj6H6-46jxjafRC/view?usp=sharing)|[log](https://drive.google.com/file/d/1pkSLBDbTLqeRqSUZlxUMxofSR6FQO_0U/view?usp=sharing)|
| aLRP Loss (GIoU)* |    ResNet-50    |  500  |   39.5   |   39.0   | [model](https://drive.google.com/file/d/1K-YGYrVMRGp0M6w7_PgIYAp2kHFOJo8C/view?usp=sharing)|[log](https://drive.google.com/file/d/1iGj9zb68sksJsYKG68T8sd900PMOTbqd/view?usp=sharing)|
| aLRP Loss (GIoU+ATSS) |    ResNet-50    |  500  |   41.3   |   41.0   | [model](https://drive.google.com/file/d/15wKL1YVPrCBLpPKtaOe8VQR2uhyKUB0L/view?usp=sharing)|[log](https://drive.google.com/file/d/1blRO6-C2itppoLCKt6q9CQEl8POn4J4q/view?usp=sharing)|
|aLRP Loss (GIoU+ATSS)|    ResNet-101    |  500  |   42.8   |   42.2   | [model](https://drive.google.com/file/d/1Cozn9fB44IPq26SN1L-qbGqbLucSMnKK/view?usp=sharing)|[log](https://drive.google.com/file/d/1T8vJug62foZna9VRhE-m5i_xpDZpC-TO/view?usp=sharing)|
|aLRP Loss (GIoU+ATSS)|    ResNext-101-64x4d    |  500  |   44.6   |   44.5   | [model](https://drive.google.com/file/d/1YB7R68VDsruBVI1YhqMVO2XO2EgX3aUA/view?usp=sharing)|[log](https://drive.google.com/file/d/1oP2jDBkQfK3x0G2kvS-ZimM9IAiTfQ_t/view?usp=sharing)|
|aLRP Loss (GIoU+ATSS)|    ResNet-101    |  800  |   45.9   |   45.4   | [model](https://drive.google.com/file/d/1L74v4LLWt5uYDEeSBMhKECpztqNG3QIQ/view?usp=sharing)|[log](https://drive.google.com/file/d/1lhz_UI5kKlhZXI1DQ7Gt1ph-DJRS-zW4/view?usp=sharing)|
|aLRP Loss (GIoU+ATSS)|    ResNext-101-64x4d    |  800  |   47.8   |   47.2   | [model](https://drive.google.com/file/d/1-sJoRM7u43rLx9ntJkvuE4BmOwEfukDs/view?usp=sharing)|[log](https://drive.google.com/file/d/1TROgjqCWmlWm9wH8YIYV8V5IsaVVY4w8/view?usp=sharing)|
|aLRP Loss (GIoU+ATSS)|    ResNext-101-64x4d-DCN    |  800  |   48.9   |   48.6   | [model](https://drive.google.com/file/d/1vO_wAPzVQm8-tCj0ReoJeo6T0EpeRv61/view?usp=sharing)|[log](https://drive.google.com/file/d/1Q6HALIEg60bpKXuJIiiLF9IzdsYdZ8BC/view?usp=sharing)|

*Following the learning rate scheduling adopted by AP Loss[2], these models are trained for 100 epochs by decreasing learning rate in 60th and 80th epochs. The rest of the models are trained for 100 epochs by scheduling the learning rate in 75th and 95th epochs.

## FoveaBox Results

|    Method     |  Backbone   | AP (minival) | oLRP (minival) | Model  |  Log  |
| :-------------: | :-----: | :------------: | :------------: | :-------: | :-------: |
|    Focal Loss+Smooth L1 |  ResNet-50  |   38.3   |   68.8  | [model](https://drive.google.com/file/d/1uOB7r6XuQvEzPvZnmHvmL69qSYFQj2mR/view?usp=sharing)|[log](https://drive.google.com/file/d/1yiKJ8UHEz1Uql-Qi4rUEVHheeFaji0Va/view?usp=sharing)|
|    AP Loss+Smooth L1  | ResNet-50 |   36.5  |   69.8   | [model](https://drive.google.com/file/d/1FyaKNJOE6Rbq2bSN6SAWnE8t_hW7OIFC/view?usp=sharing)|[log](https://drive.google.com/file/d/1O5H2RdRijVJzJgHtyxX7qrThsA_Rcrft/view?usp=sharing)|
|    aLRP Loss | ResNet-50 |   39.7   |   67.2  | [model](https://drive.google.com/file/d/1f76mMqp7yAPIKzj5Cb6Moy6Dk13I1qny/view?usp=sharing)|[log](https://drive.google.com/file/d/1UGbcaAgAwL0P_dbY5RDhy53DqPDY20j7/view?usp=sharing)|

## Faster R-CNN Results

|    Method     |  Backbone   | AP (minival) | oLRP (minival) | Model  |  Log  |
| :-------------: | :-----: | :------------: | :------------: | :-------: | :-------: |
|    Cross Entropy+Smooth L1 |  ResNet-50  |   37.8   |   69.3  | [model](https://drive.google.com/file/d/1eUahlGWfArXhc5e58IQWT7QU0TVMZGAM/view?usp=sharing)|[log](https://drive.google.com/file/d/19_0pT3H3q1I5oNTMN8rbRgPSjL-hltaL/view?usp=sharing)|
|    Cross Entropy+GIoU Loss  | ResNet-50 |   38.2  |   69.0   | [model](https://drive.google.com/file/d/1OSdruWbtYmC35BaM7pz9Oe34OuVnyu71/view?usp=sharing)|[log](https://drive.google.com/file/d/15IlJ8G5G0COF-JktijcYi37qcNkY4X0Q/view?usp=sharing)|
|    aLRP Loss | ResNet-50 |   40.7   |   66.7  | [model](https://drive.google.com/file/d/1NgbI9_5f6giKLfT9UlZZNoPH6D-Cm3U8/view?usp=sharing)|[log](https://drive.google.com/file/d/1IivL3d693s_jYD5CoUuRoSrLTKj5tTcp/view?usp=sharing)|

## Specification of Dependencies and Preparation

- Please see requirements.txt and requirements folder for the rest of the dependencies.
- Please refer to [install.md](docs/install.md) for installation instructions of MMDetection.
- Please see [getting_started.md](docs/getting_started.md) for dataset preparation and the basic usage of MMDetection.

## Training Code
The configuration files of all models listed above can be found in the `configs/alrp_loss` folder. You can follow [getting_started.md](docs/getting_started.md) for training code. As an example, to train aLRP Loss (GIoU+ATSS) on 4 GPUs as we did, use the following command:

```
./tools/dist_train.sh configs/alrp_loss/alrp_loss_retinanet_r50_fpn_ATSS_100e_coco500.py 4
```


## Test Code
The configuration files of all models listed above can be found in the `configs/alrp_loss` folder. You can follow [getting_started.md](docs/getting_started.md) for test code. As an example, to test aLRP Loss (GIoU+ATSS), first download or train a model, then use the following command to test on multiple GPUs:

```
./tools/dist_test.sh configs/alrp_loss/alrp_loss_retinanet_r50_fpn_ATSS_100e_coco500.py -PATH-TO-TRAINED-MODEL 4 --eval bbox

```
You can also test a model on a single GPU with the following example command:
```
python tools/test.py configs/alrp_loss/alrp_loss_retinanet_r50_fpn_ATSS_100e_coco500.py -PATH-TO-TRAINED-MODEL 4 --eval bbox

```

## License
Following MMDetection, this project is released under the [Apache 2.0 license](LICENSE).

## References
[1] Oksuz K, Cam BC, Akbas E, Kalkan S, Localization recall precision (LRP): A new performance metric for object detection, ECCV 2018.  
[2] Chen K, Li J, Lin W, See J, Wang J, Duan L, Chen Z, He C, Zou J, Towards Accurate One-Stage Object Detection With AP-Loss, CVPR 2019 & TPAMI.

## Contact

This repo is maintained by [Kemal Oksuz](http://github.com/kemaloksuz) and [Baris Can Cam](http://github.com/cancam).
