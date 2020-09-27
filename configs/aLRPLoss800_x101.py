import numpy as np
import os


# 1. Anchor Parameters

# Aspect Ratios of the anchors
anchor_ratios = np.array([0.5,1.0,2.0])
# Scales of the anchors
anchor_scales = np.array([2**0,2**(1.0/2.0)])
anchor_base_scale = 4
num_anchors = len(anchor_ratios)*len(anchor_scales)


# 2. The backbone network

# Set 'R50' for ResNet-50, 'R101' for ResNet-101 and 'X101' for ResNeXt101-32x8
depth = 'X101'


# 3. Dataset settings

# Training image size. Two options: If it is S then training image size is SxS. If it is [S, L] then training image size is S x L. Multiscale training is not supported now.
train_img_size = 800
# Name of the Dataset, only 'coco' supported
dataset = 'coco'
# The partition of the data. test_set can be set as 'val2017' to test on minival.
data_partition = {'dataset':'coco', 'path':'data/coco', 'train_set':'train2017', 'test_set':'test-dev2017'}
# Mean and standard deviation to normalize the data. With the provided backbones, do not change the mean. 
pixel_mean = np.array([[[102.9801, 115.9465, 122.7717]]])
# If you use the provided ResNet-50 or ResNet-101 then pixel_std is all 1 since they are converted from caffe models. 
# Else if you use the ResNext we provide from pytorch, then set std as [[[58.395, 57.120, 57.375]]].
pixel_std = np.array([[[58.395, 57.120, 57.375]]])
# We provide 3 augmentation styles: 'SSD_Style', 'No', 'Horizontal_Flip'. 
augmentation_style = 'SSD_Style'

# 4. Positive Negative Assignment

assigner = dict(type='IoUAssigner', pos_min_IoU=0.50, neg_max_IoU=0.40)

# 4. Loss Function

# You can choose the classification loss as 'aLRP' or 'AP'
classification_loss = 'aLRP'
# The regression possibilities are as follows: 
# If classification loss is aLRP then regression loss is 'aLRP'. 
# If classification loss is AP then you can choose between 'SmoothL1', 'IoULoss' or 'GIoULoss'.
regression_loss = 'aLRP'

#4.1 Settings for aLRP Loss. 
# If you use aLRP Loss, then choose an IoU type: 'IoU' or 'GIoU'
iou_type = 'GIoU'

#4.2 Settings for AP Loss
#Unless you use AP Loss, please do not change the following setting
#aLRP Loss does not have any weight.
regressor_weight = 1
classifier_weight = 1


# 5. Training Settings

# GPU ids
gpu_ids = [0,1,2,3]
# Number of images per GPU
batch_size = 4
# Learning Rate 
# 0.008 for aLRP Loss. 0.002 is used for AP Loss in the paper.
# lr is for 32 batch size, set accordingly with different batch sizes.
lr = 0.004
# Number of epochs to train
epochs = 100
# Decrease the leraning rate by a factor of 0.1 at the following epochs
lr_step = [60,80]
# Whether or not to use warm up.
warmup = True
# Gradually increase the learning rate for the first warmup_step iterations.
warmup_step = 500
warmup_factor = 0.33333333


#6. Test Configuration

# Epoch to test, by default, it is the last epoch. You can also provide a list of epochs to test.
test_epoch = [epochs - 1]
# List of scales to test. Provide a list for multiscale test.
test_img_size = [800]
# Can focus different sizes in different scales when multiscale testing is used, 0 to ignore
multi_scale_ths = [0]
# Whether to use flip augmentation during testing, by default, no flip augmentation
test_flip_augmentation = False


#7. Other Issues:

# Logging Frequency. If K then, logs at every K iterations. 
log_interval = 50
# Folder to put the trained models and the log file.
out_dir = os.path.join('./models', 'aLRPLoss800_x101')
