import numpy as np
import os

# 1. Dataset 
dataset = dict(type = 'coco', #Dataset
	       	   path = 'data/coco', #Dataset path
	       	   training_set = 'train2017', #Training split
	       	   test_set = 'val2017') #Test split

# 2. Input 
input_image = dict(image_size = 512, #for input S, training image size is SxS
	               norm_mean = np.array([[[102.9801, 115.9465, 122.7717]]]), #Mean to normalize the data
	               norm_std = np.array([[[1., 1., 1.]]]), #Standard Deviation to normalize the data
	               augmentation = 'SSD_Style' # We provide 3 augmentation styles: 'SSD_Style', 'No_Training_Augmentation', 'Horizontal_Flip' 
	               )
# 3. Anchor Parameters
anchor_design = dict(aspect_ratio = np.array([0.5,1.0,2.0]),
	                 scale = np.array([2**0,2**(1.0/2.0)]),
	                 base_scale = 4)

# 4. Backbone
# ResNet-50, ResNet-101 and ResNeXt101-32x8 are supported 
backbone = dict(type = 'ResNet',
	            depth = 50)

# 5. Assignment Method
assigner = dict(type='IoUAssigner', 
	            pos_min_IoU=0.50, 
	            neg_max_IoU=0.40)

# 6. Loss Function
loss = dict(cls_loss = dict(type = 'aLRPLoss', delta = 1.0),
	        reg_loss = dict(type = 'aLRPLoss', iou_type = 'GIoU'))

# 7. Optimization Settings
optimization = dict(image_per_gpu= 8, # Number of images per GPU
	                lr = 0.008, # Learning Rate 
	                total_epoch_num = 100, # Number of epochs to train
	                lr_decay_epoch = [60,80], # Decrease the leraning rate by a factor of 0.1 at the following epochs
	                lr_decay_weight = 0.10,
	                momentum = 0.9, # Momentum factor
	                weight_decay = 1e-4, # Weight decay factor
	                gpu_ids = [0], # GPU ids
	                num_workers = 4,
	                warmup = True, # Whether or not to use warm up.
	                warmup_step = 500, 
	                warmup_factor = 1/3)

# 8. Test Configuration
test = dict(epoch = [99], # Epochs to test
	        image_size = [500], # List of scales to test. Provide a list for multiscale test.
	        flip_augmentation = False, # Whether to use flip augmentation during testing, by default, no flip augmentation
	        multi_scale_thr = [0]) # Can focus different sizes in different scales when multiscale testing is used, 0 to ignore

# 9. Logging and Output Directory
logger = dict(interval = 50, # Logging Frequency and folder
	          out_dir = os.path.join('./models', 'aLRPLoss500_r50'))