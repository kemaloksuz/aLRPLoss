import time
import argparse
import collections

import numpy as np

import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim import lr_scheduler
from torch.autograd import Variable
from torchvision import datasets, models, transforms
import torchvision

from lib.model import model
from lib.dataloader.dataloader import CocoDataset, Resizer, Normalizer
from torch.utils.data import Dataset, DataLoader

from lib.util import coco_eval
from mmcv import Config
def main(args=None):
    
    parser     = argparse.ArgumentParser(description='Simple testing script for testing a RetinaNet network.')

    parser.add_argument('--cfg', type=str)
    parser = parser.parse_args(args)
    
    cfg = Config.fromfile(parser.cfg)

    set_name=[iset for iset in cfg['dataset'].test_set.split('+')]
    dataset_val = CocoDataset(cfg['dataset'].path, set_name=set_name, transform=transforms.Compose([Normalizer(cfg['input_image'].norm_mean, cfg['input_image'].norm_std), Resizer(cfg)]))

    # Create the model 
    if cfg['backbone'].type == 'ResNet' and cfg['backbone'].depth == 50:
        retinanet = model.resnet50(num_classes=dataset_val.num_classes(), cfg=cfg, pretrained=True)
    elif cfg['backbone'].type == 'ResNet' and cfg['backbone'].depth == 101:
        retinanet = model.resnet101(num_classes=dataset_val.num_classes(), cfg=cfg, pretrained=True)          
    elif cfg['backbone'].type == 'ResNext' and cfg['backbone'].depth == 101:
        retinanet = model.resneXt101(num_classes=dataset_val.num_classes(), cfg=cfg, pretrained=True)            
    else:
        raise ValueError('Not implemented')

    use_gpu=True
    if use_gpu:
        retinanet = retinanet.cuda()
    retinanet = torch.nn.DataParallel(module=retinanet,device_ids=[cfg['optimization'].gpu_ids[0]]).cuda()
    for test_epoch in cfg['test'].epoch:
        print('Evaluating epoch {}'.format(test_epoch))
        with torch.cuda.device(cfg['optimization'].gpu_ids[0]): 
            retinanet.load_state_dict(torch.load(cfg['logger'].out_dir + '/' +cfg['dataset'].type+'_retinanet_'+str(test_epoch)+'.pt',map_location=lambda storage, loc: storage.cuda()))
            retinanet.training = False
            retinanet.eval()
            retinanet.module.freeze_bn()
            coco_eval.evaluate_coco(dataset_val, retinanet, cfg['test'].flip_augmentation)

if __name__ == '__main__':
    main()
