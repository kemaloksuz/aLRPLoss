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
from lib.model import model_resNext
from lib.dataloader.dataloader import CocoDataset, VocDataset, Resizer, Resizer_Equal, Normalizer
from torch.utils.data import Dataset, DataLoader

from lib.util import coco_eval
from lib.util import voc_eval
from mmcv import Config
import pdb
def main(args=None):
    
    parser     = argparse.ArgumentParser(description='Simple testing script for testing a RetinaNet network.')

    parser.add_argument('--cfg', type=str)
    parser = parser.parse_args(args)
    
    cfg = Config.fromfile(parser.cfg)
  
    set_name=[iset for iset in cfg['data_partition']['test_set'].split('+')]
    if cfg['data_partition']['dataset']=='coco': 
        dataset_val = CocoDataset(cfg['data_partition']['path'], set_name=set_name, transform=transforms.Compose([Normalizer(cfg['pixel_mean'], cfg['pixel_std']), Resizer(cfg)]))
    elif cfg['data_partition']['dataset']=='voc':
        dataset_val = VocDataset(cfg['data_partition']['path'], set_name=set_name, transform=transforms.Compose([Normalizer(cfg['pixel_mean'], cfg['pixel_std']), Resizer(cfg)]))
    else:
        raise ValueError('Not implemented.')	

    if cfg['depth'] == 'R50':
        retinanet = model.resnet50(num_classes=dataset_val.num_classes(), cfg=cfg, pretrained=True)
    elif cfg['depth'] == 'R101':
        retinanet = model.resnet101(num_classes=dataset_val.num_classes(), cfg=cfg, pretrained=True)
    elif cfg['depth'] == 'X101':
        retinanet = model_resNext.resneXt101(num_classes=dataset_val.num_classes(), cfg=cfg, pretrained=True)
    else:
        raise ValueError('Not implemented.')	

    use_gpu=True
    if use_gpu:
        retinanet = retinanet.cuda()
    retinanet = torch.nn.DataParallel(module=retinanet,device_ids=[cfg['gpu_ids'][0]]).cuda()
    for test_epoch in cfg['test_epoch']:
        print('Evaluating epoch {}'.format(test_epoch))
        with torch.cuda.device(cfg['gpu_ids'][0]): 
            retinanet.load_state_dict(torch.load(cfg['out_dir'] + '/' +cfg['dataset']+'_retinanet_'+str(test_epoch)+'.pt',map_location=lambda storage, loc: storage.cuda()))

            retinanet.training = False

            retinanet.eval()
            retinanet.module.freeze_bn()

            if cfg['data_partition']['dataset']=='coco':
                coco_eval.evaluate_coco(dataset_val, retinanet, cfg['test_flip_augmentation'])
            elif cfg['data_partition']['dataset']=='voc':
                voc_eval.evaluate_voc(dataset_val, retinanet)
            else:
                raise ValueError('Not implemented.')

if __name__ == '__main__':
    main()
