import time
import argparse
import collections
import json
import numpy as np

import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim import lr_scheduler
from torch.autograd import Variable
from torchvision import datasets, models, transforms
import torchvision
import pdb
from mmcv import Config
import os
import shutil
import time

from lib.model import model

from lib.dataloader.dataloader import CocoDataset, collater, AspectRatioBasedSampler, Augmentation
from torch.utils.data import Dataset, DataLoader

print('CUDA available: {}'.format(torch.cuda.is_available()))
def log_losses(cfg_name, timestr, out_dir, epoch_num, iter_num, classification_loss, regression_loss, reg_weight, time):
    exp_name = os.path.splitext(os.path.basename(cfg_name))[0]
    classification_loss = float('%.5f'%(classification_loss.item()))
    regression_loss = float('%.5f'%(regression_loss.item()))
    log_dict = {
            "epoch" : epoch_num,
            "iter": iter_num,
            "cls_loss": classification_loss,
            "regression_loss": regression_loss,
            "regression_weight": reg_weight,
            "time": time,
            }
    json_object = json.dumps(log_dict)
    with open(out_dir + "/" + exp_name + '-' + timestr +".json", "a") as log_file:
        log_file.write(json_object)
        log_file.write('\n')

def main(args=None):
    parser     = argparse.ArgumentParser(description='Simple training script for training a RetinaNet network.')

    parser.add_argument('--resume',type=bool, default=False)
    parser.add_argument('--resume_epoch',type=int, default=-1)
    parser.add_argument('--cfg', type=str)
    parser = parser.parse_args(args)
    
    cfg = Config.fromfile(parser.cfg)
    timestr = time.strftime("%H%M%S-%Y%m%d")
    if os.path.isdir(cfg['logger'].out_dir) == False:
        os.mkdir(cfg['logger'].out_dir) 
    
    with torch.cuda.device(cfg['optimization'].gpu_ids[0]):
        set_name=[iset for iset in cfg['dataset'].training_set.split('+')]
        # Create the data loaders
        dataset_train = CocoDataset(cfg['dataset'].path, set_name=set_name, transform=Augmentation(cfg['input_image'].norm_mean, cfg['input_image'].norm_std, cfg['input_image'].image_size, cfg['input_image'].augmentation))

        sampler = AspectRatioBasedSampler(dataset_train, batch_size=cfg['optimization'].image_per_gpu*len(cfg['optimization'].gpu_ids))
        dataloader_train = DataLoader(dataset_train, num_workers=cfg['optimization'].num_workers, collate_fn=collater, batch_sampler=sampler)

        # Create the model 
        if cfg['backbone'].type == 'ResNet' and cfg['backbone'].depth == 50:
            retinanet = model.resnet50(num_classes=dataset_train.num_classes(), cfg=cfg, pretrained=True)
        elif cfg['backbone'].type == 'ResNet' and cfg['backbone'].depth == 101:
            retinanet = model.resnet101(num_classes=dataset_train.num_classes(), cfg=cfg, pretrained=True)          
        elif cfg['backbone'].type == 'ResNeXt' and cfg['backbone'].depth == 101:
            retinanet = model.resneXt101(num_classes=dataset_train.num_classes(), cfg=cfg, pretrained=True)            
        else:
            raise ValueError('Not implemented')

        use_gpu = True

        if use_gpu:
            retinanet = retinanet.cuda()
    	
        retinanet = torch.nn.DataParallel(module=retinanet, device_ids=cfg['optimization'].gpu_ids).cuda()
        
        retinanet.training = True

        optimizer = optim.SGD(retinanet.parameters(), lr=cfg['optimization'].lr, momentum=cfg['optimization'].momentum, weight_decay = cfg['optimization'].weight_decay)

        scheduler = optim.lr_scheduler.MultiStepLR(optimizer, milestones=cfg['optimization'].lr_decay_epoch, gamma=cfg['optimization'].lr_decay_weight)

        begin_epoch=0
        if parser.resume==True:
            retinanet.load_state_dict(torch.load(cfg['out_dir']+'/coco_retinanet_'+str(parser.resume_epoch)+'.pt'))
            begin_epoch=parser.resume_epoch+1 
            for jj in range(begin_epoch):
                scheduler.step()

        cls_loss_hist = collections.deque(maxlen=300)
        reg_loss_hist = collections.deque(maxlen=300)
        if cfg['loss'].cls_loss.type == 'aLRPLoss' and cfg['loss'].reg_loss.type == 'aLRPLoss':
            cls_LRP_hist = collections.deque(maxlen=50)
            reg_LRP_hist = collections.deque(maxlen=50)
        tic_hist = collections.deque(maxlen=100)

        retinanet.train()
        retinanet.module.freeze_bn()

        print('Num training images: {}'.format(len(dataset_train)))

        # Self Balance is initialized here
        if cfg['loss'].cls_loss.type == 'aLRPLoss' and cfg['loss'].reg_loss.type == 'aLRPLoss':
                cls_weight = 1
                reg_weight = 50
        else:
                cls_weight = cfg['loss'].cls_loss.weight
                reg_weight = cfg['loss'].reg_loss.weight

    
        
        for epoch_num in range(begin_epoch, cfg['optimization'].total_epoch_num):

            retinanet.train()
            retinanet.module.freeze_bn()

            for iter_num, data in enumerate(dataloader_train):
                tic_start=time.time()
                optimizer.zero_grad()
                classification_loss, regression_loss = retinanet([data['img'].cuda().float(), data['annot']])

                classification_loss = classification_loss.mean()
                regression_loss = regression_loss.mean()
                
                loss = cls_weight*classification_loss + reg_weight*regression_loss

                if bool(loss == 0):
                    continue

                loss.backward()

                if cfg['optimization'].warmup and optimizer._step_count<=cfg['optimization'].warmup_step:
                    init_lr=cfg['optimization'].lr
                    warmup_lr=init_lr*cfg['optimization'].warmup_factor + optimizer._step_count/float(cfg['optimization'].warmup_step)*(init_lr*(1-cfg['optimization'].warmup_factor))
                    for ii_ in optimizer.param_groups:
                        ii_['lr']=warmup_lr 

                optimizer.step()
                tic_stop=time.time()
                tic_hist.append(tic_stop-tic_start)
                cls_loss_hist.append(float(classification_loss))
                reg_loss_hist.append(float(regression_loss))

                if iter_num % cfg['logger'].interval == 0:
                    if cfg['loss'].cls_loss.type == 'aLRPLoss' and cfg['loss'].reg_loss.type == 'aLRPLoss':
                        cls_LRP_hist.append(float(np.mean(cls_loss_hist)))
                        reg_LRP_hist.append(float(np.mean(reg_loss_hist)))
                        print('Epoch: {} | Iteration: {} | Classification loss: avg: {:1.5f}, cur: {:1.5f} | Localisation loss: avg: {:1.5f}, cur: {:1.5f} | Total loss: avg: {:1.5f}, cur: {:1.5f} | Speed: {:1.5f} sec./iter.| Self-Balance Weight: {:1.5f}'.format(epoch_num, iter_num, np.mean(cls_loss_hist), float(classification_loss), np.mean(reg_loss_hist), float(regression_loss), np.mean(reg_loss_hist)+np.mean(cls_loss_hist), float(regression_loss)+float(classification_loss), np.mean(tic_hist), float(reg_weight)))
                    else:
                        print('Epoch: {} | Iteration: {} | Classification loss: avg: {:1.5f}, cur: {:1.5f} | Localisation loss: avg: {:1.5f}, cur: {:1.5f} | Total loss: avg: {:1.5f}, cur: {:1.5f} | Speed: {:1.5f} sec./iter.'.format(epoch_num, iter_num, np.mean(cls_loss_hist), float(classification_loss), np.mean(reg_loss_hist), float(regression_loss), np.mean(reg_loss_hist)+np.mean(cls_loss_hist), float(regression_loss)+float(classification_loss), np.mean(tic_hist)))
                    log_losses(parser.cfg, timestr, cfg['logger'].out_dir, epoch_num, iter_num, np.mean(cls_loss_hist), np.mean(reg_loss_hist), reg_weight, np.mean(tic_hist))

                del classification_loss
                del regression_loss 

            scheduler.step()
            if cfg['loss'].cls_loss.type == 'aLRPLoss' and cfg['loss'].reg_loss.type == 'aLRPLoss':
                reg_weight = (np.mean(reg_LRP_hist)+np.mean(cls_LRP_hist))/np.mean(reg_LRP_hist)
                cls_LRP_hist.clear()
                reg_LRP_hist.clear()
            torch.save(retinanet.state_dict(), cfg['logger'].out_dir + '/{}_retinanet_{}.pt'.format(cfg['dataset'].type, epoch_num))
            
        retinanet.eval()

        torch.save(retinanet.state_dict(), cfg['logger'].out_dir + '/model_final.pt'.format(epoch_num))

if __name__ == '__main__':
    main()
