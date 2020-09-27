from __future__ import print_function

from pycocotools.coco import COCO
#Use our evaluation for mAP@0.9
from .coco_eval_oLRP_largerIoU import COCOeval

import numpy as np
import json
import os
from ..util.utils import BBoxTransform, ClipBoxes

import torch
import torchvision
import pdb

import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import matplotlib.patches as patches

def get_results(anchors, classifications, regressions, im_info):
    regressBoxes = BBoxTransform()

    clipBoxes = ClipBoxes()
    num_classes = 80

    box_all=[]
    cls_all=[]
    score_all=[]
    for ii, anchor in enumerate(anchors):
        classification=classifications[ii].view(-1)
        regression=regressions[ii]

        classification=torch.sigmoid(classification)

        ###filter
        num_topk=min(1000, classification.size(0))
        ordered_score,ordered_idx=classification.sort(descending=True)
        ordered_score=ordered_score[:num_topk]
        ordered_idx=ordered_idx[:num_topk]
 
        if ii<4:
            score_th=0.01
        else:
            score_th=0

        keep_idx=(ordered_score>score_th)
        ordered_score=ordered_score[keep_idx]
        ordered_idx=ordered_idx[keep_idx]

        anchor_idx = ordered_idx // num_classes
        cls_idx = ordered_idx % num_classes

        transformed_anchor = regressBoxes(anchor[:,anchor_idx,:], regression[:,anchor_idx,:])
                
        transformed_anchor = clipBoxes(transformed_anchor, im_info[0])
                
        box_all.append(transformed_anchor[0])
        cls_all.append(cls_idx)
        score_all.append(ordered_score)

    box_all=torch.cat(box_all,dim=0)
    cls_all=torch.cat(cls_all,dim=0)
    score_all=torch.cat(score_all,dim=0)
    return score_all, cls_all, box_all

def evaluate_coco(dataset, model, flip_aug=False):
    
    model.eval()
    
    with torch.no_grad():

        # start collecting results
        results = []
        image_ids = []

        for index in range(len(dataset)):
            data = dataset[index]
 
            box_all_scales = torch.tensor([]).to('cuda', dtype=torch.float64)
            score_all_scales = torch.tensor([]).to('cuda', dtype=torch.float32)
            cls_all_scales = torch.tensor([]).to('cuda', dtype=torch.int64)
            scale_vector = torch.tensor([]).to('cuda', dtype=torch.float64)

            for scale_idx in range(0, len(data['img'])):
                scale = data['scale'][scale_idx]
                img = torch.from_numpy(data['img'][scale_idx]).permute(2, 0, 1).cuda().float().unsqueeze(dim=0)

                #You can see the original image by commenting out the following
                #img_np=img.cpu().numpy()
                #fig,ax = plt.subplots(1)            
                #ax.imshow(img_np[0][0])

                # run network
                anchors, classifications, regressions = model(img)
                score_all, cls_all, box_all = get_results(anchors, classifications, regressions, data['im_info'][scale_idx]) 
                if flip_aug:
                    #Flip the image considering padding 
                    img_f = torch.cat((torch.flip(img[:,:,:,:data['im_info'][scale_idx][0,1]], [3]), img[:,:,:,data['im_info'][scale_idx][0,1]:]),dim=3)
                    #img_np_f=img_f.cpu().numpy()
                    #imgplot_f = plt.imshow(img_np_f[0][0])
                    #plt.show()     

                    anchors_f, classifications_f, regressions_f = model(img_f) 
                    score_all_f, cls_all_f, box_all_f = get_results(anchors_f, classifications_f, regressions_f, data['im_info'][scale_idx])

                    #Correct flipped boxes
                    box_all_reverse_f = torch.zeros(box_all_f.shape).cuda().double()
                    box_all_reverse_f[:, 0] = data['im_info'][scale_idx][0,1] - box_all_f[:,2]
                    box_all_reverse_f[:, 1] = box_all_f[:,1]
                    box_all_reverse_f[:, 2] = data['im_info'][scale_idx][0,1] - box_all_f[:,0]
                    box_all_reverse_f[:, 3] = box_all_f[:,3]

                    #Concat the results (append the flipped results)
                    box_all=torch.cat((box_all, box_all_reverse_f), dim=0)
                    score_all=torch.cat((score_all, score_all_f), dim=0)
                    cls_all=torch.cat((cls_all, cls_all_f), dim=0)
                
                # scale boxes wrt img original size
                box_all /= scale
                # get valid indices wrt size th.
                if data['size_th'][scale_idx] != 0:
                    keep_size = (torch.min(box_all[:,2] - box_all[:,0],\
                                      box_all[:,3] - box_all[:,1]) < data['size_th'][scale_idx]).nonzero().reshape(-1)
                    # filter out dets
                    box_all = box_all[keep_size]
                    score_all = score_all[keep_size]
                    cls_all = cls_all[keep_size]

                box_all_scales = torch.cat((box_all_scales, box_all), dim=0)
                score_all_scales = torch.cat((score_all_scales, score_all), dim=0)
                cls_all_scales = torch.cat((cls_all_scales, cls_all), dim=0)

            #apply nms
            keep=torchvision.ops.boxes.batched_nms(box_all_scales, score_all_scales, cls_all_scales, 0.5)
            keep=keep[:100]
            
            scores=score_all_scales[keep].cpu()
            labels=cls_all_scales[keep].cpu()
            boxes=box_all_scales[keep, :].cpu()
            #scales = scale_vector[keep].cpu()
            #Visulization of the results, for more clear results please apply threshold
            #i=0
            #for box in boxes:
            #    if scores[i]>0.75:
            #        rect = patches.Rectangle((box[0],box[1]),box[2]-box[0],box[3]-box[1],linewidth=1,edgecolor='r',facecolor='none')
            #        ax.add_patch(rect)
            #    i+=1
            #plt.show()

            # convert boxes to w/h representation.
            boxes[:,2] = boxes[:,2] - boxes[:,0] +1
            boxes[:,3] = boxes[:,3] - boxes[:,1] +1
  

            if boxes.shape[0] > 0:
 
               # compute predicted labels and scores
               for box_id in range(boxes.shape[0]):
                    score = float(scores[box_id])
                    label = int(labels[box_id])
                    box = boxes[box_id, :] 

                    # append detection for each positively labeled class
                    image_result = {
                        'image_id'    : dataset.image_ids[index][1],
                        'category_id' : dataset.label_to_coco_label(label),
                        'score'       : float(score),
                        'bbox'        : box.tolist(),
                    }

                    # append detection to results
                    results.append(image_result)

            # append image to list of processed images
            image_ids.append(dataset.image_ids[index][1])

            # print progress
            print('{}/{}'.format(index, len(dataset)), end='\r') 

        # write output
        json.dump(results, open('./results/{}_bbox_results.json'.format(dataset.set_name[0]), 'w'), indent=4)

        if 'test' in dataset.set_name[0]:
            return

        # load results in COCO evaluation tool
        coco_true = dataset.coco
        coco_true = coco_true[list(coco_true.keys())[0]]
        coco_pred = coco_true.loadRes('./results/{}_bbox_results.json'.format(dataset.set_name[0]))

        # run COCO evaluation
        coco_eval = COCOeval(coco_true, coco_pred, 'bbox')
        coco_eval.params.imgIds = image_ids
        coco_eval.evaluate()
        coco_eval.accumulate()
        coco_eval.summarize()

        return
