import numpy as np
import torch
import torch.nn as nn
import pdb

def calc_iou(a, b):

    a=a.type(torch.cuda.DoubleTensor)
    b=b.type(torch.cuda.DoubleTensor)
    area = (b[:, 2] - b[:, 0]+1) * (b[:, 3] - b[:, 1]+1)

    iw = torch.min(torch.unsqueeze(a[:, 2], dim=1), b[:, 2]) - torch.max(torch.unsqueeze(a[:, 0], 1), b[:, 0])+1
    ih = torch.min(torch.unsqueeze(a[:, 3], dim=1), b[:, 3]) - torch.max(torch.unsqueeze(a[:, 1], 1), b[:, 1])+1

    iw = torch.clamp(iw, min=0)
    ih = torch.clamp(ih, min=0)

    ua = torch.unsqueeze((a[:, 2] - a[:, 0]+1) * (a[:, 3] - a[:, 1]+1), dim=1) + area - iw * ih

    #ua = torch.clamp(ua, min=1e-8)

    intersection = iw * ih

    IoU = intersection / ua

    return IoU

def bbox_overlaps(bboxes1, bboxes2):
    '''
    Calculates IoU between model prediction and target to compute
    IoULoss.

    Inputs:
        pred -> NX4 Bounding-boxes from model prediction.
        target -> Nx4 Target bounding boxes.
    
    Returns:
        ious -> Nx1 IoU value between prediction and target bounding boxes.
    '''
    
    # check if prediction and target samples equal
    num_pred = bboxes1.size(0)
    num_target = bboxes2.size(0)

    assert num_pred == num_target

    # calculate max-top-left, min-bottom-right points
    top_left = torch.max(bboxes1[:,:2], bboxes2[:,:2])
    bottom_right = torch.min(bboxes1[:,2:], bboxes2[:,2:])
    wh = (bottom_right - top_left + 1).clamp(min=0)
    overlap = wh[:,0] * wh[:,1]
    # calculate area for box1
    area1 = (bboxes1[:,2] - bboxes1[:,0] + 1) *(
            bboxes1[:,3] - bboxes1[:,1] + 1)
    
    # calculate area for box2
    area2 = (bboxes2[:,2] - bboxes2[:,0] + 1) *(
            bboxes2[:,3] - bboxes2[:,1] + 1)
    
    # calculate ious
    ious = overlap / (area1 + area2 - overlap)
    
    return ious

def compute_giou(pred, target, eps=1e-7):
    
    num_pred = pred.size(0)
    num_target = target.size(0)

    assert num_pred == num_target

    # calculate max-top-left and min-bottom-right points for overlap
    top_left = torch.max(pred[:,:2], target[:,:2])
    bottom_right = torch.min(pred[:, 2:], target[:, 2:])
    wh = (bottom_right - top_left + 1).clamp(min=0)
    # overlap
    overlap = wh[:,0] * wh[:,1]
    
    # calculate union
    area_pred = (pred[:, 2]-pred[:,0] + 1) * (pred[:,3] - pred[:,1]+1)
    area_target = (target[:, 2]-target[:, 0]+1) * (target[:,3]-target[:,1]+1)
    union = area_pred + area_target - overlap + eps
    # calculate iou
    ious = overlap/union

    # min. enclosing box
    enclose_x1y1 = torch.min(pred[:, :2], target[:, :2])
    enclose_x2y2 = torch.max(pred[:, 2:], target[:, 2:])
    enclose_wh = (enclose_x2y2 - enclose_x1y1 + 1).clamp(min=0)
    enclose_area = enclose_wh[:,0] * enclose_wh[: ,1] + eps
    

    # giou
    diff_term = (enclose_area - union) / enclose_area
    gious = ious - diff_term
    
    return gious

def compute_diou(pred, target, eps=1e-7):
    
    num_pred = pred.size(0)
    num_target = target.size(0)

    assert num_pred == num_target

    # calculate max-top-left and min-bottom-right points for overlap
    top_left = torch.max(pred[:,:2], target[:,:2])
    bottom_right = torch.min(pred[:, 2:], target[:, 2:])
    wh = (bottom_right - top_left + 1).clamp(min=0)
    # overlap
    overlap = wh[:,0] * wh[:,1]
    
    # get pred and gt centers
    pred_c_x = (pred[:, 0] + pred[:, 2]) / 2
    pred_c_y = (pred[:, 1] + pred[:, 3]) / 2
    gt_c_x = (target[:, 0] + target[:, 2]) / 2
    gt_c_y = (target[:, 1] + target[:, 3]) / 2

    # calculate union
    area_pred = (pred[:, 2]-pred[:,0] + 1) * (pred[:,3] - pred[:,1]+1)
    area_target = (target[:, 2]-target[:, 0]+1) * (target[:,3]-target[:,1]+1)
    union = area_pred + area_target - overlap + eps
    # calculate iou
    ious = overlap/union

    # min. enclosing box
    enclose_x1y1 = torch.min(pred[:, :2], target[:, :2])
    enclose_x2y2 = torch.max(pred[:, 2:], target[:, 2:])

    enclose_c = ((enclose_x2y2[:, 0] - enclose_x1y1[:, 0]) ** 2) + \
                ((enclose_x2y2[:, 1] - enclose_x1y1[:,1]) **2) + eps
    
    box_d = ((pred_c_x - gt_c_x) ** 2) + \
            ((pred_c_y - gt_c_y) ** 2)
    
    # diou
    diff_term = box_d / enclose_c

    dious = ious - diff_term
        
    return dious
