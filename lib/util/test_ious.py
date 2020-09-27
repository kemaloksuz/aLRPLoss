import numpy as np
from calc_iou import bbox_overlaps, compute_diou, compute_giou
from ious_orig import compute_diou as cdiou_orig
from ious_orig import compute_giou as cgiou_orig
import torch

if __name__ == '__main__':
    bbox_inside_weights = torch.tensor(np.asarray([[1.,1.,1.,1.]]))
    bbox_outside_weights = torch.tensor(np.asarray([[1.,1.,1.,1.]]))
    box1 = np.asarray([[-0.5,0.5,0.5,-0.5]])
    box1_ = np.asarray([[0.,0.,1.,1.]])
    box2 = np.asarray([[9.,11.,11.,9.]])
    box2_ = np.asarray([[10.,10.,1.,1.]])
    box1 = torch.tensor(box1)
    box1_ = torch.tensor(box1_)
    box2 = torch.tensor(box2)
    box2_ = torch.tensor(box2_)
    
    print('IoU: {}'.format(bbox_overlaps(box1,box2)))
    print('GIoU: {}, {}'.format(compute_giou(box1,box2), cgiou_orig(box1_, box2_, \
                                                                    bbox_inside_weights,\
                                                                    bbox_outside_weights)))
    print('DIoU: {}, {}'.format(compute_diou(box1,box2), cdiou_orig(box1_, box2_, \
                                                     bbox_inside_weights,\
                                                     bbox_outside_weights)))
