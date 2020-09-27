import torch
from ..util.calc_iou import calc_iou, bbox_overlaps, compute_giou, compute_diou
import numpy as np 
import pdb

class Assign_and_RegLoss():
    def __init__(self, image_per_gpu, fpn_anchor_num, num_classes, reg_loss, assigner, beta = 1.0):
        self.image_per_gpu = image_per_gpu
        self.output_num = int(fpn_anchor_num.sum()) 
        self.fpn_anchor_num = fpn_anchor_num
        self.num_classes = num_classes
        self.reg_loss = reg_loss
        self.assigner = assigner
        if reg_loss.type =='SmoothL1':
            self.losstype=0
        elif reg_loss.type =='IoULoss':
            self.losstype=1
        elif reg_loss.type =='GIoULoss':
            self.losstype=2            
        elif reg_loss.type =='aLRPLoss':    
            self.losstype=3
        # for aLRP Loss tau is 0.5
        self.tau = 0.50
        # Smooth L1 loss parameter
        self.beta = beta
        
    def xy_to_wh(self, bbox):
          
        w  = bbox[:, 2] - bbox[:, 0]+1.0
        h = bbox[:, 3] - bbox[:, 1]+1.0
        ctr_x   = bbox[:, 0] + 0.5 * (w-1.0)
        ctr_y   = bbox[:, 1] + 0.5 * (h-1.0)

        w  = torch.clamp(w, min=1)
        h = torch.clamp(h, min=1)    

        return torch.stack([ctr_x, ctr_y, w, h]).t()


    def delta2bbox(self, deltas, means=[0., 0., 0., 0.], stds=[0.1, 0.1, 0.2, 0.2], max_shape=None, wh_ratio_clip=16/1000):

        wx, wy, ww, wh = stds
        dx = deltas[:, 0] * wx
        dy = deltas[:, 1] * wy
        dw = deltas[:, 2] * ww
        dh = deltas[:, 3] * wh
        
        max_ratio = np.abs(np.log(wh_ratio_clip))

        dw = dw.clamp(min=-max_ratio, max=max_ratio)
        dh = dh.clamp(min=-max_ratio, max=max_ratio)

        pred_ctr_x = dx
        pred_ctr_y = dy
        pred_w = torch.exp(dw)
        pred_h = torch.exp(dh)

        x1 = pred_ctr_x - 0.5 * pred_w
        y1 = pred_ctr_y - 0.5 * pred_h
        x2 = pred_ctr_x + 0.5 * pred_w
        y2 = pred_ctr_y + 0.5 * pred_h
        
        return torch.stack([x1, y1, x2, y2], dim=-1)


    def bbox2delta(self, gt_boxes, anchor_boxes, means=[0., 0., 0., 0.], stds = [0.1, 0.1, 0.2, 0.2]):
        targets_dx = (gt_boxes[:,0] - anchor_boxes[:,0]) / anchor_boxes[:, 2]
        targets_dy = (gt_boxes[:,1] - anchor_boxes[:, 1]) / anchor_boxes[:, 3]
        targets_dw = torch.log(gt_boxes[:,2] / anchor_boxes[:,2])
        targets_dh = torch.log(gt_boxes[:,3] / anchor_boxes[:,3])
        
        targets2 = torch.stack((targets_dx, targets_dy, targets_dw, targets_dh)).t()
        means = targets2.new_tensor(means).unsqueeze(0)
        stds = targets2.new_tensor(stds).unsqueeze(0)

        targets2 = targets2.sub_(means).div_(stds)
        return targets2

    def compute(self, anchors, annotations, regressions): 
    	#Initialize data structures for assignment
        labels_b = torch.ones([self.image_per_gpu, self.output_num, self.num_classes]).cuda()*-1

    	#Initialize data structures for regression loss
        if self.losstype > 2:    
            regression_losses= torch.tensor([]).cuda()
        else:
            regression_losses = torch.zeros(self.image_per_gpu).cuda()

        anchor_boxes = self.xy_to_wh(anchors[0, :, :].type(torch.cuda.FloatTensor))
        p_num=0
        for j in range(self.image_per_gpu):
            
            bbox_annotation = annotations[j, :, :]
            bbox_annotation = bbox_annotation[bbox_annotation[:, 4] != -1]

            if bbox_annotation.shape[0] == 0:
                labels_b[j]=torch.zeros([self.output_num, self.num_classes]).cuda()
                continue

            IoU = calc_iou(anchors[0, :, :], bbox_annotation[:, :4]) # num_anchors x num_annotations
            if self.assigner['type'] == "IoUAssigner": 
                IoU_max, IoU_argmax = torch.max(IoU, dim=1) # num_anchors x 1

                ######
                gt_IoU_max, gt_IoU_argmax = torch.max(IoU, dim=0)
                gt_IoU_argmax=torch.where(IoU==gt_IoU_max)[0]
                positive_indices = torch.ge(torch.zeros(IoU_max.shape).cuda(),1)
                positive_indices[gt_IoU_argmax.long()] = True
                ######

                positive_indices = positive_indices | torch.ge(IoU_max, self.assigner['pos_min_IoU'])
                negative_indices = torch.lt(IoU_max, self.assigner['neg_max_IoU'])

                assigned_annotations = bbox_annotation[IoU_argmax, :]

                labels_b[j, negative_indices, :] = 0
                labels_b[j, positive_indices, :] = 0
                labels_b[j, positive_indices, assigned_annotations[positive_indices, 4].long()] = 1
            elif self.assigner['type'] == "ATSSAssigner":
                #1. compute center distance between all bbox and gt
                num_gt = bbox_annotation.shape[0]

                gt_cx = (bbox_annotation[:, 0] + bbox_annotation[:, 2]) / 2.0
                gt_cy = (bbox_annotation[:, 1] + bbox_annotation[:, 3]) / 2.0
                gt_points = torch.stack((gt_cx, gt_cy), dim=1)

                bboxes_cx = ((anchors[0, :, 0] + anchors[0, :, 2]) / 2.0).float()
                bboxes_cy = ((anchors[0, :, 1] + anchors[0, :, 3]) / 2.0).float()
                bboxes_points = torch.stack((bboxes_cx, bboxes_cy), dim=1)

                distances = (bboxes_points[:, None, :] -gt_points[None, :, :]).pow(2).sum(-1).sqrt()

                #2. on each pyramid level, for each gt, select k bbox whose center
                #are closest to the gt center, so we total select k*l bbox as
                #candidates for each gt
                candidate_idxs = []
                start_idx = 0
                for level, bboxes_per_level in enumerate(self.fpn_anchor_num):
                    # on each pyramid level, for each gt,
                    # select k bbox whose center are closest to the gt center
                    end_idx = int(start_idx + bboxes_per_level)
                    distances_per_level = distances[start_idx:end_idx, :]
                    selectable_k = min(self.assigner['topk'], int(bboxes_per_level))
                    _, topk_idxs_per_level = distances_per_level.topk(selectable_k, dim=0, largest=False)
                    candidate_idxs.append(topk_idxs_per_level + start_idx)
                    start_idx = end_idx
                candidate_idxs = torch.cat(candidate_idxs, dim=0)                


                #3. get corresponding iou for the these candidates, and compute the
                #mean and std, set mean + std as the iou threshold
                candidate_overlaps = IoU[candidate_idxs, torch.arange(num_gt)]
                overlaps_mean_per_gt = candidate_overlaps.mean(0)
                overlaps_std_per_gt = candidate_overlaps.std(0)
                overlaps_thr_per_gt = overlaps_mean_per_gt + overlaps_std_per_gt

                #4. select these candidates whose iou are greater than or equal to
                #the threshold as postive
                is_pos = candidate_overlaps >= overlaps_thr_per_gt[None, :]

                #5. limit the positive sample's center in gt
                for gt_idx in range(num_gt):
                    candidate_idxs[:, gt_idx] += gt_idx * self.output_num
                ep_bboxes_cx = bboxes_cx.view(1, -1).expand(num_gt, self.output_num).contiguous().view(-1)
                ep_bboxes_cy = bboxes_cy.view(1, -1).expand(num_gt, self.output_num).contiguous().view(-1)
                candidate_idxs = candidate_idxs.view(-1)

                # calculate the left, top, right, bottom distance between positive
                # bbox center and gt side
                l_ = ep_bboxes_cx[candidate_idxs].view(-1, num_gt) - bbox_annotation[:, 0]
                t_ = ep_bboxes_cy[candidate_idxs].view(-1, num_gt) - bbox_annotation[:, 1]
                r_ = bbox_annotation[:, 2] - ep_bboxes_cx[candidate_idxs].view(-1, num_gt)
                b_ = bbox_annotation[:, 3] - ep_bboxes_cy[candidate_idxs].view(-1, num_gt)
                is_in_gts = torch.stack([l_, t_, r_, b_], dim=1).min(dim=1)[0] > 0.01
                is_pos = is_pos & is_in_gts

                # if an anchor box is assigned to multiple gts,
                # the one with the highest IoU will be selected.
                IoU_inf = torch.full_like(IoU, -1).t().contiguous().view(-1)
                index = candidate_idxs.view(-1)[is_pos.view(-1)]
                IoU_inf[index] = IoU.t().contiguous().view(-1)[index]
                IoU_inf = IoU_inf.view(num_gt, -1).t()

                IoU_max, IoU_argmax = IoU_inf.max(dim=1)
                positive_indices = IoU_max > -1
                negative_indices = ~positive_indices

                assigned_annotations = bbox_annotation[IoU_argmax, :]

                labels_b[j, :, :] = 0
                labels_b[j, positive_indices, assigned_annotations[positive_indices, 4].long()] = 1

            #Regression Loss Computation Starts here
            pos_ex_num=positive_indices.sum()
            p_num+=pos_ex_num
            
            if pos_ex_num > 0:
                gt_boxes = self.xy_to_wh(assigned_annotations[positive_indices, :])
                targets2 = self.bbox2delta(gt_boxes, anchor_boxes[positive_indices, :])
                if self.losstype == 0:
                    regression_diff_abs= torch.abs(regressions[j, positive_indices, :]-targets2)

                    regression_loss = torch.where(
                        torch.le(regression_diff_abs, self.beta),
                        0.5 * torch.pow(regression_diff_abs, 2)/self.beta,
                        regression_diff_abs - 0.5 * self.beta
                    )
                    regression_losses[j]=regression_loss.sum()
                elif self.losstype == 1:
                    # convert targets and model outputs to boxes for IoU-Loss.
                    targets2_ = self.delta2bbox(targets2)
                    regression_ = self.delta2bbox(regressions[j, positive_indices, :])
                    # calculate bbox overlaps
                    ious = bbox_overlaps(regression_, targets2_)
                    regression_loss = 1 - ious
                    regression_losses[j]=regression_loss.sum()
                elif self.losstype == 2:
                    # convert targets and model outputs to boxes for IoU-Loss.
                    targets2_ = self.delta2bbox(targets2)
                    regression_ = self.delta2bbox(regressions[j, positive_indices, :])

                    # calculate bbox overlaps
                    ious = compute_giou(regression_, targets2_)
                    regression_loss = 1 - ious
                    regression_losses[j]=regression_loss.sum()
                else:
                    # convert targets and model outputs to boxes for IoU-Loss.
                    targets2_ = self.delta2bbox(targets2)
                    regression_ = self.delta2bbox(regressions[j, positive_indices, :])
                    # calculate bbox overlaps
                    if self.reg_loss.iou_type == 'IoU':
                        ious = (1-bbox_overlaps(regression_, targets2_))
                    elif self.reg_loss.iou_type =='GIoU':
                        ious = (1-compute_giou(regression_, targets2_)) / 2
                    #tau is set to 0.5 by default 
                    regression_losses=torch.cat([regression_losses, ((ious)/(1-self.tau))], dim=0)          
            else:
                if self.losstype <= 2:
                    regression_losses[j]=torch.tensor(0).float().cuda()
                else:
                    continue

  
        if self.losstype <= 2:
        	#Following AP Loss implementation, we normalize over the number of
        	#regression inputs once classical regression losses are adopted.
        	return labels_b, regression_losses.sum()/(4*p_num)
        else:
            return labels_b, regression_losses

        
