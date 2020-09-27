import torch.nn.functional as F

def FocalLoss(pred, target, gamma=2.0, alpha=0.25):
    #Discard unmatched anchors
    valid_labels=(target!=-1)

    #Get valid predictions, apply sigmoid
    pred = pred[valid_labels].sigmoid()

    #Get valid targets
    target = target[valid_labels].type_as(pred)
    
    #Compute 1-score considering the ground truth class (bg or fg)
    pt = (1 - pred) * target + pred * (1 - target)

    #Now using pt compute focal loss weight 
    focal_weight = (alpha * target + (1 - alpha) * (1 - target)) * pt.pow(gamma)
    
    #Just compute binary cross wntropy without reduction and multiply by focal weight
    loss = F.binary_cross_entropy(pred, target, reduction='none') * focal_weight

    #Normalize the total loss by number of positives
    if target.sum() > 0:
        return loss.sum()/target.sum()
    else:
        return torch.zeros(0).cuda()
