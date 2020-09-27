import torch
import pdb 
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

class aLRPLoss(torch.autograd.Function):
    @staticmethod
    def forward(ctx, logits, targets, losstype, regression_losses=None, delta=1., eps=1e-5):  
        classification_grads=torch.zeros(logits.shape).cuda()
        cls_loss=torch.zeros(0).cuda()
        if torch.max(targets)<=0:
            return cls_loss, None, None
        
        #Filter fg logits
        fg_labels=(targets==1)
        fg_logits=logits[fg_labels]
        fg_num=len(fg_logits)

        #Do not use bg with scores less than minimum fg logit
        #since changing its score does not have an effect on precision
        threshold_logit=torch.min(fg_logits)-delta

        #Get valid bg logits
        relevant_bg_labels=((targets==0)&(logits>=threshold_logit))
        relevant_bg_logits=logits[relevant_bg_labels] 
        relevant_bg_grad=torch.zeros(len(relevant_bg_logits)).cuda()
        rank=torch.zeros(fg_num).cuda()
        prec=torch.zeros(fg_num).cuda()
        fg_grad=torch.zeros(fg_num).cuda()
        
        max_prec=0                                           
        #sort the fg logits
        order=torch.argsort(fg_logits)
        #Loops over each positive following the order
        for ii in order:
            #x_ij s as score differences with fgs
            fg_relations=fg_logits-fg_logits[ii] 
            #Apply piecewise linear function and determine relations with fgs
            fg_relations=torch.clamp(fg_relations/(2*delta)+0.5,min=0,max=1)
            #Discard i=j in the summation in rank_pos
            fg_relations[ii]=0

            #x_ij s as score differences with bgs
            bg_relations=relevant_bg_logits-fg_logits[ii]
            #Apply piecewise linear function and determine relations with bgs
            bg_relations=torch.clamp(bg_relations/(2*delta)+0.5,min=0,max=1)

            #Compute the rank of the example within fgs and number of bgs with larger scores
            rank_pos=1+torch.sum(fg_relations)
            FP_num=torch.sum(bg_relations)
            #Store the total since it is normalizer also for aLRP Regression error
            rank[ii]=rank_pos+FP_num

            if losstype == 'AP': #Gradient and Cls Loss computation for AP Loss
                #Compute precision for this example 
                current_prec=rank_pos/rank[ii]
                #Compute interpolated AP and store gradients for relevant bg examples
                if (max_prec<=current_prec):
                    max_prec=current_prec
                    relevant_bg_grad += (bg_relations/rank[ii])
                else:
                    relevant_bg_grad += (bg_relations/rank[ii])*(((1-max_prec)/(1-current_prec)))
                #Store fg gradients
                fg_grad[ii]=-(1-max_prec)
                prec[ii]=max_prec                                
            elif losstype == 'aLRP': ##Gradient and Cls Loss computation for aLRP Loss
                #Compute precision for this example to compute classification loss 
                prec[ii]=rank_pos/rank[ii]                
                #For stability, set eps to a infinitesmall value (e.g. 1e-6), then compute grads
                if FP_num>eps:   
                    fg_grad[ii] = -(torch.sum(fg_relations*regression_losses)+FP_num)/rank[ii]
                    relevant_bg_grad += (bg_relations*(-fg_grad[ii]/FP_num))   
                    
        #aLRP with grad formulation fg gradient
        classification_grads[fg_labels]= fg_grad
        #aLRP with grad formulation bg gradient
        classification_grads[relevant_bg_labels]= relevant_bg_grad 
        #print("fg total grad=", '{:7.5f}'.format(classification_grads[labels_p].sum()/fg_num), "bg total grad=", '{:7.5f}'.format(classification_grads[valid_labels_n].sum()/fg_num))
 
        fg_num=torch.tensor(max(fg_num,1)).cuda()
        classification_grads /= (fg_num)
    
        cls_loss=1-prec.mean()
        ctx.save_for_backward(classification_grads)

        return cls_loss, rank, order

    @staticmethod
    def backward(ctx, out_grad1, out_grad2, out_grad3):
        g1, =ctx.saved_tensors
        return g1*out_grad1, None, None, None, None, None
