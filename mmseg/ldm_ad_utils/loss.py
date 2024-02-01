# Copyright (c) OpenMMLab. All rights reserved.
import numpy as np
import torch
import torch.nn as nn
from torch.nn import functional as F

from mmseg.registry import MODELS
from mmdet.models.losses.utils import weight_reduce_loss


@MODELS.register_module()
class ContrastiveLoss(nn.Module):

    def __init__(self,
                 reduction='mean',
                 margin=0.75,
                 loss_weight=1.0):

        super(ContrastiveLoss, self).__init__()
        self.reduction = reduction
        self.margin = margin
        self.loss_weight = loss_weight


    def forward(self,
                cls_scores, 
                mask_preds,
                batch_gt_instances,
                batch_img_metas,
                weight=None,
                reduction_override=None,
                avg_factor=None):
        

        assert reduction_override in (None, 'none', 'mean', 'sum')
        reduction = (
            reduction_override if reduction_override else self.reduction)

        def get_ood_mask(gt_instances):
            if (gt_instances.labels == 19).any():
                return gt_instances.masks[gt_instances.labels == 19]
            else:
                return gt_instances.masks.new_zeros((1, *gt_instances.masks.shape[-2:]))
        
        ood_mask = torch.stack([get_ood_mask(gt_instances) for gt_instances in batch_gt_instances]).squeeze(1)
                        
        mask_preds = F.interpolate(
            mask_preds, size=ood_mask.shape[-2:], mode='bilinear', align_corners=False)
        cls_scores = F.softmax(cls_scores, dim=-1)[..., :-1]
        mask_preds = mask_preds.sigmoid()
        seg_logits = torch.einsum('bqc, bqhw->bchw', cls_scores, mask_preds)
        
        score = -torch.max(seg_logits[:, :19], dim=1)[0]  
        ood_score = score[ood_mask == 1]
        id_score = score[ood_mask == 0]
        loss = torch.pow(id_score, 2).mean()
        if ood_mask.sum() > 0:
            loss = loss + torch.pow(torch.clamp(self.margin - ood_score, min=0.0), 2).mean()

        loss = self.loss_weight * loss

        return loss