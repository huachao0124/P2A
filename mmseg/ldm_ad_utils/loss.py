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
        assert ((ood_mask == 0) | (ood_mask == 1)).all()
        loss = torch.pow(id_score, 2).mean()
        if ood_mask.sum() > 0:
            loss = loss + torch.pow(torch.clamp(self.margin - ood_score, min=0.0), 2).mean()

        loss = self.loss_weight * loss

        return loss


@MODELS.register_module()
class ContrastiveLossCoco(nn.Module):

    def __init__(self,
                 reduction='mean',
                 margin=0.75,
                 loss_weight=1.0):

        super(ContrastiveLossCoco, self).__init__()
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

        ood_mask = torch.stack([gt_instance.ood_mask[0] for gt_instance in batch_gt_instances]).squeeze(1)        
                     
        mask_preds = F.interpolate(
            mask_preds, size=ood_mask.shape[-2:], mode='bilinear', align_corners=False)
        cls_scores = F.softmax(cls_scores, dim=-1)[..., :-1]
        mask_preds = mask_preds.sigmoid()
        seg_logits = torch.einsum('bqc, bqhw->bchw', cls_scores, mask_preds)
        
        score = -torch.max(seg_logits[:, :19], dim=1)[0]  
        ood_score = score[ood_mask == 1]
        id_score = score[ood_mask == 0]
        assert ((ood_mask == 0) | (ood_mask == 1)).all()
        
        loss = torch.pow(id_score, 2).mean()
        if ood_mask.sum() > 0:
            loss = loss + torch.pow(torch.clamp(self.margin - ood_score, min=0.0), 2).mean()

        loss = self.loss_weight * loss

        return loss


@MODELS.register_module()
class SegmentationLoss(nn.Module):

    def __init__(self,
                 reduction='mean',
                 loss_weight=1.0):

        super(SegmentationLoss, self).__init__()
        self.reduction = reduction
        self.loss_weight = loss_weight

        self.loss_fn = MODELS.build(dict(type='mmdet.CrossEntropyLoss',
                                        use_sigmoid=False,
                                        loss_weight=1.0,
                                        reduction=reduction,
                                        class_weight=[1.0, 1.0]))

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
            if (gt_instances.labels == 1).any():
                return gt_instances.masks[gt_instances.labels == 1].sum(dim=0)
            else:
                return gt_instances.masks.new_zeros(*gt_instances.masks.shape[-2:])
        
        ood_mask = torch.stack([get_ood_mask(gt_instances) for gt_instances in batch_gt_instances]).squeeze(1)
                
        mask_preds = F.interpolate(
            mask_preds, size=ood_mask.shape[-2:], mode='bilinear', align_corners=False)
        cls_scores = F.softmax(cls_scores, dim=-1)[..., :-1]
        mask_preds = mask_preds.sigmoid()
        seg_logits = torch.einsum('bqc, bqhw->bchw', cls_scores, mask_preds)
        pred_anomaly_maps = seg_logits[:, 1, :, :] - seg_logits[:, 0, :, :]
        

        if reduction == 'mean':
            # mean of empty tensor is nan
            # loss = (seg_logits[:, 0, :, :] * (ood_mask == 1) + seg_logits[:, 1, :, :] * (ood_mask == 0)).mean()
            # loss = seg_logits[:, 0, :, :][ood_mask == 1].mean() + seg_logits[:, 1, :, :][ood_mask == 0].mean()
            loss = torch.pow(torch.clamp(pred_anomaly_maps[ood_mask == 0], min=0.0), 2).mean()
            if (ood_mask == 1).any():
                loss = loss + torch.pow(torch.clamp(2 - pred_anomaly_maps[ood_mask == 1], min=0.0), 2).mean()
            # loss = torch.pow(torch.clamp(2 - seg_logits[:, 0, :, :][ood_mask == 0], min=0.0), 2).mean() + torch.pow(seg_logits[:, 1, :, :][ood_mask == 0], 2).mean()
            # if (ood_mask == 1).any():
            #     loss = loss + torch.pow(seg_logits[:, 0, :, :][ood_mask == 1], 2).mean() + torch.pow(torch.clamp(4 - seg_logits[:, 1, :, :][ood_mask == 1], min=0.0), 2).mean()
            # seg_logits_tanh = seg_logits.tanh()
            # loss = self.loss_fn(seg_logits_tanh, ood_mask)
        else:
            loss = seg_logits[:, 0, :, :][ood_mask == 1].sum() + seg_logits[:, 1, :, :][ood_mask == 0].sum()
        loss = self.loss_weight * loss

        return loss


@MODELS.register_module()
class RbALoss(nn.Module):

    def __init__(self,
                 reduction='mean',
                 loss_weight=1.0):

        super(RbALoss, self).__init__()
        self.reduction = reduction
        self.loss_weight = loss_weight

        self.loss_fn = MODELS.build(dict(type='mmdet.CrossEntropyLoss',
                                        use_sigmoid=False,
                                        loss_weight=1.0,
                                        reduction=reduction,
                                        class_weight=[1.0, 1.0]))

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
            if (gt_instances.labels >= 19).any():
                return gt_instances.masks[gt_instances.labels >= 19].sum(dim=0)
            else:
                return gt_instances.masks.new_zeros(*gt_instances.masks.shape[-2:])

        ood_mask = torch.stack([get_ood_mask(gt_instances) for gt_instances in batch_gt_instances])
                
        mask_preds = F.interpolate(
            mask_preds, size=ood_mask.shape[-2:], mode='bilinear', align_corners=False)
        cls_scores = F.softmax(cls_scores, dim=-1)[..., :-1]
        mask_preds = mask_preds.sigmoid()
        seg_logits = torch.einsum('bqc, bqhw->bchw', cls_scores, mask_preds)
        score = seg_logits.tanh()
        score = -score.sum(dim=1)
        
        ood_score = score[ood_mask == 1]
        id_score = score[ood_mask == 0]
        
        inlier_upper_threshold = 0
        outlier_lower_threshold = 5
        
        loss = torch.pow(F.relu(id_score - inlier_upper_threshold), 2).mean()
        if ood_mask.sum() > 0:
            loss = loss + torch.pow(F.relu(outlier_lower_threshold - ood_score), 2).mean()
        
        loss = self.loss_weight * loss

        return loss
