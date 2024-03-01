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


@MODELS.register_module()
class SplitSegmentationLoss(nn.Module):

    def __init__(self,
                 reduction='mean',
                 loss_weight=1.0):

        super(SplitSegmentationLoss, self).__init__()
        self.reduction = reduction
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
            if (gt_instances.labels == 1).any():
                return gt_instances.masks[gt_instances.labels == 1].sum(dim=0)
            else:
                return gt_instances.masks.new_zeros(*gt_instances.masks.shape[-2:])
        
        ood_mask = torch.stack([get_ood_mask(gt_instances) for gt_instances in batch_gt_instances]).squeeze(1)
                
        mask_preds = F.interpolate(
            mask_preds, size=ood_mask.shape[-2:], mode='bilinear', align_corners=False)
        cls_scores = F.softmax(cls_scores, dim=-1)[:, :, :-1]
        mask_preds = mask_preds.sigmoid()
        # seg_logits = torch.einsum('bqc, bqhw->bchw', cls_scores, mask_preds)
        cls_idx = torch.argmax(cls_scores, dim=-1)
        B, Q, H, W = mask_preds.shape
        weight_0 = cls_scores[:, :, 0]
        weight_1 = cls_scores[:, :, 1]
        mask_0 = (weight_0 > weight_1).float()
        mask_1 = (weight_0 < weight_1).float()
        seg_logits_0 = ((weight_0.unsqueeze(2).unsqueeze(2) * mask_preds) * (cls_idx == 0).unsqueeze(2).unsqueeze(2)).sum(dim=1)
        seg_logits_1 = ((weight_1.unsqueeze(2).unsqueeze(2) * mask_preds) * (cls_idx == 1).unsqueeze(2).unsqueeze(2)).sum(dim=1)

        # loss = torch.pow(torch.clamp(seg_logits_1[ood_mask == 0], min=0.0), 2).mean() + torch.pow(torch.clamp(2 - seg_logits_0[ood_mask == 0], min=0.0), 2).mean()
        # if (ood_mask == 1).any():
        #     loss = loss + torch.pow(torch.clamp(2 - seg_logits_1[ood_mask == 1], min=0.0), 2).mean() + torch.pow(torch.clamp(seg_logits_0[ood_mask == 1], min=0.0), 2).mean()
        
        pred_anomaly_maps = seg_logits_1 - seg_logits_0
        
        loss = torch.pow(torch.clamp(pred_anomaly_maps[ood_mask == 0], min=0.0), 2).mean()
        if (ood_mask == 1).any():
            loss = loss + torch.pow(torch.clamp(2 - pred_anomaly_maps[ood_mask == 1], min=0.0), 2).mean()

        loss = self.loss_weight * loss

        return loss


@MODELS.register_module()
class RefineLoss(nn.Module):

    def __init__(self,
                 reduction='mean',
                 loss_weight=1.0):

        super(RefineLoss, self).__init__()
        self.reduction = reduction
        self.loss_weight = loss_weight
        self.loss_fn = FocalLoss()

    def forward(self,
                pred_anomaly_score,
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
        
        ood_mask = torch.stack([get_ood_mask(gt_instances) for gt_instances in batch_gt_instances])
                
        # loss = ((pred_anomaly_score - ood_mask.unsqueeze(1)) ** 2).mean()
        loss = self.loss_fn(pred_anomaly_score, ood_mask)

        loss = self.loss_weight * loss

        return loss


class FocalLoss(nn.Module):
    def __init__(self, apply_nonlin=None, alpha=None, gamma=2, balance_index=0, smooth=1e-5, size_average=True):
        super(FocalLoss, self).__init__()
        self.apply_nonlin = apply_nonlin
        self.alpha = alpha
        self.gamma = gamma
        self.balance_index = balance_index
        self.smooth = smooth
        self.size_average = size_average

        if self.smooth is not None:
            if self.smooth < 0 or self.smooth > 1.0:
                raise ValueError('smooth value should be in [0,1]')

    def forward(self, logit, target):
        if self.apply_nonlin is not None:
            logit = self.apply_nonlin(logit)
        num_class = logit.shape[1]

        if logit.dim() > 2:
            # N,C,d1,d2 -> N,C,m (m=d1*d2*...)
            logit = logit.view(logit.size(0), logit.size(1), -1)
            logit = logit.permute(0, 2, 1).contiguous()
            logit = logit.view(-1, logit.size(-1))
        target = torch.squeeze(target, 1)
        target = target.view(-1, 1)
        alpha = self.alpha

        if alpha is None:
            alpha = torch.ones(num_class, 1)
        elif isinstance(alpha, (list, np.ndarray)):
            assert len(alpha) == num_class
            alpha = torch.FloatTensor(alpha).view(num_class, 1)
            alpha = alpha / alpha.sum()
        elif isinstance(alpha, float):
            alpha = torch.ones(num_class, 1)
            alpha = alpha * (1 - self.alpha)
            alpha[self.balance_index] = self.alpha

        else:
            raise TypeError('Not support alpha type')
        

        if alpha.device != logit.device:
            alpha = alpha.to(logit.device)

        idx = target
        
        one_hot_key = torch.zeros(target.size(0), num_class).to(logit.device)
        one_hot_key = one_hot_key.scatter_(1, idx, 1)
        

        if self.smooth:
            one_hot_key = torch.clamp(
                one_hot_key, self.smooth / (num_class - 1), 1.0 - self.smooth)
        pt = (one_hot_key * logit).sum(1) + self.smooth
        logpt = pt.log()

        gamma = self.gamma
        
        alpha = alpha[idx]
        alpha = torch.squeeze(alpha)
        loss = -1 * alpha * torch.pow((1 - pt), gamma) * logpt
        
        if self.size_average:
            loss = loss.mean()
        
        return loss
