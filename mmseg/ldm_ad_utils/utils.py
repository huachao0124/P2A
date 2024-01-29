# Copyright (c) OpenMMLab. All rights reserved.
from typing import List, Optional, Union

import numpy as np
import torch
from mmengine import ConfigDict
from mmengine.structures import InstanceData
from scipy.optimize import linear_sum_assignment
from torch import Tensor

from mmseg.registry import TASK_UTILS
from mmdet.models.task_modules.assigners.assign_result import AssignResult
from mmseg.models.assigners.base_assigner import BaseAssigner
from mmdet.models.task_modules.assigners import HungarianAssigner


@TASK_UTILS.register_module()
class FixedAssigner(BaseAssigner):

    def __init__(self, **kwargs) -> None:
        super().__init__()

    def assign(self,
               pred_instances: InstanceData,
               gt_instances: InstanceData,
               img_meta: Optional[dict] = None,
               **kwargs) -> AssignResult:
        assert isinstance(gt_instances.labels, Tensor)
        num_gts, num_preds = len(gt_instances), len(pred_instances)
        gt_labels = gt_instances.labels
        device = gt_labels.device

        # 1. assign -1 by default
        assigned_gt_inds = torch.full((num_preds, ),
                                      -1,
                                      dtype=torch.long,
                                      device=device)
        assigned_labels = torch.full((num_preds, ),
                                     -1,
                                     dtype=torch.long,
                                     device=device)

        if num_gts == 0 or num_preds == 0:
            # No ground truth or boxes, return empty assignment
            if num_gts == 0:
                # No ground truth, assign all to background
                assigned_gt_inds[:] = 0
            return AssignResult(
                num_gts=num_gts,
                gt_inds=assigned_gt_inds,
                max_overlaps=None,
                labels=assigned_labels)

        matched_row_inds = []
        matched_col_inds = []
        for idx in range(num_preds):
            matched_idx = torch.nonzero(gt_labels == idx)
            if matched_idx.shape[0] > 0:
                matched_row_inds.append(idx)
                matched_col_inds.append(matched_idx.item())
        matched_row_inds = torch.tensor(matched_row_inds).to(device)
        matched_col_inds = torch.tensor(matched_col_inds).to(device)
        assert len(matched_row_inds) == min(num_gts, num_preds) and len(matched_col_inds) == min(num_gts, num_preds)
        
        # 4. assign backgrounds and foregrounds
        # assign all indices to backgrounds first
        assigned_gt_inds[:] = 0
        # assign foregrounds based on matching results
                
        assigned_gt_inds[matched_row_inds] = matched_col_inds + 1
        assigned_labels[matched_row_inds] = gt_labels[matched_col_inds]
        return AssignResult(
            num_gts=num_gts,
            gt_inds=assigned_gt_inds,
            max_overlaps=None,
            labels=assigned_labels)


@TASK_UTILS.register_module()
class GroupHungarianAssigner(HungarianAssigner):
    """Computes one-to-one matching between predictions and ground truth.

    This class computes an assignment between the targets and the predictions
    based on the costs. The costs are weighted sum of some components.
    For DETR the costs are weighted sum of classification cost, regression L1
    cost and regression iou cost. The targets don't include the no_object, so
    generally there are more predictions than targets. After the one-to-one
    matching, the un-matched are treated as backgrounds. Thus each query
    prediction will be assigned with `0` or a positive integer indicating the
    ground truth index:

    - 0: negative sample, no assigned gt
    - positive integer: positive sample, index (1-based) of assigned gt

    Args:
        match_costs (:obj:`ConfigDict` or dict or \
            List[Union[:obj:`ConfigDict`, dict]]): Match cost configs.
    """

    def assign(self,
               pred_instances: InstanceData,
               gt_instances: InstanceData,
               img_meta: Optional[dict] = None,
               **kwargs) -> AssignResult:
        assert isinstance(gt_instances.labels, Tensor)
        num_gts, num_preds = len(gt_instances), len(pred_instances)
        gt_labels = gt_instances.labels
        device = gt_labels.device

        # 1. assign -1 by default
        assigned_gt_inds = torch.full((num_preds, ),
                                      -1,
                                      dtype=torch.long,
                                      device=device)
        assigned_labels = torch.full((num_preds, ),
                                     -1,
                                     dtype=torch.long,
                                     device=device)

        if num_gts == 0 or num_preds == 0:
            # No ground truth or boxes, return empty assignment
            if num_gts == 0:
                # No ground truth, assign all to background
                assigned_gt_inds[:] = 0
            return AssignResult(
                num_gts=num_gts,
                gt_inds=assigned_gt_inds,
                max_overlaps=None,
                labels=assigned_labels)

        # assign all indices to backgrounds first
        assigned_gt_inds[:] = 0
        
        num_queries_per_class = num_preds // img_meta['num_classes']
        for class_idx in range(img_meta['num_classes']):
          
            if class_idx < img_meta['num_classes'] - 1:
                remain_gt_indices = (gt_labels == class_idx)
            else:
                remain_gt_indices = (gt_labels >= class_idx)
            if remain_gt_indices.sum() == 0:
                continue
            # 2. compute weighted cost
            remain_gt_instances = gt_instances[remain_gt_indices]
            remain_gt_instances.labels = class_idx * remain_gt_instances.labels.new_ones(\
                                                        remain_gt_instances.labels.shape)
            cost_list = []
            for match_cost in self.match_costs:
                cost = match_cost(
                    pred_instances=pred_instances[class_idx*num_queries_per_class: \
                                                (class_idx+1)*num_queries_per_class],
                    gt_instances=remain_gt_instances,
                    img_meta=img_meta)
                cost_list.append(cost)
            cost = torch.stack(cost_list).sum(dim=0)
            # 3. do Hungarian matching on CPU using linear_sum_assignment
            cost = cost.detach().cpu()
            if linear_sum_assignment is None:
                raise ImportError('Please run "pip install scipy" '
                                'to install scipy first.')

            
            matched_row_inds, matched_col_inds = linear_sum_assignment(cost)
            matched_row_inds += class_idx * num_queries_per_class
            matched_col_inds = np.nonzero(remain_gt_indices.cpu().numpy())[0][matched_col_inds]
            matched_row_inds = torch.from_numpy(matched_row_inds).to(device)
            matched_col_inds = torch.from_numpy(matched_col_inds).to(device)

            # 4. assign backgrounds and foregrounds
            # assign foregrounds based on matching results
            assigned_gt_inds[matched_row_inds] = matched_col_inds + 1
            assigned_labels[matched_row_inds] = gt_labels[matched_col_inds]
        for class_idx in range(img_meta['num_classes']):
            if class_idx < img_meta['num_classes'] - 1:
                assert ((assigned_labels[class_idx*num_queries_per_class:\
                                        (class_idx+1)*num_queries_per_class] == class_idx) | \
                        (assigned_labels[class_idx*num_queries_per_class:\
                                        (class_idx+1)*num_queries_per_class] == -1)).all()
            else:
                assert ((assigned_labels[class_idx*num_queries_per_class:\
                                        (class_idx+1)*num_queries_per_class] >= class_idx) | \
                        (assigned_labels[class_idx*num_queries_per_class:\
                                        (class_idx+1)*num_queries_per_class] == -1)).all()
            
        return AssignResult(
            num_gts=num_gts,
            gt_inds=assigned_gt_inds,
            max_overlaps=None,
            labels=assigned_labels)

