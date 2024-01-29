import os.path as osp
from collections import OrderedDict
from typing import Dict, List, Optional, Sequence

import numpy as np
import torch
from mmengine.dist import is_main_process
from mmengine.evaluator import BaseMetric
from mmengine.logging import MMLogger, print_log
from mmengine.utils import mkdir_or_exist
from PIL import Image
from prettytable import PrettyTable

from mmseg.registry import METRICS
from sklearn.metrics import roc_auc_score, roc_curve, auc, precision_recall_curve, average_precision_score
from ood_metrics import fpr_at_95_tpr, calc_metrics, plot_roc, plot_pr,plot_barcode

@METRICS.register_module()
class AnomalyMetric(BaseMetric):
    """IoU evaluation metric.

    Args:
        ignore_index (int): Index that will be ignored in evaluation.
            Default: 255.
        nan_to_num (int, optional): If specified, NaN values will be replaced
            by the numbers defined by the user. Default: None.
        collect_device (str): Device name used for collecting results from
            different ranks during distributed training. Must be 'cpu' or
            'gpu'. Defaults to 'cpu'.
        output_dir (str): The directory for output prediction. Defaults to
            None.
        format_only (bool): Only format result for results commit without
            perform evaluation. It is useful when you want to save the result
            to a specific format and submit it to the test server.
            Defaults to False.
        prefix (str, optional): The prefix that will be added in the metric
            names to disambiguate homonymous metrics of different evaluators.
            If prefix is not provided in the argument, self.default_prefix
            will be used instead. Defaults to None.
    """
    METAINFO = dict(
        classes=('road', 'sidewalk', 'building', 'wall', 'fence', 'pole',
                 'traffic light', 'traffic sign', 'vegetation', 'terrain',
                 'sky', 'person', 'rider', 'car', 'truck', 'bus', 'train',
                 'motorcycle', 'bicycle', 'anomaly'),
        palette=[[128, 64, 128], [244, 35, 232], [70, 70, 70], [102, 102, 156],
                 [190, 153, 153], [153, 153, 153], [250, 170, 30], [220, 220, 0],
                 [107, 142, 35], [152, 251, 152], [70, 130, 180],
                 [220, 20, 60], [255, 0, 0], [0, 0, 142], [0, 0, 70],
                 [0, 60, 100], [0, 80, 100], [0, 0, 230], [119, 11, 32], 
                 [0, 255, 0]])

    def __init__(self,
                 ignore_index: int = 255,
                 nan_to_num: Optional[int] = None,
                 collect_device: str = 'cpu',
                 output_dir: Optional[str] = None,
                 format_only: bool = False,
                 prefix: Optional[str] = None,
                 **kwargs) -> None:
        super().__init__(collect_device=collect_device, prefix=prefix)

        self.ignore_index = ignore_index
        self.nan_to_num = nan_to_num
        self.output_dir = output_dir
        if self.output_dir and is_main_process():
            mkdir_or_exist(self.output_dir)
        self.format_only = format_only

    def process(self, data_batch: dict, data_samples: Sequence[dict]) -> None:
        """Process one batch of data and data_samples.

        The processed results should be stored in ``self.results``, which will
        be used to compute the metrics when all batches have been processed.

        Args:
            data_batch (dict): A batch of data from the dataloader.
            data_samples (Sequence[dict]): A batch of outputs from the model.
        """
        seg_logits = []
        gt_anomaly_maps = []
        for data_sample in data_samples:
            seg_logits.append(data_sample['seg_logits']['data'])
            gt_anomaly_maps.append(data_sample['gt_sem_seg']['data'])
        
        self.results.append(seg_logits)
        self.results.append(gt_anomaly_maps)
            
    
    def compute_metrics(self, results: list) -> Dict[str, float]:
        """Compute the metrics from processed results.

        Args:
            results (list): The processed results of each batch.

        Returns:
            Dict[str, float]: The computed metrics. The keys are the names of
                the metrics, and the values are corresponding results. The key
                mainly includes aAcc, mIoU, mAcc, mDice, mFscore, mPrecision,
                mRecall.
        """
        seg_logits = self.results[0]
        gt_anomaly_maps = self.results[1]
        
        seg_logits = torch.stack(seg_logits)
        # pred_anomaly_maps = seg_logits[:, -1, :, :].flatten().cpu().numpy()
        # pred_anomaly_maps = 1 - torch.max(seg_logits[:, :19, :, :], dim=1)[0].flatten().cpu().numpy()
        # pred_anomaly_maps = seg_logits[:, -1, :, :].flatten().cpu().numpy() * (1 - torch.max(seg_logits[:, :19, :, :], dim=1)[0].flatten().cpu().numpy())
        pred_anomaly_maps = seg_logits[:, -1, :, :].flatten().cpu().numpy() / torch.max(seg_logits[:, :19, :, :], dim=1)[0].flatten().cpu().numpy()
        gt_anomaly_maps = torch.stack(gt_anomaly_maps).flatten().cpu().numpy()
        gt_anomaly_maps[gt_anomaly_maps > 0] = 1
        
        ood_mask = (gt_anomaly_maps == 1)
        ind_mask = (gt_anomaly_maps == 0)

        ood_out = pred_anomaly_maps[ood_mask]
        ind_out = pred_anomaly_maps[ind_mask]

        ood_label = np.ones(len(ood_out))
        ind_label = np.zeros(len(ind_out))
        
        val_out = np.concatenate((ind_out, ood_out))
        val_label = np.concatenate((ind_label, ood_label))

        fpr, tpr, _ = roc_curve(val_label, val_out)    
        roc_auc = auc(fpr, tpr)
        prc_auc = average_precision_score(val_label, val_out)
        fpr = fpr_at_95_tpr(val_out, val_label)
 
        metrics = {'fpr_at_95_tpr': fpr, 'AUPRC': prc_auc, 'AUROC': roc_auc}
        
        print(metrics)

        return metrics

