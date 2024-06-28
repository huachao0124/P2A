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
            self.results.append((data_sample['seg_logits']['data'].cpu().numpy(), data_sample['gt_sem_seg']['data'].cpu().numpy()))
        
    
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
        logger: MMLogger = MMLogger.get_current_instance()
        if self.format_only:
            logger.info(f'results are saved to {osp.dirname(self.output_dir)}')
            return OrderedDict()
        
        results = tuple(zip(*results))
        assert len(results) == 2
        
        seg_logits = np.stack(results[0])
        gt_anomaly_maps = np.stack(results[1])
                
        has_anomaly = np.array([(1 in np.unique(gt_anomaly_map)) for gt_anomaly_map in gt_anomaly_maps]).astype(np.bool_)
        
        seg_logits = seg_logits[has_anomaly]
        gt_anomaly_maps = gt_anomaly_maps[has_anomaly].flatten()        
        
                
        # pred_anomaly_maps = seg_logits[:, 19, :, :].flatten()
        # pred_anomaly_maps = (1 - np.max(seg_logits[:, :19, :, :], axis=1)).flatten()
        # pred_anomaly_maps = seg_logits[:, 19, :, :].flatten() * (1 - np.max(seg_logits[:, :19, :, :], axis=1)).flatten()
        pred_anomaly_maps = seg_logits[:, -1, :, :].flatten() / np.max(seg_logits[:, :19, :, :], axis=1).flatten()
        
        assert ((gt_anomaly_maps == 0) | (gt_anomaly_maps == 1)).all()
        
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
        
        # summary
        metrics = dict()
        for key, val in zip(('AUPRC', 'FPR@95TPR', 'AUROC'), (prc_auc, fpr, roc_auc)):
            metrics[key] = np.round(val * 100, 2)
        metrics = OrderedDict(metrics)
        metrics.update({'Dataset': 'RoadAnomaly'})
        metrics.move_to_end('Dataset', last=False)
        class_table_data = PrettyTable()
        for key, val in metrics.items():
            class_table_data.add_column(key, [val])

        print_log('anomaly segmentation results:', logger)
        print_log('\n' + class_table_data.get_string(), logger=logger)

        return metrics
  

@METRICS.register_module()
class AnomalyMetricDoublePart(BaseMetric):
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
        seg_logits = []
        gt_anomaly_maps = []
        for data_sample in data_samples:
            self.results.append((data_sample['id_seg_logits']['data'].cpu().numpy(), data_sample['ood_seg_logits']['data'].cpu().numpy(), data_sample['gt_sem_seg']['data'].cpu().numpy()))
        
    
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
        logger: MMLogger = MMLogger.get_current_instance()
        if self.format_only:
            logger.info(f'results are saved to {osp.dirname(self.output_dir)}')
            return OrderedDict()
        
        results = tuple(zip(*results))
        assert len(results) == 3
        
        id_seg_logits = np.stack(results[0])
        ood_seg_logits = np.stack(results[1])
        gt_anomaly_maps = np.stack(results[2])
                
        has_anomaly = np.array([(1 in np.unique(gt_anomaly_map)) for gt_anomaly_map in gt_anomaly_maps]).astype(np.bool_)
        
        id_seg_logits = id_seg_logits[has_anomaly]
        ood_seg_logits = ood_seg_logits[has_anomaly]
        gt_anomaly_maps = gt_anomaly_maps[has_anomaly].flatten()        
                
        assert ((gt_anomaly_maps == 0) | (gt_anomaly_maps == 1)).all()
        
        ood_mask = (gt_anomaly_maps == 1)
        ind_mask = (gt_anomaly_maps == 0)
        
        AUPRCs = []
        FPRs = []
        AUROCs = []
        score_types = ['id', 'ood', 'ood * id', 'ood / id']
        
        for i in range(4):
            if i == 0:
                pred_anomaly_maps = (1 - np.max(id_seg_logits[:, :19, :, :], axis=1)).flatten()
            elif i == 1:
                pred_anomaly_maps = ood_seg_logits[:, 0, :, :].flatten()
            elif i == 2:
                pred_anomaly_maps = ood_seg_logits[:, 0, :, :].flatten() * (1 - np.max(id_seg_logits[:, :19, :, :], axis=1)).flatten()
            else:
                pred_anomaly_maps = ood_seg_logits[:, 0, :, :].flatten() / np.max(id_seg_logits[:, :19, :, :], axis=1).flatten()

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

            AUPRCs.append(prc_auc)
            FPRs.append(fpr)
            AUROCs.append(roc_auc)

        AUPRCs = np.array(AUPRCs)
        FPRs = np.array(FPRs)
        AUROCs = np.array(AUROCs)
        # summary
        metrics = dict()
        for key, val in zip(('AUPRC', 'FPR@95TPR', 'AUROC'), (AUPRCs, FPRs, AUROCs)):
            metrics[key] = np.round(val * 100, 2)
        metrics.update({'Score Type': score_types})
        metrics = OrderedDict(metrics)
        # metrics.update({'Dataset': 'RoadAnomaly'})
        metrics.move_to_end('Score Type', last=False)
        class_table_data = PrettyTable()
        for key, val in metrics.items():
            class_table_data.add_column(key, val)

        print_log('anomaly segmentation results:', logger)
        # print_log('\n' + class_table_data.get_string(), logger=logger)

        return metrics


@METRICS.register_module()
class AnomalyMetricP2A(BaseMetric):
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
        for data_sample in data_samples:
            self.results.append((data_sample['seg_logits']['data'].cpu().numpy(), data_sample['gt_sem_seg']['data'].cpu().numpy()))
        
    
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
        logger: MMLogger = MMLogger.get_current_instance()
        if self.format_only:
            logger.info(f'results are saved to {osp.dirname(self.output_dir)}')
            return OrderedDict()
        
        results = tuple(zip(*results))
        assert len(results) == 2
        
        seg_logits = np.stack(results[0])
        gt_anomaly_maps = np.stack(results[1])
        

        seg_logits_ind = seg_logits[:, :19, :, :]
        seg_logits_p2a = seg_logits[:, 19:, :, :]
        # seg_logits_p2a = np.tanh(seg_logits)
        # seg_logits_p2a = seg_logits
        
        has_anomaly = np.array([(1 in np.unique(gt_anomaly_map)) for gt_anomaly_map in gt_anomaly_maps]).astype(np.bool_)
        
        # seg_logits = seg_logits[has_anomaly]
        # gt_anomaly_maps = gt_anomaly_maps[has_anomaly]
        
        def normalize(p):
            assert len(p.shape) == 2
            return (p - p.min()) / (p.max() - p.min())
        pred_anomaly_maps = seg_logits_p2a[:, 1, :, :] - seg_logits_p2a[:, 0, :, :]
        # pred_anomaly_maps = -np.max(seg_logits_ind, axis=1)
        
        # positive_logits = seg_logits_p2a[:, 1, :, :]
        # negative_logits = -seg_logits_p2a[:, 0, :, :]
        # heat_maps = np.stack([normalize(p) for p in negative_logits])
        # positive_logits = positive_logits * heat_maps
        # pred_anomaly_maps = positive_logits + negative_logits
        
        
        # segmentation_map = np.argmax(seg_logits_ind, axis=1)
        # print(segmentation_map.shape)
        # # filter_map = (segmentation_map == 2) | (segmentation_map == 8) | (segmentation_map == 10)
        # filter_map = (segmentation_map == 1) | (segmentation_map == 2) | (segmentation_map == 3) | (segmentation_map == 4) | (segmentation_map == 8) | (segmentation_map == 9) | (segmentation_map == 10)
        # pred_anomaly_maps[filter_map] -= 50
        pred_anomaly_maps = pred_anomaly_maps.flatten()

        # pred_anomaly_maps = seg_logits[:, 1, :, :].flatten() - seg_logits[:, 0, :, :].flatten()
        # pred_anomaly_maps = seg_logits[:, 1, :, :].flatten()
        gt_anomaly_maps = gt_anomaly_maps.flatten()
        
        # assert ((gt_anomaly_maps == 0) | (gt_anomaly_maps == 1)).all()
        
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
        
        # summary
        metrics = dict()
        for key, val in zip(('AUPRC', 'FPR@95TPR', 'AUROC'), (prc_auc, fpr, roc_auc)):
            metrics[key] = np.round(val * 100, 2)
        metrics = OrderedDict(metrics)
        metrics.update({'Dataset': 'RoadAnomaly'})
        metrics.move_to_end('Dataset', last=False)
        class_table_data = PrettyTable()
        for key, val in metrics.items():
            class_table_data.add_column(key, [val])

        print_log('anomaly segmentation results:', logger)
        print_log('\n' + class_table_data.get_string(), logger=logger)

        return metrics


@METRICS.register_module()
class AnomalyMetricRbA(BaseMetric):
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
        for data_sample in data_samples:
            self.results.append((data_sample['seg_logits']['data'].cpu().numpy(), data_sample['gt_sem_seg']['data'].cpu().numpy()))
    
    def compute_metrics(self, results: list) -> Dict[str, float]:
        logger: MMLogger = MMLogger.get_current_instance()
        if self.format_only:
            logger.info(f'results are saved to {osp.dirname(self.output_dir)}')
            return OrderedDict()
        
        results = tuple(zip(*results))
        assert len(results) == 2
        
        seg_logits = np.stack(results[0])
        gt_anomaly_maps = np.stack(results[1])
                
        has_anomaly = np.array([(1 in np.unique(gt_anomaly_map)) for gt_anomaly_map in gt_anomaly_maps]).astype(np.bool_)
        
        # seg_logits = seg_logits[has_anomaly]
        # gt_anomaly_maps = gt_anomaly_maps[has_anomaly]    
        
                
        pred_anomaly_maps = -torch.from_numpy(seg_logits[:, :19]).tanh().sum(dim=1).numpy().flatten()
        gt_anomaly_maps = gt_anomaly_maps.flatten()
        
        # assert ((gt_anomaly_maps == 0) | (gt_anomaly_maps == 1)).all()
        
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
        
        # summary
        metrics = dict()
        for key, val in zip(('AUPRC', 'FPR@95TPR', 'AUROC'), (prc_auc, fpr, roc_auc)):
            metrics[key] = np.round(val * 100, 2)
        metrics = OrderedDict(metrics)
        metrics.update({'Dataset': 'FS_LF'})
        metrics.move_to_end('Dataset', last=False)
        class_table_data = PrettyTable()
        for key, val in metrics.items():
            class_table_data.add_column(key, [val])

        print_log('anomaly segmentation results:', logger)
        print_log('\n' + class_table_data.get_string(), logger=logger)

        return metrics



@METRICS.register_module()
class AnomalyMetricP2A4(BaseMetric):
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
        for data_sample in data_samples:
            self.results.append((data_sample['seg_logits']['data'].cpu().numpy(), data_sample['gt_sem_seg']['data'].cpu().numpy()))
        
    
    def compute_metrics(self, results: list) -> Dict[str, float]:
        logger: MMLogger = MMLogger.get_current_instance()
        if self.format_only:
            logger.info(f'results are saved to {osp.dirname(self.output_dir)}')
            return OrderedDict()
        
        results = tuple(zip(*results))
        assert len(results) == 2
        
        seg_logits = np.stack(results[0])
        gt_anomaly_maps = np.stack(results[1])
        
        # pred_anomaly_maps = seg_logits[:, 1, :, :].flatten() - seg_logits[:, 0, :, :].flatten()
        pred_anomaly_maps = seg_logits[:, -1, :, :].flatten()
        gt_anomaly_maps = gt_anomaly_maps.flatten()
        
        # assert ((gt_anomaly_maps == 0) | (gt_anomaly_maps == 1)).all()
        
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
        
        # summary
        metrics = dict()
        for key, val in zip(('AUPRC', 'FPR@95TPR', 'AUROC'), (prc_auc, fpr, roc_auc)):
            metrics[key] = np.round(val * 100, 2)
        metrics = OrderedDict(metrics)
        metrics.update({'Dataset': 'RoadAnomaly'})
        metrics.move_to_end('Dataset', last=False)
        class_table_data = PrettyTable()
        for key, val in metrics.items():
            class_table_data.add_column(key, [val])

        print_log('anomaly segmentation results:', logger)
        print_log('\n' + class_table_data.get_string(), logger=logger)

        return metrics


@METRICS.register_module()
class AnomalyMetricP2A5(BaseMetric):
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
        for data_sample in data_samples:
            self.results.append((data_sample['seg_logits']['data'].cpu().numpy(), data_sample['gt_sem_seg']['data'].cpu().numpy()))
        
    
    def compute_metrics(self, results: list) -> Dict[str, float]:
        logger: MMLogger = MMLogger.get_current_instance()
        if self.format_only:
            logger.info(f'results are saved to {osp.dirname(self.output_dir)}')
            return OrderedDict()
        
        results = tuple(zip(*results))
        assert len(results) == 2
        
        seg_logits = np.stack(results[0])
        gt_anomaly_maps = np.stack(results[1])
        
        print(seg_logits.shape)
        pred_anomaly_maps = seg_logits[:, -1, :, :].flatten()
        gt_anomaly_maps = gt_anomaly_maps.flatten()
        
        # assert ((gt_anomaly_maps == 0) | (gt_anomaly_maps == 1)).all()
        
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
        
        # summary
        metrics = dict()
        for key, val in zip(('AUPRC', 'FPR@95TPR', 'AUROC'), (prc_auc, fpr, roc_auc)):
            metrics[key] = np.round(val * 100, 2)
        metrics = OrderedDict(metrics)
        metrics.update({'Dataset': 'RoadAnomaly'})
        metrics.move_to_end('Dataset', last=False)
        class_table_data = PrettyTable()
        for key, val in metrics.items():
            class_table_data.add_column(key, [val])

        print_log('anomaly segmentation results:', logger)
        print_log('\n' + class_table_data.get_string(), logger=logger)

        return metrics


