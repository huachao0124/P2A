from mmseg.registry import DATASETS, TRANSFORMS
from mmseg.datasets.basesegdataset import BaseSegDataset
from mmseg.datasets.cityscapes import CityscapesDataset

import mmcv
from mmcv.transforms.base import BaseTransform
import mmengine.fileio as fileio

import random
import os.path as osp
import pickle
import copy
import cv2
import numpy as np
from typing import Any, Callable, List, Optional, Sequence, Tuple, Union
from torchvision import transforms


@DATASETS.register_module()
class CityscapesWithAnomaliesDataset(CityscapesDataset):
    """Cityscapes dataset.

    The ``img_suffix`` is fixed to '_leftImg8bit.png' and ``seg_map_suffix`` is
    fixed to '_gtFine_labelTrainIds.png' for Cityscapes dataset.
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
                 num_anomalies = 100, 
                 img_size = (1024, 2048), 
                 **kwargs) -> None:
        self.num_anomalies = num_anomalies
        self.anomalies = [None for _ in range(num_anomalies)]
        super().__init__(**kwargs)
    
    def get_data_info(self, idx: int) -> dict:
        """Get annotation by index and automatically call ``full_init`` if the
        dataset has not been fully initialized.

        Args:
            idx (int): The index of data.

        Returns:
            dict: The idx-th annotation of the dataset.
        """
        if self.serialize_data:
            start_addr = 0 if idx == 0 else self.data_address[idx - 1].item()
            end_addr = self.data_address[idx].item()
            bytes = memoryview(
                self.data_bytes[start_addr:end_addr])  # type: ignore
            data_info = pickle.loads(bytes)  # type: ignore
        else:
            data_info = copy.deepcopy(self.data_list[idx])
        # Some codebase needs `sample_idx` of data information. Here we convert
        # the idx to a positive number and save it in data information.
        if idx >= 0:
            data_info['sample_idx'] = idx
        else:
            data_info['sample_idx'] = len(self) + idx
        # select random anomalies
        curr_num_anomalies = random.choices(range(4), weights=[2, 10, 5, 2], k=1)[0]
        selected_anomalies_indices = random.choices(range(self.num_anomalies), k=curr_num_anomalies)
        data_info['anomalies'] = []
        for idx in selected_anomalies_indices:
            with open(f'ldm/buffer/{idx}.pkl', 'rb') as f:
                data_info['anomalies'].append(pickle.load(f))
        # data_info['anomalies'] = self.anomalies[selected_anomalies_indices]
        
        return data_info
    

@TRANSFORMS.register_module()
class PasteAnomalies(BaseTransform):
    def __init__(self, rotate_prob=0.5, flip_prob=0.5, degree=(-20, 20)):
        self.rotate_prob = rotate_prob
        self.flip_prob = flip_prob
        assert 0 <= rotate_prob <= 1 and 0 <= flip_prob <= 1
        if isinstance(degree, (float, int)):
            assert degree > 0, f'degree {degree} should be positive'
            self.degree = (-degree, degree)
        else:
            self.degree = degree
        assert len(self.degree) == 2, f'degree {self.degree} should be a ' \
                                      f'tuple of (min, max)'
    
    def transform(self, results: dict) -> dict:
        
        for idx, anomaly in enumerate(results['anomalies']):
            k = np.random.randint(0, 4)
            image = np.rot90(anomaly['image'], k)
            mask = np.rot90(anomaly['mask'], k)
            
            axis = np.random.randint(0, 2)
            image = np.flip(image, axis=axis).copy()
            mask = np.flip(mask, axis=axis).copy()
            
            angle = np.random.uniform(min(*self.degree), max(*self.degree))
            image = mmcv.imrotate(image, angle=angle)
            mask = mmcv.imrotate(mask, angle=angle)
            
            long_side = random.randint(32, 1024 // 2 ** len(results['anomalies']))
            h, w = image.shape[:2]
            new_h, new_w = int(h / max(h, w) * long_side), int(w / max(h, w) * long_side)
            
            resize_image = cv2.resize(image, (new_w, new_h))
            resize_mask = cv2.resize(mask, (new_w, new_h))
            
            img_h, img_w = results['img'].shape[:2]
            x = random.randint(0, img_w - new_w)
            y = random.randint(448, img_h - new_h)
            results['img'][y: (y + new_h), x: (x + new_w)][resize_mask > 0] = resize_image[resize_mask > 0]
            results['gt_seg_map'][y: (y + new_h), x: (x + new_w)][resize_mask > 0] = 19
            
            from PIL import Image
            Image.fromarray(results['img']).save(f'samples/{idx}_0.jpg')
        
        return results
    
