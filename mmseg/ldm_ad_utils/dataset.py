from mmseg.registry import DATASETS, TRANSFORMS
from mmseg.datasets.basesegdataset import BaseSegDataset
from mmseg.datasets.cityscapes import CityscapesDataset

import mmcv
from mmcv.transforms.base import BaseTransform
import mmengine.fileio as fileio
from mmengine.dataset.base_dataset import Compose

import random
import os
import os.path as osp
import pickle
import copy
import cv2
import numpy as np
from typing import Any, Callable, List, Optional, Sequence, Tuple, Union
from torchvision import transforms
from torch.utils.data import Dataset
import json


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
                 num_classes = 20, 
                 img_size = (1024, 2048), 
                 **kwargs) -> None:
        self.num_anomalies = num_anomalies
        self.num_classes = num_classes
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
        # data_info['anomalies'] = self.anomalies[selected_anomalies_indices]
        data_info['num_anomalies'] = self.num_anomalies
        data_info['num_classes'] = self.num_classes
        return data_info
    

@TRANSFORMS.register_module()
class PasteAnomalies(BaseTransform):
    def __init__(self, 
                 buffer_path='ldm/buffer', 
                 part_instance=False, 
                 rotate_prob=0.5, 
                 flip_prob=0.5, 
                 degree=(-20, 20)):
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
        self.buffer_path = buffer_path
        self.part_instance = part_instance
    
    def transform(self, results: dict) -> dict:        
        # select random anomalies
        curr_num_anomalies = random.choices(range(10), weights=[2, 10, 5, 2, 1, 1, 1, 1, 1, 1], k=1)[0]
        selected_anomalies_indices = random.choices(range(results['num_anomalies']), k=curr_num_anomalies)
        results['anomalies'] = []
        for idx in selected_anomalies_indices:
            try:
                with open(f'{self.buffer_path}/{idx}.pkl', 'rb') as f:
                    results['anomalies'].append(pickle.load(f))
            except FileNotFoundError:
                print("before generation")
        
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
            
            long_side = random.randint(64, 2048 // min(2 ** len(results['anomalies']), 16))
            h, w = image.shape[:2]
            new_h, new_w = int(h / max(h, w) * long_side), int(w / max(h, w) * long_side)
            
            resize_image = cv2.resize(image, (new_w, new_h))
            resize_mask = cv2.resize(mask, (new_w, new_h))
            
            img_h, img_w = results['img'].shape[:2]
                        
            x = random.randint(0, img_w - new_w)
            try:
                y = random.randint(448, img_h - new_h)
            except:
                y = random.randint(0, img_h - new_h)
            results['img'][y: (y + new_h), x: (x + new_w)][resize_mask > 0] = resize_image[resize_mask > 0]
            if not self.part_instance:
                results['gt_seg_map'][y: (y + new_h), x: (x + new_w)][resize_mask > 0] = 19
            else:
                results['gt_seg_map'][y: (y + new_h), x: (x + new_w)][resize_mask > 0] = 19 + idx
        
        # from PIL import Image
        # Image.fromarray(results['img']).save(f'samples/{idx}_0.jpg')
        return results
    

@DATASETS.register_module()
class RoadAnomalyDataset(Dataset):
    def __init__(self, 
                 data_root: str = None, 
                 pipeline: List[Union[dict, Callable]] = [], 
                 **kwargs):
        with open(os.path.join(data_root, 'frame_list.json'), 'r') as f:
            self.img_list = json.load(f)
        self.data_root = data_root
        self.pipeline = Compose(pipeline)
    
    def __len__(self):
        return len(self.img_list)
    
    def __getitem__(self, idx):
        data_info = {'img_path': os.path.join(self.data_root, 'frames', self.img_list[idx])}
        data_info['reduce_zero_label'] = False
        data_info['seg_map_path'] = os.path.join(self.data_root, 'frames', \
                        self.img_list[idx].replace('jpg', 'labels'), 'labels_semantic.png')
        data_info['seg_fields'] = []
        
        data_info = self.pipeline(data_info)
        return data_info

@DATASETS.register_module()
class FSLostAndFoundDataset(BaseSegDataset):
    METAINFO = dict(
    classes=('normal', 'anomaly'),
    palette=[[0, 0, 0], [255, 0, 0]])
    
    def __init__(self,
                 img_suffix='.png',
                 seg_map_suffix='.png',
                 **kwargs) -> None:
        super().__init__(
            img_suffix=img_suffix, seg_map_suffix=seg_map_suffix, **kwargs)
