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
import glob
from PIL import Image


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
        curr_num_anomalies = random.choices(range(6), weights=[2, 10, 5, 2, 1, 1], k=1)[0]
        selected_anomalies_indices = random.choices(range(results['num_anomalies']), k=curr_num_anomalies)
        results['anomalies'] = []
        for idx in selected_anomalies_indices:
            try:
                with open(f'{self.buffer_path}/{idx}.pkl', 'rb') as f:
                    results['anomalies'].append(pickle.load(f))
            except FileNotFoundError:
                print("before generation")
        
        for idx, anomaly in enumerate(results['anomalies']):
            # k = np.random.randint(0, 4)
            # image = np.rot90(anomaly['image'], k)
            # mask = np.rot90(anomaly['mask'], k)
            image = anomaly['image']
            mask = anomaly['mask']
            if random.random() < self.flip_prob:
                axis = np.random.randint(0, 2)
                image = np.flip(image, axis=1).copy()
                mask = np.flip(mask, axis=1).copy()
            
            # angle = np.random.uniform(min(*self.degree), max(*self.degree))
            # image = mmcv.imrotate(image, angle=angle)
            # mask = mmcv.imrotate(mask, angle=angle)
            
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
        
        r = random.randint(0, 100000)
        if r < 100:
            img = cv2.cvtColor(results['img'], cv2.COLOR_BGR2RGB)
            Image.fromarray(img).save(f'samples/{r}.jpg')
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


@TRANSFORMS.register_module()
class UnifyGT(BaseTransform):
    def __init__(self, label_map={0: 0, 1: 1, 255: 0}):
        super().__init__()
        self.label_map = label_map
    
    def transform(self, results: dict) -> dict:        
        new_gt_seg_map = np.zeros_like(results['gt_seg_map'])
        for k, v in self.label_map.items():
            new_gt_seg_map[results['gt_seg_map'] == k] = v
        results['gt_seg_map'] = new_gt_seg_map
        return results


@TRANSFORMS.register_module()
class PasteCocoObjects(BaseTransform):
    def mix_object(self, current_labeled_image, current_labeled_mask, cut_object_image, cut_object_mask):
        train_id_out = 254

        cut_object_mask[cut_object_mask == train_id_out] = 254

        mask = cut_object_mask == 254

        ood_mask = np.expand_dims(mask, axis=2)
        ood_boxes = extract_bboxes(ood_mask)
        ood_boxes = ood_boxes[0, :]  # (y1, x1, y2, x2)
        y1, x1, y2, x2 = ood_boxes[0], ood_boxes[1], ood_boxes[2], ood_boxes[3]
        cut_object_mask = cut_object_mask[y1:y2, x1:x2]
        cut_object_image = cut_object_image[y1:y2, x1:x2, :]

        mask = cut_object_mask == 254

        idx = np.transpose(np.repeat(np.expand_dims(cut_object_mask, axis=0), 3, axis=0), (1, 2, 0))

        if mask.shape[0] != 0:
            h_start_point = random.randint(0, current_labeled_mask.shape[0] - cut_object_mask.shape[0])
            h_end_point = h_start_point + cut_object_mask.shape[0]
            w_start_point = random.randint(0, current_labeled_mask.shape[1] - cut_object_mask.shape[1])
            w_end_point = w_start_point + cut_object_mask.shape[1]
        else:
            h_start_point = 0
            h_end_point = 0
            w_start_point = 0
            w_end_point = 0

        current_labeled_image[h_start_point:h_end_point, w_start_point:w_end_point, :][np.where(idx == 254)] = \
            cut_object_image[np.where(idx == 254)]
        current_labeled_mask[h_start_point:h_end_point, w_start_point:w_end_point][np.where(cut_object_mask == 254)] = \
            cut_object_mask[np.where(cut_object_mask == 254)]

        return current_labeled_image, current_labeled_mask
    
    def transform(self, results: dict) -> dict:
        
        if np.random.uniform() < results['anomaly_mix_ratio']:
            coco_gt_path = random.choice(results['anomalies'])
            coco_img_path = coco_gt_path.replace('annotations/ood_seg_train2017','images/train2017')
            coco_img_path = coco_img_path.replace('png','jpg')
            coco_img = cv2.imread(coco_img_path)
            coco_gt = cv2.imread(coco_gt_path)
            img, sem_seg_gt = self.mix_object(current_labeled_image=results['img'], \
                current_labeled_mask=results['gt_seg_map'], cut_object_image=coco_img, cut_object_mask=coco_gt)
            results['img'] = img
            results['gt_seg_map'] = sem_seg_gt
        
        return results


@DATASETS.register_module()
class CityscapesWithCocoDataset(CityscapesDataset):
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
                 anomaly_file_path, 
                 anomaly_mix_ratio = 0.2, 
                 img_size = (1024, 2048), 
                 **kwargs) -> None:
        super().__init__(**kwargs)
        self.anomaly_mix_ratio = anomaly_mix_ratio
        self.anomalies = glob.glob(anomaly_file_path)
    
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
        data_info['anomalies'] = self.anomalies
        data_info['anomaly_mix_ratio'] = self.anomaly_mix_ratio
        return data_info