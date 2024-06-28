import mmengine
from mmengine.model import is_model_wrapper
from mmengine.hooks.hook import Hook
from mmengine.dist import get_dist_info, get_rank, is_distributed
from mmseg.registry import HOOKS

import os
import time
import pickle
import numpy as np
import cv2
import random
from PIL import Image
import torch
import threading
from tqdm import tqdm

import os.path as osp
import warnings
from typing import Optional, Sequence

import mmcv
import mmengine.fileio as fileio
from mmengine.runner import Runner
from mmengine.visualization import Visualizer
from mmseg.structures import SegDataSample


classes = ('road', 'sidewalk', 'building', 'wall', 'fence', 'pole',
            'traffic light', 'traffic sign', 'vegetation', 'terrain',
            'sky', 'person', 'rider', 'car', 'truck', 'bus', 'train',
            'motorcycle', 'bicycle', 'others')


@HOOKS.register_module()
class TextInitQueriesHook(Hook):
    priority = 'VERY_LOW'
    
    def before_run(self, runner):
        
        if is_model_wrapper(runner.model):
            model = runner.model.module
        else:
            model = runner.model
        
        text_embeddings = model.ldm.model.get_learned_conditioning([' '.join(classes)])[0][1:23]
        text_embeddings = torch.cat((text_embeddings[:6], torch.mean(text_embeddings[6:8], dim=0, keepdim=True), \
                                        torch.mean(text_embeddings[8:10], dim=0, keepdim=True), text_embeddings[10:]), dim=0)
        text_embeddings = text_embeddings.unsqueeze(1).repeat(1, model.decode_head.num_queries_per_class, 1).flatten(0, 1)
        text_embeddings = torch.cat((text_embeddings, text_embeddings[-1:].repeat(model.decode_head.num_queries - \
                        model.decode_head.num_classes * model.decode_head.num_queries_per_class, 1)))
        
        model.decode_head.text_embed = text_embeddings


@HOOKS.register_module()
class GeneratePseudoAnomalyHook(Hook):
    priority = 'VERY_LOW'
    
    def before_train(self, runner):
        # with open('ldm/object365.txt', 'r') as f:
        #     content = f.readlines()
        # self.objects = [eval(c)['name'] for c in content]
        
        with open('ldm/objects.txt', 'r') as f:
            self.objects = f.read().splitlines()
        
        if runner.easy_start:
            return
        
        rank, word_size = get_dist_info()
        if rank == 0 and not os.path.exists(runner.buffer_path):
            os.makedirs(runner.buffer_path)
            print("start generation")
            
        interval = runner.train_dataloader.dataset.num_anomalies // word_size
        
        num_samples = interval
        prompts = [['a', 'photo', 'of', 'a'] for _ in range(interval)]
        a_prompt = 'best quality'
        select_objects = random.choices(self.objects, k=num_samples)        
        p_prompts = [' '.join(s + [ob]) + ', ' + a_prompt for s, ob in zip(prompts, select_objects)]
        n_prompt = 'lowres, bad anatomy, bad hands, cropped, worst quality'
                
        image_resolution = 512
        detect_resolution = 512
        ddim_steps = 40
        control_start_step = 20
        control_end_step = 40
        guess_mode = False
        self_control = True
        strength = 1.4
        scale = 9.0
        seed = int(time.time() + rank) % 1000000
        eta = 1.0
        
        if is_model_wrapper(runner.model):
            model = runner.model.module
        else:
            model = runner.model
        
        for idx in tqdm(range(0, num_samples, 4)):
            num_s = 4 if idx + 4 < num_samples else num_samples - idx
            cond = {"c_concat": None, "c_crossattn": [model.ldm.model.get_learned_conditioning(p_prompts[idx: idx + num_s])]}
            un_cond = {"c_concat": None, "c_crossattn": [model.ldm.model.get_learned_conditioning([n_prompt] * num_s)]}
            
            H, W = image_resolution, image_resolution
            shape = (4, H // 8, W // 8)
            model.ldm.model.control_scales = [strength * (0.825 ** float(12 - i)) for i in range(13)] if guess_mode else ([strength] * 13)
            imgs, intermediates = model.ldm.sample_create_image_mask(ddim_steps, num_s,
                                                            shape, cond, verbose=False, eta=eta,
                                                            unconditional_guidance_scale=scale,
                                                            unconditional_conditioning=un_cond, 
                                                            control_start_step=control_start_step, 
                                                            control_end_step=control_end_step, 
                                                            self_control=self_control)
            
            imgs = model.ldm.model.decode_first_stage(imgs)
            B, C, H, W = imgs.shape
            
            imgs = ((imgs.permute(0, 2, 3, 1) + 1) / 2 * 255).cpu().numpy().clip(0, 255).astype(np.uint8)
            masks = (intermediates['pseudo_masks'].squeeze(1).cpu().numpy() * 255).astype(np.uint8)
            contours = intermediates['contours']
            
            for i, (img, mask, contour) in enumerate(zip(imgs, masks, contours)):
                x, y, w, h = cv2.boundingRect(contour)
                new_w, new_h = int(w / max(w, h) * W), int(h / max(w, h) * H)
                extracted_img = cv2.resize(img[y:y+h, x:x+w], (new_w, new_h))
                extracted_img = cv2.cvtColor(extracted_img, cv2.COLOR_RGB2BGR)
                extracted_mask = cv2.resize(mask[y:y+h, x:x+w], (new_w, new_h))
                # self.plot_mask_on_img(extracted_img, extracted_mask, rank * interval + idx + i)
                with open(f'{runner.buffer_path}/{rank * interval + idx + i}.pkl', 'wb') as f:
                    pickle.dump({'image': extracted_img, 'mask': extracted_mask}, f)
            
        torch.distributed.barrier()
        if rank == 0:
            print("finish generation")    
    
    def before_train_iter(self, runner, batch_idx, data_batch):
        if batch_idx != 0 and batch_idx % 1000 == 0:
            rank, word_size = get_dist_info()
            interval = runner.train_dataloader.dataset.num_anomalies // word_size
            
            num_samples = 16
            prompts = [['a', 'photo', 'of', 'a'] for _ in range(interval)]
            a_prompt = 'best quality'
            select_objects = random.choices(self.objects, k=num_samples)        
            p_prompts = [' '.join(s + [ob]) + ', ' + a_prompt for s, ob in zip(prompts, select_objects)]
            n_prompt = 'lowres, bad anatomy, bad hands, cropped, worst quality'
            
            image_resolution = 512
            detect_resolution = 512
            ddim_steps = 40
            control_start_step = 20
            control_end_step = 40
            guess_mode = False
            self_control = True
            strength = 1.4
            scale = 9.0
            seed = int(time.time() + rank) % 1000000
            eta = 1.0
            
            if is_distributed():
                model = runner.model.module
            else:
                model = runner.model
            
            for idx in range(0, num_samples, 4):
                num_s = 4 if idx + 4 < num_samples else num_samples - idx
                cond = {"c_concat": None, "c_crossattn": [model.ldm.model.get_learned_conditioning(p_prompts[idx: idx + num_s])]}
                un_cond = {"c_concat": None, "c_crossattn": [model.ldm.model.get_learned_conditioning([n_prompt] * num_s)]}
            
                H, W = image_resolution, image_resolution
                shape = (4, H // 8, W // 8)
                imgs, intermediates = model.ldm.sample_create_image_mask(ddim_steps, num_s,
                                                                shape, cond, verbose=False, eta=eta,
                                                                unconditional_guidance_scale=scale,
                                                                unconditional_conditioning=un_cond, 
                                                                control_start_step=control_start_step, 
                                                                control_end_step=control_end_step, 
                                                                self_control=self_control)
            
            
                imgs = model.ldm.model.decode_first_stage(imgs)
                B, C, H, W = imgs.shape
            
                imgs = ((imgs.permute(0, 2, 3, 1) + 1) / 2 * 255).cpu().numpy().clip(0, 255).astype(np.uint8)
                masks = (intermediates['pseudo_masks'].squeeze(1).cpu().numpy() * 255).astype(np.uint8)
                contours = intermediates['contours']
                replace_indices = random.sample(range(rank * interval, (rank + 1) * interval), k=num_s)
                for i, (img, mask, contour) in enumerate(zip(imgs, masks, contours)):
                    x, y, w, h = cv2.boundingRect(contour)
                    new_w, new_h = int(w / max(w, h) * W), int(h / max(w, h) * H)
                    extracted_img = cv2.resize(img[y:y+h, x:x+w], (new_w, new_h))
                    extracted_img = cv2.cvtColor(extracted_img, cv2.COLOR_RGB2BGR)
                    extracted_mask = cv2.resize(mask[y:y+h, x:x+w], (new_w, new_h))
                    # self.plot_mask_on_img(extracted_img, extracted_mask, replace_indices[i])
                    with open(f'{runner.buffer_path}/{replace_indices[i]}.pkl', 'wb') as f:
                        pickle.dump({'image': extracted_img, 'mask': extracted_mask}, f)
    
    
    def plot_mask_on_img(self, img, mask, idx):
        red_mask = np.zeros_like(img)
        red_mask[:, :, :1][mask == 255] = 255  # 设置红色通道为1
        red_mask_on_img = img.copy()
        red_mask_on_img[:, :, :1][mask == 255] = 0.5 * img[:, :, :1][mask == 255] + 0.5 * red_mask[:, :, :1][mask == 255]
        Image.fromarray(img).save(f'samples/{idx}.jpg')


from mmseg.visualization.local_visualizer import SegLocalVisualizer
from mmengine.dist import master_only
from mmengine.registry import VISBACKENDS, VISUALIZERS

@VISUALIZERS.register_module()
class VisualizerHeatMap(SegLocalVisualizer):
    @master_only
    def add_datasample(
            self,
            name: str,
            image: np.ndarray,
            data_sample: Optional[SegDataSample] = None,
            draw_gt: bool = True,
            draw_pred: bool = True,
            show: bool = False,
            wait_time: float = 0,
            # TODO: Supported in mmengine's Viusalizer.
            out_file: Optional[str] = None,
            step: int = 0,
            with_labels: Optional[bool] = True) -> None:
        classes = self.dataset_meta.get('classes', None)
        palette = self.dataset_meta.get('palette', None)

        gt_img_data = None
        pred_img_data = None

        if draw_gt and data_sample is not None:
            if 'gt_sem_seg' in data_sample:
                assert classes is not None, 'class information is ' \
                                            'not provided when ' \
                                            'visualizing semantic ' \
                                            'segmentation results.'
                gt_img_data = self._draw_sem_seg(image, data_sample.gt_sem_seg,
                                                 classes, palette, with_labels)

            if 'gt_depth_map' in data_sample:
                gt_img_data = gt_img_data if gt_img_data is not None else image
                gt_img_data = self._draw_depth_map(gt_img_data,
                                                   data_sample.gt_depth_map)

        if draw_pred and data_sample is not None:

            if 'pred_sem_seg' in data_sample:

                assert classes is not None, 'class information is ' \
                                            'not provided when ' \
                                            'visualizing semantic ' \
                                            'segmentation results.'
                pred_img_data = self._draw_sem_seg(image,
                                                   data_sample.pred_sem_seg,
                                                   classes, palette,
                                                   with_labels)

            if 'pred_depth_map' in data_sample:
                pred_img_data = pred_img_data if pred_img_data is not None \
                    else image
                pred_img_data = self._draw_depth_map(
                    pred_img_data, data_sample.pred_depth_map)

        if gt_img_data is not None and pred_img_data is not None:
            drawn_img = np.concatenate((gt_img_data, pred_img_data), axis=1)
        elif gt_img_data is not None:
            drawn_img = gt_img_data
        else:
            drawn_img = pred_img_data

        seg_logits_ind = data_sample.seg_logits.data[:19, :, :].cpu().numpy()
        seg_logits_p2a = data_sample.seg_logits.data[19:, :, :].cpu().numpy()

        # heatmap = (data_sample.seg_logits.data[1, :, :] - data_sample.seg_logits.data[0, :, :]).cpu().numpy()
        # heatmap = data_sample.seg_logits.data[1, :, :].cpu().numpy()
        # heatmap = -np.max(data_sample.seg_logits.data[:, :, :].cpu().numpy(), axis=0)

        # seg_logits = data_sample.seg_logits.data.cpu().numpy()
        # heatmap = -seg_logits[0, :, :]
        # heatmap = (heatmap - heatmap.min()) / (heatmap.max() - heatmap.min())
        # heatmap = seg_logits[1, :, :] * heatmap - seg_logits[0, :, :]

        # heatmap = seg_logits_p2a[1, :, :] - seg_logits_p2a[0, :, :]
        # segmentation_map = np.argmax(seg_logits_ind, axis=0)
        # filter_map = (segmentation_map == 2) | (segmentation_map == 8) | (segmentation_map == 10)
        # filter_map = (segmentation_map == 4) | (segmentation_map == 8) | (segmentation_map == 9) | (segmentation_map == 10) | (segmentation_map == 16)
        # heatmap = -np.max(seg_logits_ind.data.cpu().numpy(), axis=0)
        # heatmap = (heatmap - heatmap.min()) / (heatmap.max() - heatmap.min())
        # heatmap[filter_map] = 0 
        # heatmap = (heatmap * 255).astype(np.uint8)

        heatmap1 = seg_logits_p2a[1, :, :]
        heatmap2 = -seg_logits_p2a[0, :, :]
        heatmap3 = seg_logits_p2a[1, :, :] - seg_logits_p2a[0, :, :]
        heatmap1 = (heatmap1 - heatmap1.min()) / (heatmap1.max() - heatmap1.min())
        heatmap2 = (heatmap2 - heatmap2.min()) / (heatmap2.max() - heatmap2.min())
        heatmap3 = (heatmap3 - heatmap3.min()) / (heatmap3.max() - heatmap3.min())
        heatmap = np.concatenate((heatmap1, heatmap2, heatmap3), axis=1)
        heatmap = (heatmap * 255).astype(np.uint8)
        heatmap = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)[:, :, ::-1]
        drawn_img = np.concatenate((drawn_img, heatmap), axis=1)

        if show:
            self.show(drawn_img, win_name=name, wait_time=wait_time)

        if out_file is not None:
            mmcv.imwrite(mmcv.rgb2bgr(drawn_img), out_file)
        else:
            self.add_image(name, drawn_img, step)



@HOOKS.register_module()
class SegVisualizationWithResizeHook(Hook):
    """Segmentation Visualization Hook. Used to visualize validation and
    testing process prediction results.

    In the testing phase:

    1. If ``show`` is True, it means that only the prediction results are
        visualized without storing data, so ``vis_backends`` needs to
        be excluded.

    Args:
        draw (bool): whether to draw prediction results. If it is False,
            it means that no drawing will be done. Defaults to False.
        interval (int): The interval of visualization. Defaults to 50.
        show (bool): Whether to display the drawn image. Default to False.
        wait_time (float): The interval of show (s). Defaults to 0.
        backend_args (dict, Optional): Arguments to instantiate a file backend.
            See https://mmengine.readthedocs.io/en/latest/api/fileio.htm
            for details. Defaults to None.
            Notes: mmcv>=2.0.0rc4, mmengine>=0.2.0 required.
    """

    def __init__(self,
                 draw: bool = False,
                 interval: int = 50,
                 show: bool = False,
                 wait_time: float = 0.,
                 backend_args: Optional[dict] = None):
        self._visualizer: Visualizer = Visualizer.get_current_instance()
        self.interval = interval
        self.show = show
        if self.show:
            # No need to think about vis backends.
            self._visualizer._vis_backends = {}
            warnings.warn('The show is True, it means that only '
                          'the prediction results are visualized '
                          'without storing data, so vis_backends '
                          'needs to be excluded.')

        self.wait_time = wait_time
        self.backend_args = backend_args.copy() if backend_args else None
        self.draw = draw
        if not self.draw:
            warnings.warn('The draw is False, it means that the '
                          'hook for visualization will not take '
                          'effect. The results will NOT be '
                          'visualized or stored.')

    def _after_iter(self,
                    runner: Runner,
                    batch_idx: int,
                    data_batch: dict,
                    outputs: Sequence[SegDataSample],
                    mode: str = 'val') -> None:
        """Run after every ``self.interval`` validation iterations.

        Args:
            runner (:obj:`Runner`): The runner of the validation process.
            batch_idx (int): The index of the current batch in the val loop.
            data_batch (dict): Data from dataloader.
            outputs (Sequence[:obj:`SegDataSample`]): Outputs from model.
            mode (str): mode (str): Current mode of runner. Defaults to 'val'.
        """
        if self.draw is False or mode == 'train':
            return
        
        # heatmaps = []
        # for output in outputs:
        #     anomaly_score = (output.seg_logits.data[1, :, :] - output.seg_logits.data[0, :, :]).cpu().numpy()
        #     print(anomaly_score.max(), anomaly_score.min())

        if self.every_n_inner_iters(batch_idx, self.interval):
            for output in outputs:
                img_path = output.img_path
                img_bytes = fileio.get(
                    img_path, backend_args=self.backend_args)
                img = mmcv.imfrombytes(img_bytes, channel_order='rgb')
                window_name = f'{mode}_{osp.basename(img_path)}'
                img = mmcv.imresize(
                        img, 
                        (output.gt_sem_seg.shape[1], output.gt_sem_seg.shape[0]),
                        interpolation='nearest')
                
                                
                self._visualizer.add_datasample(
                    window_name,
                    img,
                    data_sample=output,
                    show=self.show,
                    wait_time=self.wait_time,
                    step=runner.iter)
