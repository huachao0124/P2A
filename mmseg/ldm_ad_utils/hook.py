import mmengine
from mmengine.model import is_model_wrapper
from mmengine.hooks.hook import Hook
from mmengine.dist import get_dist_info, get_rank, is_distributed
from mmseg.registry import HOOKS

import os
import time
import numpy as np
import cv2
import random
from PIL import Image
import torch


@HOOKS.register_module()
class GeneratePseudoAnomalyHook(Hook):
    priority = 'VERY_LOW'
    
    def before_build(self, runner):
        # for i in range(runner._train_dataloader.dataset.num_anomalies):
        #     img = cv2.imread(f'samples/images/{i}.jpg')
        #     mask = cv2.imread(f'samples/masks/{i}.jpg')
        #     mask = cv2.cvtColor(mask, cv2.COLOR_BGR2GRAY)
        #     runner._train_dataloader.dataset.anomalies[i] = {'image': img, 'mask': mask}
        
        with open('ldm/object365.txt', 'r') as f:
            content = f.readlines()
        self.objects = [eval(c)['name'] for c in content]
        
        rank, word_size = get_dist_info()
        interval = runner._train_dataloader.dataset.num_anomalies // word_size
        
        
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
        
        # if is_distributed():
        #     model = runner.model.module
        # else:
        #     model = runner.model
        
        for idx in range(0, num_samples, 4):
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
                self.plot_mask_on_img(extracted_img, extracted_mask, rank * interval + idx + i)
                runner._train_dataloader.dataset.anomalies[rank * interval + idx + i] = \
                                    {'image': extracted_img, \
                                    'mask': extracted_mask}
        
    
    # def before_train_iter(self, runner, **kwargs):
    #     for i in range(runner._train_dataloader.dataset.num_anomalies):
    #         print(i, runner._train_dataloader.dataset.anomalies[i] is None)
    
    
    # def before_train_epoch(self, runner):
    #     rank, word_size = get_dist_info()
    #     interval = runner.train_dataloader.dataset.num_anomalies // word_size
        
    #     num_samples = 4
    #     prompts = [['a', 'photo', 'of', 'a'] for _ in range(interval)]
    #     a_prompt = 'best quality'
    #     select_objects = random.choices(self.objects, k=num_samples)        
    #     p_prompts = [' '.join(s + [ob]) + ', ' + a_prompt for s, ob in zip(prompts, select_objects)]
    #     n_prompt = 'lowres, bad anatomy, bad hands, cropped, worst quality'
        
    #     image_resolution = 512
    #     detect_resolution = 512
    #     ddim_steps = 40
    #     control_start_step = 20
    #     control_end_step = 40
    #     guess_mode = False
    #     self_control = True
    #     strength = 1.4
    #     scale = 9.0
    #     seed = int(time.time()) % 1000000
    #     eta = 1.0
        
    #     if is_distributed():
    #         model = runner.model.module
    #     else:
    #         model = runner.model
        
    #     cond = {"c_concat": None, "c_crossattn": [model.ldm.model.get_learned_conditioning(p_prompts)]}
    #     un_cond = {"c_concat": None, "c_crossattn": [model.ldm.model.get_learned_conditioning([n_prompt] * num_samples)]}
        
    #     H, W = image_resolution, image_resolution
    #     shape = (4, H // 8, W // 8)
    #     imgs, intermediates = model.ldm.sample_create_image_mask(ddim_steps, num_samples,
    #                                                     shape, cond, verbose=False, eta=eta,
    #                                                     unconditional_guidance_scale=scale,
    #                                                     unconditional_conditioning=un_cond, 
    #                                                     control_start_step=control_start_step, 
    #                                                     control_end_step=control_end_step, 
    #                                                     self_control=self_control)
        
        
        
    #     imgs = model.ldm.model.decode_first_stage(imgs)
    #     B, C, H, W = imgs.shape
        
    #     imgs = ((imgs.permute(0, 2, 3, 1) + 1) / 2 * 255).cpu().numpy().clip(0, 255).astype(np.uint8)
    #     masks = (intermediates['pseudo_masks'].squeeze(1).cpu().numpy() * 255).astype(np.uint8)
    #     contours = intermediates['contours']
    #     replace_indices = random.sample(range(rank * interval, (rank + 1) * interval), k=num_samples)[0]
    #     for i, (img, mask, contour) in enumerate(zip(imgs, masks, contours)):
    #         x, y, w, h = cv2.boundingRect(contour)
    #         new_w, new_h = int(w / max(w, h) * W), int(h / max(w, h) * H)
    #         extracted_img = cv2.resize(img[y:y+h, x:x+w], (new_w, new_h))
    #         extracted_mask = cv2.resize(mask[y:y+h, x:x+w], (new_w, new_h))
    #         self.plot_mask_on_img(extracted_img, extracted_mask, replace_indices[i])
    #         runner.train_dataloader.dataset.anomalies[replace_indices[i]] = {'image': torch.from_numpy(extracted_img).permute(2, 0, 1), \
    #                                                         'mask': torch.from_numpy(extracted_mask)}
    
    def plot_mask_on_img(self, img, mask, idx):
        red_mask = np.zeros_like(img)
        red_mask[:, :, :1][mask == 255] = 255  # 设置红色通道为1
        red_mask_on_img = img.copy()
        red_mask_on_img[:, :, :1][mask == 255] = 0.5 * img[:, :, :1][mask == 255] + 0.5 * red_mask[:, :, :1][mask == 255]
        Image.fromarray(img).save(f'samples/{idx}.jpg')
        
