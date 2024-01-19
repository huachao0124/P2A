import os
import mmengine
from mmseg.registry import HOOKS
from mmengine.hooks.hook import Hook
from mmengine.dist import get_dist_info, get_rank
import time
import numpy as np
import cv2
import random
from PIL import Image
import torch


@HOOKS.register_module()
class GeneratePseudoAnomalyHook(Hook):
    priority = 'VERY_LOW'
    
    def before_train(self, runner):
        with open('ldm/object365.txt', 'r') as f:
            content = f.readlines()
        self.objects = [eval(c)['name'] for c in content]
        
        rank, word_size = get_dist_info()
        interval = len(runner.model.anomalies) // word_size
        
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
        seed = int(time.time()) % 1000000
        eta = 1.0
        
        
        for idx in range(0, num_samples, 4):
            num_s = 4 if idx + 4 < num_samples else num_samples - idx
            cond = {"c_concat": None, "c_crossattn": [runner.model.ldm.model.get_learned_conditioning(p_prompts[idx: idx + num_s])]}
            un_cond = {"c_concat": None, "c_crossattn": [runner.model.ldm.model.get_learned_conditioning([n_prompt] * num_s)]}
            
            H, W = image_resolution, image_resolution
            shape = (4, H // 8, W // 8)
            runner.model.ldm.model.control_scales = [strength * (0.825 ** float(12 - i)) for i in range(13)] if guess_mode else ([strength] * 13)
            imgs, intermediates = runner.model.ldm.sample_create_image_mask(ddim_steps, num_s,
                                                            shape, cond, verbose=False, eta=eta,
                                                            unconditional_guidance_scale=scale,
                                                            unconditional_conditioning=un_cond, 
                                                            control_start_step=control_start_step, 
                                                            control_end_step=control_end_step, 
                                                            self_control=self_control)
            
            
            
            imgs = runner.model.ldm.model.decode_first_stage(imgs)
            B, C, H, W = imgs.shape
            
            imgs = ((imgs.permute(0, 2, 3, 1) + 1) / 2 * 255).cpu().numpy().clip(0, 255).astype(np.uint8)
            masks = (intermediates['pseudo_masks'].squeeze(1).cpu().numpy() * 255).astype(np.uint8)
            contours = intermediates['contours']
            
            for i, (img, mask, contour) in enumerate(zip(imgs, masks, contours)):
                x, y, w, h = cv2.boundingRect(contour)
                new_w, new_h = int(w / max(w, h) * W), int(h / max(w, h) * H)
                extracted_img = cv2.resize(img[y:y+h, x:x+w], (new_w, new_h))
                extracted_mask = cv2.resize(mask[y:y+h, x:x+w], (new_w, new_h))
                self.plot_mask_on_img(extracted_img, extracted_mask, rank * interval + idx * 4 + i)
                runner.model.anomalies[rank * interval + idx * 4 + i] = {'image': torch.from_numpy(extracted_img).permute(2, 0, 1), \
                                                                'mask': torch.from_numpy(extracted_mask)}
        
    
    def before_train_epoch(self, runner):
        rank, word_size = get_dist_info()
        interval = len(runner.model.anomalies) // word_size
        
        num_samples = 4
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
        seed = int(time.time()) % 1000000
        eta = 1.0
        
        cond = {"c_concat": None, "c_crossattn": [runner.model.ldm.model.get_learned_conditioning(p_prompts)]}
        un_cond = {"c_concat": None, "c_crossattn": [runner.model.ldm.model.get_learned_conditioning([n_prompt] * num_samples)]}
        
        H, W = image_resolution, image_resolution
        shape = (4, H // 8, W // 8)
        runner.model.ldm.model.control_scales = [strength * (0.825 ** float(12 - i)) for i in range(13)] if guess_mode else ([strength] * 13)
        imgs, intermediates = runner.model.ldm.sample_create_image_mask(ddim_steps, num_samples,
                                                        shape, cond, verbose=False, eta=eta,
                                                        unconditional_guidance_scale=scale,
                                                        unconditional_conditioning=un_cond, 
                                                        control_start_step=control_start_step, 
                                                        control_end_step=control_end_step, 
                                                        self_control=self_control)
        
        
        
        imgs = runner.model.ldm.model.decode_first_stage(imgs)
        B, C, H, W = imgs.shape
        
        imgs = ((imgs.permute(0, 2, 3, 1) + 1) / 2 * 255).cpu().numpy().clip(0, 255).astype(np.uint8)
        masks = (intermediates['pseudo_masks'].squeeze(1).cpu().numpy() * 255).astype(np.uint8)
        contours = intermediates['contours']
        replace_indices = random.sample(range(rank * interval, (rank + 1) * interval), k=num_samples)[0]
        for i, (img, mask, contour) in enumerate(zip(imgs, masks, contours)):
            x, y, w, h = cv2.boundingRect(contour)
            new_w, new_h = int(w / max(w, h) * W), int(h / max(w, h) * H)
            extracted_img = cv2.resize(img[y:y+h, x:x+w], (new_w, new_h))
            extracted_mask = cv2.resize(mask[y:y+h, x:x+w], (new_w, new_h))
            self.plot_mask_on_img(extracted_img, extracted_mask, replace_indices[i])
            runner.model.anomalies[replace_indices[i]] = {'image': torch.from_numpy(extracted_img).permute(2, 0, 1), \
                                                            'mask': torch.from_numpy(extracted_mask)}
    
    def plot_mask_on_img(self, img, mask, idx):
        red_mask = np.zeros_like(img)
        red_mask[:, :, :1][mask == 255] = 255  # 设置红色通道为1
        img[:, :, :1][mask == 255] = 0.5 * img[:, :, :1][mask == 255] + 0.5 * red_mask[:, :, :1][mask == 255]
        Image.fromarray(img).save(f'samples/{idx}.jpg')
        
