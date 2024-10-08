import logging
from typing import Dict, List, Tuple, Optional, Union

import torch
from torch import Tensor, nn
from torch.nn import functional as F

from mmcv.ops import point_sample
from mmengine.logging import print_log
from mmengine.model import BaseModule
from mmengine.structures import InstanceData

from mmseg.registry import MODELS
from mmseg.utils import (ConfigType, OptConfigType, OptMultiConfig,
                         OptSampleList, SampleList, add_prefix)
from mmseg.models.segmentors.encoder_decoder import EncoderDecoder
from mmseg.structures import SegDataSample
from mmseg.models.decode_heads.mask2former_head import Mask2FormerHead

from mmdet.models.dense_heads import \
        Mask2FormerHead as MMDET_Mask2FormerHead
from mmdet.utils import InstanceList, reduce_mean
from mmdet.models.utils import multi_apply, get_uncertain_point_coords_with_randomness

from abc import ABCMeta, abstractmethod
from typing import List, Tuple

from mmengine.model import BaseModel
from mmengine.structures import PixelData
from torch import Tensor

from mmseg.models.utils import resize
from einops import rearrange
import numpy as np

@MODELS.register_module()
class EncoderDecoderLDM(EncoderDecoder):
    def __init__(self,
                 backbone: ConfigType,
                 ldm: ConfigType, 
                 decode_head: ConfigType,
                 with_ldm: bool = False, 
                 with_ldm_as_backbone: bool = False, 
                 neck: OptConfigType = None,
                 auxiliary_head: OptConfigType = None,
                 train_cfg: OptConfigType = None,
                 test_cfg: OptConfigType = None,
                 data_preprocessor: OptConfigType = None,
                 pretrained: Optional[str] = None,
                 init_cfg: OptMultiConfig = None):
        super().__init__(
                 backbone,
                 decode_head,
                 neck,
                 auxiliary_head,
                 train_cfg,
                 test_cfg,
                 data_preprocessor,
                 pretrained,
                 init_cfg)
        
        if with_ldm:
            self.ldm = MODELS.build(ldm)
            self.freeze_ldm()
            
        self.with_ldm_as_backbone = with_ldm_as_backbone
    
    def freeze_ldm(self):
        for m in self.ldm.parameters():
            m.requires_grad = False
    
    
    def extract_feat(self, inputs: Tensor) -> List[Tensor]:
        if self.with_ldm_as_backbone:
            """Extract features from images."""
            r50_features = self.backbone(inputs)
            
            encoder_posterior = self.ldm.model.encode_first_stage(inputs)
            z = self.ldm.model.get_first_stage_encoding(encoder_posterior).detach()
            t = torch.full((z.shape[0], ), 50, device=inputs.device, dtype=torch.long)
            cond = {"c_concat": None, "c_crossattn": [self.ldm.model.get_learned_conditioning([""] * z.shape[0])]}
            all_sd_features = self.ldm.model.model.diffusion_model.forward_features(z, t, context=cond["c_crossattn"])
            out_indices = (11, 7, 4)
            sd_features = []
            for idx in out_indices:
                sd_features.append(all_sd_features[idx])
            
            out_features = [r50_features[0]]
            for layer in range(1, 4):
                out_features.append(torch.cat((r50_features[layer], sd_features[layer - 1]), dim=1))
            
            if self.with_neck:
                out_features = self.neck(out_features)
            
            return out_features
        else:
            x = self.backbone(inputs)
            if self.with_neck:
                x = self.neck(x)
            return x
    
    
    def postprocess_result(self,
                           seg_logits: Tensor,
                           data_samples: OptSampleList = None) -> SampleList:
        """ Convert results list to `SegDataSample`.
        Args:
            seg_logits (Tensor): The segmentation results, seg_logits from
                model of each input image.
            data_samples (list[:obj:`SegDataSample`]): The seg data samples.
                It usually includes information such as `metainfo` and
                `gt_sem_seg`. Default to None.
        Returns:
            list[:obj:`SegDataSample`]: Segmentation results of the
            input images. Each SegDataSample usually contain:

            - ``pred_sem_seg``(PixelData): Prediction of semantic segmentation.
            - ``seg_logits``(PixelData): Predicted logits of semantic
                segmentation before normalization.
        """
        batch_size, C, H, W = seg_logits.shape

        if data_samples is None:
            data_samples = [SegDataSample() for _ in range(batch_size)]
            only_prediction = True
        else:
            only_prediction = False

        for i in range(batch_size):
            if not only_prediction:
                img_meta = data_samples[i].metainfo
                # remove padding area
                if 'img_padding_size' not in img_meta:
                    padding_size = img_meta.get('padding_size', [0] * 4)
                else:
                    padding_size = img_meta['img_padding_size']
                padding_left, padding_right, padding_top, padding_bottom =\
                    padding_size
                # i_seg_logits shape is 1, C, H, W after remove padding
                i_seg_logits = seg_logits[i:i + 1, :,
                                          padding_top:H - padding_bottom,
                                          padding_left:W - padding_right]

                flip = img_meta.get('flip', None)
                if flip:
                    flip_direction = img_meta.get('flip_direction', None)
                    assert flip_direction in ['horizontal', 'vertical']
                    if flip_direction == 'horizontal':
                        i_seg_logits = i_seg_logits.flip(dims=(3, ))
                    else:
                        i_seg_logits = i_seg_logits.flip(dims=(2, ))

                # # resize as original shape
                # i_seg_logits = resize(
                #     i_seg_logits,
                #     size=img_meta['ori_shape'],
                #     mode='bilinear',
                #     align_corners=self.align_corners,
                #     warning=False).squeeze(0)
                i_seg_logits = i_seg_logits.squeeze(0)
            else:
                i_seg_logits = seg_logits[i]

            if C > 1:
                i_seg_pred = i_seg_logits.argmax(dim=0, keepdim=True)
            else:
                i_seg_logits = i_seg_logits.sigmoid()
                i_seg_pred = (i_seg_logits >
                              self.decode_head.threshold).to(i_seg_logits)
            data_samples[i].set_data({
                'seg_logits':
                PixelData(**{'data': i_seg_logits}),
                'pred_sem_seg':
                PixelData(**{'data': i_seg_pred})
            })

        return data_samples


@MODELS.register_module()
class EncoderDecoderLDMReshape(EncoderDecoderLDM):
    def extract_feat(self, inputs: Tensor) -> List[Tensor]:
        if self.with_ldm_as_backbone:
            """Extract features from images."""
            r50_features = self.backbone(inputs)
            
            B, C, H, W = inputs.shape
            nH, nW = H // 512, W // 512
            inputs = rearrange(inputs, 'B C (nH H) (nW W) -> (B nH nW) C H W', nH=nH, nW=nW, H=512, W=512)
            encoder_posterior = self.ldm.model.encode_first_stage(inputs)
            z = self.ldm.model.get_first_stage_encoding(encoder_posterior).detach()
            t = torch.full((z.shape[0], ), 50, device=inputs.device, dtype=torch.long)
            cond = {"c_concat": None, "c_crossattn": [self.ldm.model.get_learned_conditioning([""] * z.shape[0])]}
            all_sd_features = self.ldm.model.model.diffusion_model.forward_features(z, t, context=cond["c_crossattn"])
            out_indices = (11, 7, 4)
            sd_features = []
            for idx in out_indices:
                sd_features.append(rearrange(all_sd_features[idx], '(B nH nW) C H W -> B C (nH H) (nW W)', nH=nH, nW=nW))
            
            out_features = [r50_features[0]]
            for layer in range(1, 4):
                out_features.append(torch.cat((r50_features[layer], sd_features[layer - 1]), dim=1))
            
            if self.with_neck:
                out_features = self.neck(out_features)
            
            return out_features
        else:
            x = self.backbone(inputs)
            if self.with_neck:
                x = self.neck(x)
            return x


@MODELS.register_module()
class EncoderDecoderLDMP2A(EncoderDecoder):
    def __init__(self,
                 backbone: ConfigType,
                 ldm: ConfigType, 
                 decode_head: ConfigType,
                 with_ldm: bool = False, 
                 with_ldm_as_backbone: bool = False, 
                 neck: OptConfigType = None,
                 auxiliary_head: OptConfigType = None,
                 train_cfg: OptConfigType = None,
                 test_cfg: OptConfigType = None,
                 data_preprocessor: OptConfigType = None,
                 pretrained: Optional[str] = None,
                 init_cfg: OptMultiConfig = None):
        super().__init__(
                 backbone,
                 decode_head,
                 neck,
                 auxiliary_head,
                 train_cfg,
                 test_cfg,
                 data_preprocessor,
                 pretrained,
                 init_cfg)
        
        if with_ldm:
            self.ldm = MODELS.build(ldm)
            self.freeze_ldm()
        self.with_ldm_as_backbone = with_ldm_as_backbone
        
        self.freeze_model_but_p2a()
    
    def freeze_ldm(self):
        for m in self.ldm.parameters():
            m.requires_grad = False
            
    def freeze_model_but_p2a(self):
        for m in self.parameters():
            m.requires_grad = False
        
        for m in self.decode_head.query_feat_p2a.parameters():
            m.requires_grad = True
        for m in self.decode_head.query_embed_p2a.parameters():
            m.requires_grad = True
        for m in self.decode_head.cls_embed_p2a.parameters():
            m.requires_grad = True
        # for m in self.decode_head.transformer_decoder.parameters():
        #     m.requires_grad = True
    
    def extract_feat(self, inputs: Tensor) -> List[Tensor]:
        if self.with_ldm_as_backbone:
            """Extract features from images."""
            r50_features = self.backbone(inputs)
            
            encoder_posterior = self.ldm.model.encode_first_stage(inputs)
            z = self.ldm.model.get_first_stage_encoding(encoder_posterior).detach()
            t = torch.full((z.shape[0], ), 50, device=inputs.device, dtype=torch.long)
            cond = {"c_concat": None, "c_crossattn": [self.ldm.model.get_learned_conditioning([""] * z.shape[0])]}
            all_sd_features = self.ldm.model.model.diffusion_model.forward_features(z, t, context=cond["c_crossattn"])
            out_indices = (11, 7, 4)
            sd_features = []
            for idx in out_indices:
                sd_features.append(all_sd_features[idx])
            
            out_features = [r50_features[0]]
            for layer in range(1, 4):
                out_features.append(torch.cat((r50_features[layer], sd_features[layer - 1]), dim=1))
            
            if self.with_neck:
                out_features = self.neck(out_features)
            
            return out_features
        else:
            x = self.backbone(inputs)
            if self.with_neck:
                x = self.neck(x)
            return x
    
    
    def postprocess_result(self,
                           seg_logits: Tensor,
                           data_samples: OptSampleList = None) -> SampleList:
        """ Convert results list to `SegDataSample`.
        Args:
            seg_logits (Tensor): The segmentation results, seg_logits from
                model of each input image.
            data_samples (list[:obj:`SegDataSample`]): The seg data samples.
                It usually includes information such as `metainfo` and
                `gt_sem_seg`. Default to None.
        Returns:
            list[:obj:`SegDataSample`]: Segmentation results of the
            input images. Each SegDataSample usually contain:

            - ``pred_sem_seg``(PixelData): Prediction of semantic segmentation.
            - ``seg_logits``(PixelData): Predicted logits of semantic
                segmentation before normalization.
        """
        batch_size, C, H, W = seg_logits.shape

        if data_samples is None:
            data_samples = [SegDataSample() for _ in range(batch_size)]
            only_prediction = True
        else:
            only_prediction = False

        for i in range(batch_size):
            if not only_prediction:
                img_meta = data_samples[i].metainfo
                # remove padding area
                if 'img_padding_size' not in img_meta:
                    padding_size = img_meta.get('padding_size', [0] * 4)
                else:
                    padding_size = img_meta['img_padding_size']
                padding_left, padding_right, padding_top, padding_bottom =\
                    padding_size
                # i_seg_logits shape is 1, C, H, W after remove padding
                i_seg_logits = seg_logits[i:i + 1, :,
                                          padding_top:H - padding_bottom,
                                          padding_left:W - padding_right]

                flip = img_meta.get('flip', None)
                if flip:
                    flip_direction = img_meta.get('flip_direction', None)
                    assert flip_direction in ['horizontal', 'vertical']
                    if flip_direction == 'horizontal':
                        i_seg_logits = i_seg_logits.flip(dims=(3, ))
                    else:
                        i_seg_logits = i_seg_logits.flip(dims=(2, ))

                # # resize as original shape
                # i_seg_logits = resize(
                #     i_seg_logits,
                #     size=img_meta['ori_shape'],
                #     mode='bilinear',
                #     align_corners=self.align_corners,
                #     warning=False).squeeze(0)
                i_seg_logits = i_seg_logits.squeeze(0)
            else:
                i_seg_logits = seg_logits[i]

            if C > 1:
                i_seg_pred = i_seg_logits.argmax(dim=0, keepdim=True)
            else:
                i_seg_logits = i_seg_logits.sigmoid()
                i_seg_pred = (i_seg_logits >
                              self.decode_head.threshold).to(i_seg_logits)
            data_samples[i].set_data({
                'seg_logits':
                PixelData(**{'data': i_seg_logits}),
                'pred_sem_seg':
                PixelData(**{'data': i_seg_pred})
            })

        return data_samples


# 分2类
@MODELS.register_module()
class Mask2FormerHeadP2A(MMDET_Mask2FormerHead):
    def __init__(self,
                 num_classes,
                 align_corners=False,
                 ignore_index=255,
                 **kwargs):
        super().__init__(**kwargs)

        self.num_classes = num_classes
        self.align_corners = align_corners
        self.out_channels = num_classes
        self.ignore_index = ignore_index

        feat_channels = kwargs['feat_channels']
        self.cls_embed = nn.Linear(feat_channels, self.num_classes + 1)
        
        # modify
        self.query_feat_p2a = nn.Embedding.from_pretrained(\
                                torch.zeros_like(self.query_feat.weight), freeze=False)
        self.query_embed_p2a = nn.Embedding.from_pretrained(\
                                torch.zeros_like(self.query_embed.weight), freeze=False)
        self.cls_embed_p2a = nn.Linear(feat_channels, 2)
            

    def _seg_data_to_instance_data(self, batch_data_samples: SampleList):
        batch_img_metas = []
        batch_gt_instances = []

        for data_sample in batch_data_samples:
            batch_img_metas.append(data_sample.metainfo)
            gt_sem_seg = data_sample.gt_sem_seg.data
            classes = torch.unique(
                gt_sem_seg,
                sorted=False,
                return_inverse=False,
                return_counts=False)

            # remove ignored region
            gt_labels = classes[classes != self.ignore_index]
            # modify
            gt_labels = gt_labels[gt_labels >= self.num_classes]

            masks = []
            for class_id in gt_labels:
                masks.append(gt_sem_seg == class_id)

            if len(masks) == 0:
                gt_masks = torch.zeros(
                    (0, gt_sem_seg.shape[-2],
                     gt_sem_seg.shape[-1])).to(gt_sem_seg).long()
            else:
                gt_masks = torch.stack(masks).squeeze(1).long()
            
            # modify
            gt_labels = gt_labels.new_zeros(gt_labels.shape)
            instance_data = InstanceData(labels=gt_labels, masks=gt_masks)
            batch_gt_instances.append(instance_data)
        return batch_gt_instances, batch_img_metas

    def loss(self, x: Tuple[Tensor], batch_data_samples: SampleList,
             train_cfg: ConfigType) -> dict:
        # batch SegDataSample to InstanceDataSample
        batch_gt_instances, batch_img_metas = self._seg_data_to_instance_data(
            batch_data_samples)

        # forward
        all_cls_scores, all_mask_preds = self(x, batch_data_samples)

        # loss
        losses = self.loss_by_feat(all_cls_scores, all_mask_preds,
                                   batch_gt_instances, batch_img_metas)

        return losses

    def predict(self, x: Tuple[Tensor], batch_img_metas: List[dict],
                test_cfg: ConfigType) -> Tuple[Tensor]:
        """Test without augmentaton.

        Args:
            x (tuple[Tensor]): Multi-level features from the
                upstream network, each is a 4D-tensor.
            batch_img_metas (List[:obj:`SegDataSample`]): The Data
                Samples. It usually includes information such as
                `gt_sem_seg`.
            test_cfg (ConfigType): Test config.

        Returns:
            Tensor: A tensor of segmentation mask.
        """
        batch_data_samples = [
            SegDataSample(metainfo=metainfo) for metainfo in batch_img_metas
        ]

        all_cls_scores, all_mask_preds = self(x, batch_data_samples)
        mask_cls_results = all_cls_scores[-1]
        mask_pred_results = all_mask_preds[-1]
        if 'pad_shape' in batch_img_metas[0]:
            size = batch_img_metas[0]['pad_shape']
        else:
            size = batch_img_metas[0]['img_shape']
        # upsample mask
        mask_pred_results = F.interpolate(
            mask_pred_results, size=size, mode='bilinear', align_corners=False)
        # modify
        # cls_score = F.softmax(mask_cls_results, dim=-1)[..., :-1]
        cls_score = F.softmax(mask_cls_results, dim=-1)
        mask_pred = mask_pred_results.sigmoid()
        seg_logits = torch.einsum('bqc, bqhw->bchw', cls_score, mask_pred)
        return seg_logits
    
    def _forward_head(self, decoder_out: Tensor, mask_feature: Tensor,
                      attn_mask_target_size: Tuple[int, int]) -> Tuple[Tensor]:
        decoder_out = self.transformer_decoder.post_norm(decoder_out)
        # shape (num_queries, batch_size, c)
        # modify
        cls_pred = self.cls_embed(decoder_out)
        cls_pred_p2a = self.cls_embed_p2a(decoder_out)
        # shape (num_queries, batch_size, c)
        mask_embed = self.mask_embed(decoder_out)
        # shape (num_queries, batch_size, h, w)
        mask_pred = torch.einsum('bqc,bchw->bqhw', mask_embed, mask_feature)
        attn_mask = F.interpolate(
            mask_pred,
            attn_mask_target_size,
            mode='bilinear',
            align_corners=False)
        # shape (num_queries, batch_size, h, w) ->
        #   (batch_size * num_head, num_queries, h, w)
        attn_mask = attn_mask.flatten(2).unsqueeze(1).repeat(
            (1, self.num_heads, 1, 1)).flatten(0, 1)
        attn_mask = attn_mask.sigmoid() < 0.5
        attn_mask = attn_mask.detach()

        return cls_pred, cls_pred_p2a, mask_pred, attn_mask
    
    def forward(self, x: List[Tensor],
                batch_data_samples: SampleList) -> Tuple[List[Tensor]]:
        batch_size = x[0].shape[0]
        mask_features, multi_scale_memorys = self.pixel_decoder(x)
        # multi_scale_memorys (from low resolution to high resolution)
        decoder_inputs = []
        decoder_positional_encodings = []
        for i in range(self.num_transformer_feat_level):
            decoder_input = self.decoder_input_projs[i](multi_scale_memorys[i])
            # shape (batch_size, c, h, w) -> (batch_size, h*w, c)
            decoder_input = decoder_input.flatten(2).permute(0, 2, 1)
            level_embed = self.level_embed.weight[i].view(1, 1, -1)
            decoder_input = decoder_input + level_embed
            # shape (batch_size, c, h, w) -> (batch_size, h*w, c)
            mask = decoder_input.new_zeros(
                (batch_size, ) + multi_scale_memorys[i].shape[-2:],
                dtype=torch.bool)
            decoder_positional_encoding = self.decoder_positional_encoding(
                mask)
            decoder_positional_encoding = decoder_positional_encoding.flatten(
                2).permute(0, 2, 1)
            decoder_inputs.append(decoder_input)
            decoder_positional_encodings.append(decoder_positional_encoding)
        # shape (num_queries, c) -> (batch_size, num_queries, c)
        query_feat = (self.query_feat.weight + self.query_feat_p2a.weight).\
                        unsqueeze(0).repeat((batch_size, 1, 1))
        query_embed = (self.query_embed.weight + self.query_embed_p2a.weight).\
                        unsqueeze(0).repeat((batch_size, 1, 1))
        # query_feat = (self.query_feat.weight).\
        #                 unsqueeze(0).repeat((batch_size, 1, 1))
        # query_embed = (self.query_embed.weight).\
        #                 unsqueeze(0).repeat((batch_size, 1, 1))

        cls_pred_list = []
        cls_pred_p2a_list = []
        mask_pred_list = []
        cls_pred, mask_pred, attn_mask = self._forward_head(
            query_feat, mask_features, multi_scale_memorys[0].shape[-2:])
        cls_pred_list.append(cls_pred)
        mask_pred_list.append(mask_pred)

        for i in range(self.num_transformer_decoder_layers):
            level_idx = i % self.num_transformer_feat_level
            # if a mask is all True(all background), then set it all False.
            mask_sum = (attn_mask.sum(-1) != attn_mask.shape[-1]).unsqueeze(-1)
            attn_mask = attn_mask & mask_sum
            # cross_attn + self_attn
            layer = self.transformer_decoder.layers[i]
            query_feat = layer(
                query=query_feat,
                key=decoder_inputs[level_idx],
                value=decoder_inputs[level_idx],
                query_pos=query_embed,
                key_pos=decoder_positional_encodings[level_idx],
                cross_attn_mask=attn_mask,
                query_key_padding_mask=None,
                # here we do not apply masking on padded region
                key_padding_mask=None)
            cls_pred, cls_pred_p2a, mask_pred, attn_mask = self._forward_head(
                query_feat, mask_features, multi_scale_memorys[
                    (i + 1) % self.num_transformer_feat_level].shape[-2:])

            cls_pred_list.append(cls_pred)
            cls_pred_p2a_list.append(cls_pred_p2a)
            mask_pred_list.append(mask_pred)
        
        return cls_pred_list, cls_pred_p2a_list, mask_pred_list

    def _get_targets_single(self, cls_score: Tensor, mask_pred: Tensor,
                            gt_instances: InstanceData,
                            img_meta: dict) -> Tuple[Tensor]:
        """Compute classification and mask targets for one image.

        Args:
            cls_score (Tensor): Mask score logits from a single decoder layer
                for one image. Shape (num_queries, cls_out_channels).
            mask_pred (Tensor): Mask logits for a single decoder layer for one
                image. Shape (num_queries, h, w).
            gt_instances (:obj:`InstanceData`): It contains ``labels`` and
                ``masks``.
            img_meta (dict): Image informtation.

        Returns:
            tuple[Tensor]: A tuple containing the following for one image.

                - labels (Tensor): Labels of each image. \
                    shape (num_queries, ).
                - label_weights (Tensor): Label weights of each image. \
                    shape (num_queries, ).
                - mask_targets (Tensor): Mask targets of each image. \
                    shape (num_queries, h, w).
                - mask_weights (Tensor): Mask weights of each image. \
                    shape (num_queries, ).
                - pos_inds (Tensor): Sampled positive indices for each \
                    image.
                - neg_inds (Tensor): Sampled negative indices for each \
                    image.
                - sampling_result (:obj:`SamplingResult`): Sampling results.
        """
        gt_labels = gt_instances.labels
        gt_masks = gt_instances.masks
        # sample points
        num_queries = cls_score.shape[0]
        num_gts = gt_labels.shape[0]

        point_coords = torch.rand((1, self.num_points, 2),
                                  device=cls_score.device)
        # shape (num_queries, num_points)
        mask_points_pred = point_sample(
            mask_pred.unsqueeze(1), point_coords.repeat(num_queries, 1,
                                                        1)).squeeze(1)
        # shape (num_gts, num_points)
        gt_points_masks = point_sample(
            gt_masks.unsqueeze(1).float(), point_coords.repeat(num_gts, 1,
                                                               1)).squeeze(1)

        sampled_gt_instances = InstanceData(
            labels=gt_labels, masks=gt_points_masks)
        sampled_pred_instances = InstanceData(
            scores=cls_score, masks=mask_points_pred)
        # assign and sample
        assign_result = self.assigner.assign(
            pred_instances=sampled_pred_instances,
            gt_instances=sampled_gt_instances,
            img_meta=img_meta)
        pred_instances = InstanceData(scores=cls_score, masks=mask_pred)
        sampling_result = self.sampler.sample(
            assign_result=assign_result,
            pred_instances=pred_instances,
            gt_instances=gt_instances)
        pos_inds = sampling_result.pos_inds
        neg_inds = sampling_result.neg_inds

        # label target
        # modify
        labels = gt_labels.new_full((self.num_queries, ),
                                    1,
                                    dtype=torch.long)
        labels[pos_inds] = gt_labels[sampling_result.pos_assigned_gt_inds]
        label_weights = gt_labels.new_ones((self.num_queries, ))
        
        # mask target
        mask_targets = gt_masks[sampling_result.pos_assigned_gt_inds]
        mask_weights = mask_pred.new_zeros((self.num_queries, ))
        mask_weights[pos_inds] = 1.0

        return (labels, label_weights, mask_targets, mask_weights, pos_inds,
                neg_inds, sampling_result)
    
    def _loss_by_feat_single(self, cls_scores: Tensor, mask_preds: Tensor,
                             batch_gt_instances: List[InstanceData],
                             batch_img_metas: List[dict]) -> Tuple[Tensor]:
        """Loss function for outputs from a single decoder layer.

        Args:
            cls_scores (Tensor): Mask score logits from a single decoder layer
                for all images. Shape (batch_size, num_queries,
                cls_out_channels). Note `cls_out_channels` should includes
                background.
            mask_preds (Tensor): Mask logits for a pixel decoder for all
                images. Shape (batch_size, num_queries, h, w).
            batch_gt_instances (list[obj:`InstanceData`]): each contains
                ``labels`` and ``masks``.
            batch_img_metas (list[dict]): List of image meta information.

        Returns:
            tuple[Tensor]: Loss components for outputs from a single \
                decoder layer.
        """
        num_imgs = cls_scores.size(0)
        cls_scores_list = [cls_scores[i] for i in range(num_imgs)]
        mask_preds_list = [mask_preds[i] for i in range(num_imgs)]
        (labels_list, label_weights_list, mask_targets_list, mask_weights_list,
         avg_factor) = self.get_targets(cls_scores_list, mask_preds_list,
                                        batch_gt_instances, batch_img_metas)
        # shape (batch_size, num_queries)
        labels = torch.stack(labels_list, dim=0)
        # shape (batch_size, num_queries)
        label_weights = torch.stack(label_weights_list, dim=0)
        # shape (num_total_gts, h, w)
        mask_targets = torch.cat(mask_targets_list, dim=0)
        # shape (batch_size, num_queries)
        mask_weights = torch.stack(mask_weights_list, dim=0)
        
        # classfication loss
        # shape (batch_size * num_queries, )
        cls_scores = cls_scores.flatten(0, 1)
        labels = labels.flatten(0, 1)
        label_weights = label_weights.flatten(0, 1)

        class_weight = cls_scores.new_tensor(self.class_weight)
        
        
        loss_cls = self.loss_cls(
            cls_scores,
            labels,
            label_weights,
            avg_factor=class_weight[labels].sum())

        num_total_masks = reduce_mean(cls_scores.new_tensor([avg_factor]))
        num_total_masks = max(num_total_masks, 1)

        # extract positive ones
        # shape (batch_size, num_queries, h, w) -> (num_total_gts, h, w)
        mask_preds = mask_preds[mask_weights > 0]

        if mask_targets.shape[0] == 0:
            # zero match
            loss_dice = mask_preds.sum()
            loss_mask = mask_preds.sum()
            return loss_cls, loss_mask, loss_dice

        with torch.no_grad():
            points_coords = get_uncertain_point_coords_with_randomness(
                mask_preds.unsqueeze(1), None, self.num_points,
                self.oversample_ratio, self.importance_sample_ratio)
            # shape (num_total_gts, h, w) -> (num_total_gts, num_points)
            mask_point_targets = point_sample(
                mask_targets.unsqueeze(1).float(), points_coords).squeeze(1)
        # shape (num_queries, h, w) -> (num_queries, num_points)
        mask_point_preds = point_sample(
            mask_preds.unsqueeze(1), points_coords).squeeze(1)

        # dice loss
        loss_dice = self.loss_dice(
            mask_point_preds, mask_point_targets, avg_factor=num_total_masks)

        # mask loss
        # shape (num_queries, num_points) -> (num_queries * num_points, )
        mask_point_preds = mask_point_preds.reshape(-1)
        # shape (num_total_gts, num_points) -> (num_total_gts * num_points, )
        mask_point_targets = mask_point_targets.reshape(-1)
        loss_mask = self.loss_mask(
            mask_point_preds,
            mask_point_targets,
            avg_factor=num_total_masks * self.num_points)
        
        return loss_cls, loss_mask, loss_dice


@MODELS.register_module()
class EncoderDecoderLDMP2A2(EncoderDecoder):
    def __init__(self,
                 backbone: ConfigType,
                 ldm: ConfigType, 
                 decode_head: ConfigType,
                 with_ldm: bool = False, 
                 with_ldm_as_backbone: bool = False, 
                 neck: OptConfigType = None,
                 auxiliary_head: OptConfigType = None,
                 train_cfg: OptConfigType = None,
                 test_cfg: OptConfigType = None,
                 data_preprocessor: OptConfigType = None,
                 pretrained: Optional[str] = None,
                 init_cfg: OptMultiConfig = None):
        super().__init__(
                 backbone,
                 decode_head,
                 neck,
                 auxiliary_head,
                 train_cfg,
                 test_cfg,
                 data_preprocessor,
                 pretrained,
                 init_cfg)
        
        if with_ldm:
            self.ldm = MODELS.build(ldm)
            self.freeze_ldm()
        self.with_ldm_as_backbone = with_ldm_as_backbone
        
        self.freeze_model_but_p2a()
    
    def freeze_ldm(self):
        for m in self.ldm.parameters():
            m.requires_grad = False
            
    def freeze_model_but_p2a(self):
        for m in self.parameters():
            m.requires_grad = False
        
        for m in self.decode_head.parameters():
            m.requires_grad = True
        
        # for m in self.decode_head.query_feat_p2a.parameters():
        #     m.requires_grad = True
        # for m in self.decode_head.query_embed_p2a.parameters():
        #     m.requires_grad = True
        # for m in self.decode_head.cls_embed_p2a.parameters():
        #     m.requires_grad = True
        # for m in self.decode_head.mask_embed.parameters():
        #     m.requires_grad = True
        
        # for m in self.decode_head.transformer_decoder.parameters():
        #     m.requires_grad = True
        for m in self.decode_head.cls_embed.parameters():
            m.requires_grad = False
    
    def extract_feat(self, inputs: Tensor) -> List[Tensor]:
        if self.with_ldm_as_backbone:
            """Extract features from images."""
            r50_features = self.backbone(inputs)
            
            encoder_posterior = self.ldm.model.encode_first_stage(inputs)
            z = self.ldm.model.get_first_stage_encoding(encoder_posterior).detach()
            t = torch.full((z.shape[0], ), 50, device=inputs.device, dtype=torch.long)
            cond = {"c_concat": None, "c_crossattn": [self.ldm.model.get_learned_conditioning([""] * z.shape[0])]}
            all_sd_features = self.ldm.model.model.diffusion_model.forward_features(z, t, context=cond["c_crossattn"])
            out_indices = (11, 7, 4)
            sd_features = []
            for idx in out_indices:
                sd_features.append(all_sd_features[idx])
                
            out_features = [r50_features[0]]
            for layer in range(1, 4):
                out_features.append(torch.cat((r50_features[layer], sd_features[layer - 1]), dim=1))
            
            if self.with_neck:
                out_features = self.neck(out_features)
            
            return out_features
        else:
            x = self.backbone(inputs)
            if self.with_neck:
                x = self.neck(x)
            return x
    
    
    def postprocess_result(self,
                           seg_logits: Tensor,
                           data_samples: OptSampleList = None) -> SampleList:
        """ Convert results list to `SegDataSample`.
        Args:
            seg_logits (Tensor): The segmentation results, seg_logits from
                model of each input image.
            data_samples (list[:obj:`SegDataSample`]): The seg data samples.
                It usually includes information such as `metainfo` and
                `gt_sem_seg`. Default to None.
        Returns:
            list[:obj:`SegDataSample`]: Segmentation results of the
            input images. Each SegDataSample usually contain:

            - ``pred_sem_seg``(PixelData): Prediction of semantic segmentation.
            - ``seg_logits``(PixelData): Predicted logits of semantic
                segmentation before normalization.
        """
        batch_size, C, H, W = seg_logits.shape

        if data_samples is None:
            data_samples = [SegDataSample() for _ in range(batch_size)]
            only_prediction = True
        else:
            only_prediction = False

        for i in range(batch_size):
            if not only_prediction:
                img_meta = data_samples[i].metainfo
                # remove padding area
                if 'img_padding_size' not in img_meta:
                    padding_size = img_meta.get('padding_size', [0] * 4)
                else:
                    padding_size = img_meta['img_padding_size']
                padding_left, padding_right, padding_top, padding_bottom =\
                    padding_size
                # i_seg_logits shape is 1, C, H, W after remove padding
                i_seg_logits = seg_logits[i:i + 1, :,
                                          padding_top:H - padding_bottom,
                                          padding_left:W - padding_right]

                flip = img_meta.get('flip', None)
                if flip:
                    flip_direction = img_meta.get('flip_direction', None)
                    assert flip_direction in ['horizontal', 'vertical']
                    if flip_direction == 'horizontal':
                        i_seg_logits = i_seg_logits.flip(dims=(3, ))
                    else:
                        i_seg_logits = i_seg_logits.flip(dims=(2, ))

                # # resize as original shape
                # i_seg_logits = resize(
                #     i_seg_logits,
                #     size=img_meta['ori_shape'],
                #     mode='bilinear',
                #     align_corners=self.align_corners,
                #     warning=False).squeeze(0)
                i_seg_logits = i_seg_logits.squeeze(0)
            else:
                i_seg_logits = seg_logits[i]

            if C > 1:
                i_seg_pred = i_seg_logits[:19].argmax(dim=0, keepdim=True)
            else:
                i_seg_logits = i_seg_logits.sigmoid()
                i_seg_pred = (i_seg_logits >
                              self.decode_head.threshold).to(i_seg_logits)
            

            data_samples[i].set_data({
                'seg_logits':
                PixelData(**{'data': i_seg_logits}),
                'pred_sem_seg':
                PixelData(**{'data': i_seg_pred})
            })

        return data_samples

    def predict(self,
                inputs: Tensor,
                data_samples: OptSampleList = None) -> SampleList:
        if data_samples is not None:
            batch_img_metas = [
                data_sample.metainfo for data_sample in data_samples
            ]
        else:
            batch_img_metas = [
                dict(
                    ori_shape=inputs.shape[2:],
                    img_shape=inputs.shape[2:],
                    pad_shape=inputs.shape[2:],
                    padding_size=[0, 0, 0, 0])
            ] * inputs.shape[0]

        seg_logits = self.inference(inputs, batch_img_metas)
        assert seg_logits.shape[0] == 1
        # import os
        # for seg_logit, img_meta in zip(seg_logits, batch_img_metas):
        #     seg_logit = resize(
        #         seg_logit.unsqueeze(0),
        #         size=img_meta['ori_shape'],
        #         mode='bilinear',
        #         align_corners=self.align_corners,
        #         warning=False).squeeze(0)
        #     np.save(os.path.join('road-anomaly-benchmark/predictions/ObstacleTrack', img_meta['img_path'].split('/')[-1][:-5]), seg_logit.detach().cpu().numpy())

        return self.postprocess_result(seg_logits, data_samples)    


@MODELS.register_module()
class EncoderDecoderLDMP2A2Reshape(EncoderDecoderLDMP2A2):    
    def extract_feat(self, inputs: Tensor) -> List[Tensor]:
        if self.with_ldm_as_backbone:
            """Extract features from images."""
            r50_features = self.backbone(inputs)
            
            B, C, H, W = inputs.shape
            nH, nW = H // 512, W // 512
            inputs = rearrange(inputs, 'B C (nH H) (nW W) -> (B nH nW) C H W', nH=nH, nW=nW, H=512, W=512)
            encoder_posterior = self.ldm.model.encode_first_stage(inputs)
            z = self.ldm.model.get_first_stage_encoding(encoder_posterior).detach()
            t = torch.full((z.shape[0], ), 50, device=inputs.device, dtype=torch.long)
            cond = {"c_concat": None, "c_crossattn": [self.ldm.model.get_learned_conditioning([""] * z.shape[0])]}
            all_sd_features = self.ldm.model.model.diffusion_model.forward_features(z, t, context=cond["c_crossattn"])
            out_indices = (11, 7, 4)
            sd_features = []
            for idx in out_indices:
                sd_features.append(rearrange(all_sd_features[idx], '(B nH nW) C H W -> B C (nH H) (nW W)', nH=nH, nW=nW))
            
            out_features = [r50_features[0]]
            for layer in range(1, 4):
                out_features.append(torch.cat((r50_features[layer], sd_features[layer - 1]), dim=1))
            
            if self.with_neck:
                out_features = self.neck(out_features)
            
            return out_features
        else:
            x = self.backbone(inputs)
            if self.with_neck:
                x = self.neck(x)
            return x


# 分3类，正常，异常，和void
@MODELS.register_module()
class Mask2FormerHeadP2A2(Mask2FormerHeadP2A):
    def __init__(self, loss_seg=None, use_attn_mask=True, **kwargs):
        super().__init__(**kwargs)

        feat_channels = kwargs['feat_channels']
        self.loss_seg = None
        if loss_seg is not None:
            self.loss_seg = MODELS.build(loss_seg)
        self.cls_embed_p2a = nn.Linear(feat_channels, 3)    
        self.use_attn_mask = use_attn_mask

    def _seg_data_to_instance_data(self, batch_data_samples: SampleList):
        """Perform forward propagation to convert paradigm from MMSegmentation
        to MMDetection to ensure ``MMDET_Mask2FormerHead`` could be called
        normally. Specifically, ``batch_gt_instances`` would be added.

        Args:
            batch_data_samples (List[:obj:`SegDataSample`]): The Data
                Samples. It usually includes information such as
                `gt_sem_seg`.

        Returns:
            tuple[Tensor]: A tuple contains two lists.

                - batch_gt_instances (list[:obj:`InstanceData`]): Batch of
                    gt_instance. It usually includes ``labels``, each is
                    unique ground truth label id of images, with
                    shape (num_gt, ) and ``masks``, each is ground truth
                    masks of each instances of a image, shape (num_gt, h, w).
                - batch_img_metas (list[dict]): List of image meta information.
        """
        batch_img_metas = []
        batch_gt_instances = []

        for data_sample in batch_data_samples:
            batch_img_metas.append(data_sample.metainfo)
            gt_sem_seg = data_sample.gt_sem_seg.data
            classes = torch.unique(
                gt_sem_seg,
                sorted=False,
                return_inverse=False,
                return_counts=False)

            # remove ignored region
            gt_labels = classes[classes != self.ignore_index]
            # modify
            # gt_labels = gt_labels[gt_labels >= self.num_classes]
            # labels of i.d. are 0，labels of o.o.d are 1，作为一个2(3)分类任务
            masks = []
            new_gt_labels = []
            for class_id in gt_labels:
                masks.append(gt_sem_seg == class_id)
                if class_id < self.num_classes:
                    new_gt_labels.append(0)
                else:
                    new_gt_labels.append(1)

            if len(masks) == 0:
                gt_masks = torch.zeros(
                    (0, gt_sem_seg.shape[-2],
                     gt_sem_seg.shape[-1])).to(gt_sem_seg).long()
            else:
                gt_masks = torch.stack(masks).squeeze(1).long()
            
            # modify
            gt_labels = torch.tensor(new_gt_labels).to(gt_labels)
            instance_data = InstanceData(labels=gt_labels, masks=gt_masks)
            batch_gt_instances.append(instance_data)
        return batch_gt_instances, batch_img_metas
    
    def predict(self, x: Tuple[Tensor], batch_img_metas: List[dict],
                test_cfg: ConfigType) -> Tuple[Tensor]:
        batch_data_samples = [
            SegDataSample(metainfo=metainfo) for metainfo in batch_img_metas
        ]

        all_cls_scores, all_cls_scores_p2a, all_mask_preds = self(x, batch_data_samples)
        mask_cls_results = all_cls_scores[-1]
        mask_cls_results_p2a = all_cls_scores_p2a[-1]
        mask_pred_results = all_mask_preds[-1]
        if 'pad_shape' in batch_img_metas[0]:
            size = batch_img_metas[0]['pad_shape']
        else:
            size = batch_img_metas[0]['img_shape']
        # upsample mask
        mask_pred_results = F.interpolate(
            mask_pred_results, size=size, mode='bilinear', align_corners=False)
        # modify
        cls_score = F.softmax(mask_cls_results, dim=-1)[:, :, :-1]
        cls_score_p2a = F.softmax(mask_cls_results_p2a, dim=-1)[:, :, :-1]
        mask_pred = mask_pred_results.sigmoid()
        seg_logits = torch.einsum('bqc, bqhw->bchw', cls_score, mask_pred)
        seg_logits_p2a = torch.einsum('bqc, bqhw->bchw', cls_score_p2a, mask_pred)

        seg_logits_combined = torch.cat((seg_logits, seg_logits_p2a), dim=1)
        
        return seg_logits_combined
    
    def _get_targets_single(self, cls_score: Tensor, mask_pred: Tensor,
                            gt_instances: InstanceData,
                            img_meta: dict) -> Tuple[Tensor]:
        """Compute classification and mask targets for one image.

        Args:
            cls_score (Tensor): Mask score logits from a single decoder layer
                for one image. Shape (num_queries, cls_out_channels).
            mask_pred (Tensor): Mask logits for a single decoder layer for one
                image. Shape (num_queries, h, w).
            gt_instances (:obj:`InstanceData`): It contains ``labels`` and
                ``masks``.
            img_meta (dict): Image informtation.

        Returns:
            tuple[Tensor]: A tuple containing the following for one image.

                - labels (Tensor): Labels of each image. \
                    shape (num_queries, ).
                - label_weights (Tensor): Label weights of each image. \
                    shape (num_queries, ).
                - mask_targets (Tensor): Mask targets of each image. \
                    shape (num_queries, h, w).
                - mask_weights (Tensor): Mask weights of each image. \
                    shape (num_queries, ).
                - pos_inds (Tensor): Sampled positive indices for each \
                    image.
                - neg_inds (Tensor): Sampled negative indices for each \
                    image.
                - sampling_result (:obj:`SamplingResult`): Sampling results.
        """
        gt_labels = gt_instances.labels
        gt_masks = gt_instances.masks
        # sample points
        num_queries = cls_score.shape[0]
        num_gts = gt_labels.shape[0]

        point_coords = torch.rand((1, self.num_points, 2),
                                  device=cls_score.device)
        # shape (num_queries, num_points)
        mask_points_pred = point_sample(
            mask_pred.unsqueeze(1), point_coords.repeat(num_queries, 1,
                                                        1)).squeeze(1)
        # shape (num_gts, num_points)
        gt_points_masks = point_sample(
            gt_masks.unsqueeze(1).float(), point_coords.repeat(num_gts, 1,
                                                               1)).squeeze(1)

        sampled_gt_instances = InstanceData(
            labels=gt_labels, masks=gt_points_masks)
        sampled_pred_instances = InstanceData(
            scores=cls_score, masks=mask_points_pred)
        # assign and sample
        assign_result = self.assigner.assign(
            pred_instances=sampled_pred_instances,
            gt_instances=sampled_gt_instances,
            img_meta=img_meta)
        pred_instances = InstanceData(scores=cls_score, masks=mask_pred)
        sampling_result = self.sampler.sample(
            assign_result=assign_result,
            pred_instances=pred_instances,
            gt_instances=gt_instances)
        pos_inds = sampling_result.pos_inds
        neg_inds = sampling_result.neg_inds

        # label target
        # modify
        labels = gt_labels.new_full((self.num_queries, ),
                                    2,
                                    dtype=torch.long)
        labels[pos_inds] = gt_labels[sampling_result.pos_assigned_gt_inds]
        label_weights = gt_labels.new_ones((self.num_queries, ))
        
        # mask target
        mask_targets = gt_masks[sampling_result.pos_assigned_gt_inds]
        mask_weights = mask_pred.new_zeros((self.num_queries, ))
        mask_weights[pos_inds] = 1.0

        return (labels, label_weights, mask_targets, mask_weights, pos_inds,
                neg_inds, sampling_result)

    def loss_by_feat(self, all_cls_scores: Tensor, all_mask_preds: Tensor,
                     batch_gt_instances: List[InstanceData],
                     batch_img_metas: List[dict]) -> Dict[str, Tensor]:
        """Loss function.

        Args:
            all_cls_scores (Tensor): Classification scores for all decoder
                layers with shape (num_decoder, batch_size, num_queries,
                cls_out_channels). Note `cls_out_channels` should includes
                background.
            all_mask_preds (Tensor): Mask scores for all decoder layers with
                shape (num_decoder, batch_size, num_queries, h, w).
            batch_gt_instances (list[obj:`InstanceData`]): each contains
                ``labels`` and ``masks``.
            batch_img_metas (list[dict]): List of image meta information.

        Returns:
            dict[str, Tensor]: A dictionary of loss components.
        """
        num_dec_layers = len(all_cls_scores)
        batch_gt_instances_list = [
            batch_gt_instances for _ in range(num_dec_layers)
        ]
        img_metas_list = [batch_img_metas for _ in range(num_dec_layers)]
        losses_cls, losses_mask, losses_dice, losses_seg = multi_apply(
            self._loss_by_feat_single, all_cls_scores, all_mask_preds,
            batch_gt_instances_list, img_metas_list)

        loss_dict = dict()
        # loss from the last decoder layer
        loss_dict['loss_cls'] = losses_cls[-1]
        loss_dict['loss_mask'] = losses_mask[-1]
        loss_dict['loss_dice'] = losses_dice[-1]
        if losses_seg[0] is not None:
            loss_dict['loss_seg'] = losses_seg[-1]
        # loss from other decoder layers
        num_dec_layer = 0
        for loss_cls_i, loss_mask_i, loss_dice_i, loss_seg in zip(
                losses_cls[:-1], losses_mask[:-1], losses_dice[:-1], losses_seg[:-1]):
            loss_dict[f'd{num_dec_layer}.loss_cls'] = loss_cls_i
            loss_dict[f'd{num_dec_layer}.loss_mask'] = loss_mask_i
            loss_dict[f'd{num_dec_layer}.loss_dice'] = loss_dice_i
            if loss_seg is not None:
                loss_dict[f'd{num_dec_layer}.loss_seg'] = loss_seg
            num_dec_layer += 1
        return loss_dict
    
    
    def _loss_by_feat_single(self, cls_scores: Tensor, mask_preds: Tensor,
                             batch_gt_instances: List[InstanceData],
                             batch_img_metas: List[dict]) -> Tuple[Tensor]:
        """Loss function for outputs from a single decoder layer.

        Args:
            cls_scores (Tensor): Mask score logits from a single decoder layer
                for all images. Shape (batch_size, num_queries,
                cls_out_channels). Note `cls_out_channels` should includes
                background.
            mask_preds (Tensor): Mask logits for a pixel decoder for all
                images. Shape (batch_size, num_queries, h, w).
            batch_gt_instances (list[obj:`InstanceData`]): each contains
                ``labels`` and ``masks``.
            batch_img_metas (list[dict]): List of image meta information.

        Returns:
            tuple[Tensor]: Loss components for outputs from a single \
                decoder layer.
        """
        num_imgs = cls_scores.size(0)
        cls_scores_list = [cls_scores[i] for i in range(num_imgs)]
        mask_preds_list = [mask_preds[i] for i in range(num_imgs)]
        (labels_list, label_weights_list, mask_targets_list, mask_weights_list,
         avg_factor) = self.get_targets(cls_scores_list, mask_preds_list,
                                        batch_gt_instances, batch_img_metas)
        # shape (batch_size, num_queries)
        labels = torch.stack(labels_list, dim=0)
        # shape (batch_size, num_queries)
        label_weights = torch.stack(label_weights_list, dim=0)
        # shape (num_total_gts, h, w)
        mask_targets = torch.cat(mask_targets_list, dim=0)
        # shape (batch_size, num_queries)
        mask_weights = torch.stack(mask_weights_list, dim=0)
        
        # classfication loss
        # shape (batch_size * num_queries, )
        cls_scores = cls_scores.flatten(0, 1)
        labels = labels.flatten(0, 1)
        label_weights = label_weights.flatten(0, 1)

        class_weight = cls_scores.new_tensor(self.class_weight)
        
        
        loss_cls = self.loss_cls(
            cls_scores,
            labels,
            label_weights,
            avg_factor=class_weight[labels].sum())

        num_total_masks = reduce_mean(cls_scores.new_tensor([avg_factor]))
        num_total_masks = max(num_total_masks, 1)

        # extract positive ones
        # shape (batch_size, num_queries, h, w) -> (num_total_gts, h, w)
        mask_preds = mask_preds[mask_weights > 0]

        if mask_targets.shape[0] == 0:
            # zero match
            loss_dice = mask_preds.sum()
            loss_mask = mask_preds.sum()
            return loss_cls, loss_mask, loss_dice

        with torch.no_grad():
            points_coords = get_uncertain_point_coords_with_randomness(
                mask_preds.unsqueeze(1), None, self.num_points,
                self.oversample_ratio, self.importance_sample_ratio)
            # shape (num_total_gts, h, w) -> (num_total_gts, num_points)
            mask_point_targets = point_sample(
                mask_targets.unsqueeze(1).float(), points_coords).squeeze(1)
        # shape (num_queries, h, w) -> (num_queries, num_points)
        mask_point_preds = point_sample(
            mask_preds.unsqueeze(1), points_coords).squeeze(1)

        # dice loss
        loss_dice = self.loss_dice(
            mask_point_preds, mask_point_targets, avg_factor=num_total_masks)

        # mask loss
        # shape (num_queries, num_points) -> (num_queries * num_points, )
        mask_point_preds = mask_point_preds.reshape(-1)
        # shape (num_total_gts, num_points) -> (num_total_gts * num_points, )
        mask_point_targets = mask_point_targets.reshape(-1)
        loss_mask = self.loss_mask(
            mask_point_preds,
            mask_point_targets,
            avg_factor=num_total_masks * self.num_points)
        
        loss_seg = None
        if self.loss_seg is not None:
            loss_seg = self.loss_seg(torch.stack(cls_scores_list), \
                                    torch.stack(mask_preds_list), batch_gt_instances, batch_img_metas)
        
        return loss_cls, loss_mask, loss_dice, loss_seg

    def forward(self, x: List[Tensor],
                batch_data_samples: SampleList) -> Tuple[List[Tensor]]:
        batch_size = x[0].shape[0]
        mask_features, multi_scale_memorys = self.pixel_decoder(x)
        # multi_scale_memorys (from low resolution to high resolution)
        decoder_inputs = []
        decoder_positional_encodings = []
        for i in range(self.num_transformer_feat_level):
            decoder_input = self.decoder_input_projs[i](multi_scale_memorys[i])
            # shape (batch_size, c, h, w) -> (batch_size, h*w, c)
            decoder_input = decoder_input.flatten(2).permute(0, 2, 1)
            level_embed = self.level_embed.weight[i].view(1, 1, -1)
            decoder_input = decoder_input + level_embed
            # shape (batch_size, c, h, w) -> (batch_size, h*w, c)
            mask = decoder_input.new_zeros(
                (batch_size, ) + multi_scale_memorys[i].shape[-2:],
                dtype=torch.bool)
            decoder_positional_encoding = self.decoder_positional_encoding(
                mask)
            decoder_positional_encoding = decoder_positional_encoding.flatten(
                2).permute(0, 2, 1)
            decoder_inputs.append(decoder_input)
            decoder_positional_encodings.append(decoder_positional_encoding)
        # shape (num_queries, c) -> (batch_size, num_queries, c)
        query_feat = (self.query_feat.weight + self.query_feat_p2a.weight).\
                        unsqueeze(0).repeat((batch_size, 1, 1))
        query_embed = (self.query_embed.weight + self.query_embed_p2a.weight).\
                        unsqueeze(0).repeat((batch_size, 1, 1))

        cls_pred_list = []
        cls_pred_p2a_list = []
        mask_pred_list = []
        cls_pred, cls_pred_p2a, mask_pred, attn_mask = self._forward_head(
            query_feat, mask_features, multi_scale_memorys[0].shape[-2:])
        cls_pred_list.append(cls_pred)
        cls_pred_p2a_list.append(cls_pred_p2a)
        mask_pred_list.append(mask_pred)

        for i in range(self.num_transformer_decoder_layers):
            level_idx = i % self.num_transformer_feat_level
            # if a mask is all True(all background), then set it all False.
            mask_sum = (attn_mask.sum(-1) != attn_mask.shape[-1]).unsqueeze(-1)
            attn_mask = attn_mask & mask_sum
            # cross_attn + self_attn
            if not self.use_attn_mask:
                attn_mask = None
            layer = self.transformer_decoder.layers[i]
            query_feat = layer(
                query=query_feat,
                key=decoder_inputs[level_idx],
                value=decoder_inputs[level_idx],
                query_pos=query_embed,
                key_pos=decoder_positional_encodings[level_idx],
                cross_attn_mask=attn_mask,
                query_key_padding_mask=None,
                # here we do not apply masking on padded region
                key_padding_mask=None)
            cls_pred, cls_pred_p2a, mask_pred, attn_mask = self._forward_head(
                query_feat, mask_features, multi_scale_memorys[
                    (i + 1) % self.num_transformer_feat_level].shape[-2:])

            cls_pred_list.append(cls_pred)
            cls_pred_p2a_list.append(cls_pred_p2a)
            mask_pred_list.append(mask_pred)

        return cls_pred_list, cls_pred_p2a_list, mask_pred_list

    def loss(self, x: Tuple[Tensor], batch_data_samples: SampleList,
             train_cfg: ConfigType) -> dict:
        # batch SegDataSample to InstanceDataSample
        batch_gt_instances, batch_img_metas = self._seg_data_to_instance_data(
            batch_data_samples)

        # forward
        all_cls_scores, all_cls_scores_p2a, all_mask_preds = self(x, batch_data_samples)

        # loss
        losses = self.loss_by_feat(all_cls_scores_p2a, all_mask_preds,
                                   batch_gt_instances, batch_img_metas)

        return losses


@MODELS.register_module()
class Mask2FormerHeadP2A3(Mask2FormerHeadP2A2):
    
    def loss_by_feat(self, all_cls_scores: Tensor, all_mask_preds: Tensor,
                     batch_gt_instances: List[InstanceData],
                     batch_img_metas: List[dict]) -> Dict[str, Tensor]:
        num_dec_layers = len(all_cls_scores)
        batch_gt_instances_list = [
            batch_gt_instances for _ in range(num_dec_layers)
        ]
        img_metas_list = [batch_img_metas for _ in range(num_dec_layers)]
        # modify: just train the last layer
        # losses_cls, losses_mask, losses_dice, losses_contrastive = multi_apply(
        #     self._loss_by_feat_single, all_cls_scores, all_mask_preds,
        #     batch_gt_instances_list, img_metas_list)
        
        loss_cls, loss_mask, loss_dice = self._loss_by_feat_single(all_cls_scores[-1], all_mask_preds[-1],
                                                                                    batch_gt_instances_list[-1], img_metas_list[-1])
        
        loss_dict = dict()
        # loss from the last decoder layer
        loss_dict['loss_cls'] = loss_cls
        loss_dict['loss_mask'] = loss_mask
        loss_dict['loss_dice'] = loss_dice
        
        return loss_dict


@MODELS.register_module()
class Mask2FormerHeadWithoutMask(Mask2FormerHead):
    
    def forward(self, x: List[Tensor],
                batch_data_samples: SampleList) -> Tuple[List[Tensor]]:
        batch_size = x[0].shape[0]
        mask_features, multi_scale_memorys = self.pixel_decoder(x)
        # multi_scale_memorys (from low resolution to high resolution)
        decoder_inputs = []
        decoder_positional_encodings = []
        for i in range(self.num_transformer_feat_level):
            decoder_input = self.decoder_input_projs[i](multi_scale_memorys[i])
            # shape (batch_size, c, h, w) -> (batch_size, h*w, c)
            decoder_input = decoder_input.flatten(2).permute(0, 2, 1)
            level_embed = self.level_embed.weight[i].view(1, 1, -1)
            decoder_input = decoder_input + level_embed
            # shape (batch_size, c, h, w) -> (batch_size, h*w, c)
            mask = decoder_input.new_zeros(
                (batch_size, ) + multi_scale_memorys[i].shape[-2:],
                dtype=torch.bool)
            decoder_positional_encoding = self.decoder_positional_encoding(
                mask)
            decoder_positional_encoding = decoder_positional_encoding.flatten(
                2).permute(0, 2, 1)
            decoder_inputs.append(decoder_input)
            decoder_positional_encodings.append(decoder_positional_encoding)
        # shape (num_queries, c) -> (batch_size, num_queries, c)
        query_feat = self.query_feat.weight.unsqueeze(0).repeat(
            (batch_size, 1, 1))
        query_embed = self.query_embed.weight.unsqueeze(0).repeat(
            (batch_size, 1, 1))

        cls_pred_list = []
        mask_pred_list = []
        cls_pred, mask_pred, attn_mask = self._forward_head(
            query_feat, mask_features, multi_scale_memorys[0].shape[-2:])
        cls_pred_list.append(cls_pred)
        mask_pred_list.append(mask_pred)

        for i in range(self.num_transformer_decoder_layers):
            level_idx = i % self.num_transformer_feat_level
            # if a mask is all True(all background), then set it all False.
            mask_sum = (attn_mask.sum(-1) != attn_mask.shape[-1]).unsqueeze(-1)
            attn_mask = attn_mask & mask_sum
            # attn_mask = torch.zeros_like(attn_mask).bool()
            attn_mask = None
            # cross_attn + self_attn
            layer = self.transformer_decoder.layers[i]
            query_feat = layer(
                query=query_feat,
                key=decoder_inputs[level_idx],
                value=decoder_inputs[level_idx],
                query_pos=query_embed,
                key_pos=decoder_positional_encodings[level_idx],
                cross_attn_mask=attn_mask,
                query_key_padding_mask=None,
                # here we do not apply masking on padded region
                key_padding_mask=None)
            cls_pred, mask_pred, attn_mask = self._forward_head(
                query_feat, mask_features, multi_scale_memorys[
                    (i + 1) % self.num_transformer_feat_level].shape[-2:])

            cls_pred_list.append(cls_pred)
            mask_pred_list.append(mask_pred)

        return cls_pred_list, mask_pred_list

