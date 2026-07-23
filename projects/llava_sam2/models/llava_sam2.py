from typing import Literal

import torch
import torch.nn as nn
import torch.nn.functional as F
from third_parts.mmdet.models.losses import CrossEntropyLoss
import os
import gc

from xtuner.registry import BUILDER
from xtuner.model.utils import get_peft_model_state_dict

from .lisa import LisaModel
from mask3d import get_model, load_mesh, prepare_data

from xtuner.utils import PROMPT_TEMPLATE
from xtuner.tools.utils import get_stop_criteria
from transformers import GenerationConfig
from projects.llava_sam2.models.preprocess.image_resize import DirectResize
from mmengine.runner import get_state_dict
import numpy as np

from .internvl import InternVL_Slowfast
from .utils import dynamic_preprocess

import torchvision.transforms as T
from torchvision.transforms.functional import InterpolationMode

from pycocotools import mask as _mask

from types import MethodType

from xtuner.model.utils import guess_load_checkpoint

from mmcv.ops import point_sample
from third_parts.mmdet.models.utils import get_uncertain_point_coords_with_randomness
import torch.distributed as dist
class diceloss(torch.nn.Module):
    def __init__(self, smooth: float = 1e-3):
        super().__init__()
        self.smooth = smooth
    def forward(self, logits: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        probs = torch.sigmoid(logits)
        target = target.float()
        a = torch.sum(probs * target)
        b = torch.sum(probs)+self.smooth
        c = torch.sum(target)+self.smooth
        d = (2 * a ) / (b + c )
        return 1.0 - d


import pudb

def iou_3d(box1, box2):
    """
    box1, box2: (xmin, ymin, zmin, xmax, ymax, zmax)
    returns IoU in [0, 1]
    """
    x1_min, y1_min, z1_min, x1_max, y1_max, z1_max = box1
    x2_min, y2_min, z2_min, x2_max, y2_max, z2_max = box2

    # intersection box
    xi_min = max(x1_min, x2_min)
    yi_min = max(y1_min, y2_min)
    zi_min = max(z1_min, z2_min)
    xi_max = min(x1_max, x2_max)
    yi_max = min(y1_max, y2_max)
    zi_max = min(z1_max, z2_max)

    # side lengths of intersection
    inter_dx = max(0.0, xi_max - xi_min)
    inter_dy = max(0.0, yi_max - yi_min)
    inter_dz = max(0.0, zi_max - zi_min)

    inter_vol = inter_dx * inter_dy * inter_dz

    # volumes of each box
    vol1 = max(0.0, x1_max - x1_min) * max(0.0, y1_max - y1_min) * max(0.0, z1_max - z1_min)
    vol2 = max(0.0, x2_max - x2_min) * max(0.0, y2_max - y2_min) * max(0.0, z2_max - z2_min)

    # union volume
    union_vol = vol1 + vol2 - inter_vol

    if union_vol == 0.0:
        return 0.0

    return inter_vol / union_vol


class VideoLLaVASAMModel(LisaModel):
    def __init__(self,
                 mllm,
                 tokenizer,
                 grounding_encoder=None,
                 loss_mask=None,
                 loss_dice=None,
                 torch_dtype=torch.bfloat16,
                 pretrained_pth=None,
                 frozen_sam2_decoder=True,
                 special_tokens=None,
                 loss_sample_points=False,
                 num_points=12544,
                 # for slow fast arch
                 fast_pool=False,
                 fast_pool_size=4,
                 use_fast_supervision=False,
                 # for inference
                 phi3=True,
                 template=None,
                 # for arch selection
                 arch_type:Literal['intern_vl', 'qwen', 'llava']='intern_vl',
                 # for inference large model
                 split_model=False,
                 # ext
                 preprocessor=None,
                 # bs
                 bs:int=0,
                 ):
        super(LisaModel, self).__init__()
        self.split_model = split_model
        if split_model:
            mllm.model_split = split_model
        if special_tokens is None:
            special_tokens = ['[SEG]']
        self.special_tokens = special_tokens
        if 'special_tokens' not in mllm.keys():
            mllm.special_tokens = special_tokens
        self.mllm = BUILDER.build(mllm)
        self.arch_type = arch_type

        self.fast_pool = fast_pool
        self.fast_pool_size = fast_pool_size
        if hasattr(self.mllm, '_post_init'):
            self.mllm._post_init(
                fast_pool_size=self.fast_pool_size,
                fast_pool=self.fast_pool
            )
        else:
            print("No _post_init() in mllm !!!")

        self.tokenizer = BUILDER.build(tokenizer)
        self._add_special_tokens()
        #self.grounding_encoder = BUILDER.build(grounding_encoder)
        #self.grounding_encoder.requires_grad_(False)
        # if not frozen_sam2_decoder:
        #     self.grounding_encoder.sam2_model.sam_mask_decoder.requires_grad_(True)

        if self.mllm.use_llm_lora:
            if self.arch_type == 'intern_vl':
                self.mllm.model.language_model.base_model.model.get_input_embeddings().requires_grad_(True)
                self.mllm.model.language_model.base_model.model.get_output_embeddings().requires_grad_(True)
            elif self.arch_type == 'qwen':
                self.mllm.model.model.base_model.model.get_input_embeddings().requires_grad_(True)
                self.mllm.model.get_output_embeddings().weight.requires_grad_(True)
            elif self.arch_type == 'llava':
                self.mllm.model.language_model.base_model.model.get_input_embeddings().requires_grad_(True)
                self.mllm.model.language_model.base_model.model.get_output_embeddings().requires_grad_(True)
            # self.mllm.model.language_model.base_model.model.lm_head.requires_grad_(True)
            # self.mllm.model.language_model.base_model.model.model.embed_tokens.requires_grad_(True)

        if self.arch_type == 'intern_vl':
            in_dim = self.mllm.model.config.llm_config.hidden_size
        elif self.arch_type == 'qwen':
            in_dim = self.mllm.model.config.hidden_size
        elif self.arch_type == 'llava':
            # for llava, the hidden size is in language model
            in_dim = self.mllm.model.language_model.config.hidden_size
        out_dim = 256 #self.grounding_encoder.hidden_dim
        self.text_hidden_fcs = nn.Sequential(
            nn.Linear(in_dim, in_dim), nn.ReLU(inplace=True),
            nn.Linear(in_dim, out_dim), nn.Dropout(0.0)
        )

        if use_fast_supervision:
            self.text_exist_fcs = nn.Sequential(
                nn.Linear(in_dim, in_dim), nn.ReLU(inplace=True),
                nn.Linear(in_dim, 1), nn.Dropout(0.0)
            )

        self.loss_mask = BUILDER.build(loss_mask)
        self.loss_dice = BUILDER.build(loss_dice)
        if use_fast_supervision:
            self.loss_exists = BUILDER.build(dict(
                type=CrossEntropyLoss,
                use_sigmoid=True,
                reduction='mean',
                loss_weight=1.0)
            )

        self.torch_dtype = torch_dtype

        if pretrained_pth is not None:
            pretrained_state_dict = guess_load_checkpoint(pretrained_pth)
            self.load_state_dict(pretrained_state_dict, strict=False)
            print(f'Load pretrained weight from {pretrained_pth}')

        self.loss_sample_points = loss_sample_points
        self.num_points = num_points
        self.oversample_ratio = 3.0
        self.importance_sample_ratio = 0.75

        if fast_pool:
            self.fast_token_idx = self.tokenizer("<FAST_IMG_CONTEXT>", add_special_tokens=False).input_ids[0]
        else:
            self.fast_token_idx = None
        self.use_fast_supervision = use_fast_supervision

        self.phi3 = phi3
        self.template = template

        if preprocessor is None:
            self.preprocessor = preprocessor
        else:
            self.preprocessor = BUILDER.build(preprocessor)

        self.bs = bs

    def _merge_lora(self):
        # print('pre merge lora: ', self.mllm.model.language_model.base_model.model.get_input_embeddings().weight.shape)
        try:
            self.mllm.model.language_model = self.mllm.model.language_model.merge_and_unload()
        except:
            print("Skip language model, no LoRA in it !!!")
        try:
            self.mllm.model.vision_model = self.mllm.model.vision_model.merge_and_unload()
        except:
            print("Skip vision encoder, no LoRA in it !!!")
        # print('after merge lora: ', self.mllm.model.language_model.get_input_embeddings().weight.shape)
        return

    # def state_dict(self, *args, **kwargs):
    #     return get_state_dict(self)
    #     state_dict = super(LisaModel, self).state_dict(*args, **kwargs)
    #     from collections import OrderedDict

    #     to_return = OrderedDict()
    #     # Step 1. visual_encoder
    #     if self.mllm.use_visual_encoder_lora:
    #         to_return.update(
    #             get_peft_model_state_dict(
    #                 self.mllm.model.vision_model, state_dict=state_dict))
    #         raise NotImplementedError
    #     elif not self.mllm.freeze_visual_encoder:
    #         to_return.update({
    #             k: v
    #             for k, v in state_dict.items() if 'visual_encoder.' in k
    #         })
    #         raise NotImplementedError
    #     # Step 2. LLM
    #     if self.mllm.use_llm_lora:
    #         if self.arch_type == 'intern_vl':
    #             to_return.update(
    #                 get_peft_model_state_dict(self.mllm.model.language_model, state_dict=state_dict)
    #             )
    #         elif self.arch_type == 'qwen':
    #             to_return.update(
    #                 get_peft_model_state_dict(self.mllm.model.model, state_dict=state_dict)
    #             )
    #         elif self.arch_type == 'llava':
    #             to_return.update(
    #                 get_peft_model_state_dict(self.mllm.model.language_model, state_dict=state_dict)
    #             )
    #     elif not self.mllm.freeze_llm:
    #         to_return.update(
    #             {k: v
    #              for k, v in state_dict.items() if 'llm.' in k})
    #         raise NotImplementedError
    #     # Step 3. Projector
    #     return to_return





    # def all_state_dict(self, *args, **kwargs):
    #     state_dict = super(LisaModel, self).state_dict(*args, **kwargs)
    #     return state_dict

    def activation_checkpointing_disable(self):
        if self.arch_type == 'qwen':
            self.mllm.model.model.gradient_checkpointing_disable()
        else:
            self.mllm.model.language_model.gradient_checkpointing_disable()


    def _add_special_tokens(self):
        special_tokens = self.special_tokens
        _num_new_tokens = self.tokenizer.add_tokens(special_tokens, special_tokens=True)

        # if not isinstance(self.mllm.model.language_model.get_output_embeddings(), nn.Linear):
        #     print("Change the lm_head to nn.Linear !!!")
        #     transposed = False
        #     old_lm_head = self.mllm.model.language_model.get_output_embeddings()
        #     old_num_tokens, old_lm_head_dim = (
        #         old_lm_head.weight.size() if not transposed else old_lm_head.weight.t().size()
        #     )
        #     new_lm_head_shape = (old_lm_head_dim, len(tokenizer)) if not transposed else (
        #     len(tokenizer), old_lm_head_dim)
        #     has_new_lm_head_bias = old_lm_head.bias is not None
        #     new_lm_head = nn.Linear(*new_lm_head_shape, bias=has_new_lm_head_bias).to(self.device)
        #     new_lm_head.weight = old_lm_head.weight
        #     new_lm_head.bias = old_lm_head.bias
        #     self.mllm.model.language_model.set_output_embeddings(new_lm_head)

        # this is already done in mllm
        # if num_new_tokens > 0:
        #     self.mllm.model.language_model.resize_token_embeddings(len(self.tokenizer))

        # assert isinstance(self.mllm, InternVL_Slowfast)
        self.seg_token_idx = self.tokenizer("[SEG]", add_special_tokens=False).input_ids[0]

    def check_obj_number(self, pred_embeddings_list_video, gt_masks_video, fix_number=5):
        assert len(pred_embeddings_list_video) == len(gt_masks_video)
        ret_pred_embeddings_list_video = []
        ret_gt_masks_video = []
        for pred_mebeds, gt_masks in zip(pred_embeddings_list_video, gt_masks_video):
            # assert len(pred_mebeds) == len(gt_masks)
            if len(pred_mebeds) != len(gt_masks):
                min_num = min(len(pred_mebeds), len(gt_masks))
                pred_mebeds = pred_mebeds[:min_num]
                gt_masks = gt_masks[:min_num]
            if len(pred_mebeds) != fix_number:
                if len(pred_mebeds) > fix_number:
                    _idxs = torch.randperm(pred_mebeds.shape[0])
                    _idxs = _idxs[:fix_number]
                    pred_mebeds = pred_mebeds[_idxs]
                    gt_masks = gt_masks[_idxs]
                else:
                    n_repeat = fix_number // len(pred_mebeds) + 1
                    pred_mebeds = torch.cat([pred_mebeds] * n_repeat, dim=0)[:fix_number]
                    gt_masks = torch.cat([gt_masks] * n_repeat, dim=0)[:fix_number]
            ret_pred_embeddings_list_video.append(pred_mebeds)
            ret_gt_masks_video.append(gt_masks)
        return ret_pred_embeddings_list_video, ret_gt_masks_video

    def _get_pesudo_data(self, dtype, device):
        assert self.bs > 0
        g_pixel_values = torch.zeros((3, 1024, 1024), dtype=dtype, device=device)
        g_pixel_values = [g_pixel_values] * self.bs
        frames_per_batch = [1] * self.bs
        gt_masks = torch.zeros((5, 256, 256), dtype=torch.uint8, device=device)
        gt_masks = [gt_masks] * self.bs
        return g_pixel_values, frames_per_batch, gt_masks

    def forward(self, data, data_samples=None, mode='loss'):
        g_pixel_values = data.pop('g_pixel_values', None)
        gt_masks = data.pop('masks', None)
        frames_per_batch = data.pop('frames_per_batch', None)
        input_ids = data['input_ids']
        fast_exists = data.pop('fast_exists', None)
        # if self.arch_type == 'llava' and data.get('pixel_values', None) is not None:
        #     data['pixel_values'] = data['pixel_values'].to(self.torch_dtype)
        if self.fast_pool:
            output = self.mllm(data, data_samples, mode, fast_token_idx=self.fast_token_idx)
        else:
            output = self.mllm(data, data_samples, mode)
        if gt_masks is None:
            # require zero seg datas
            seg_valid = False
            g_pixel_values, frames_per_batch, gt_masks = self._get_pesudo_data(
                dtype=self.torch_dtype,
                device=input_ids.device,
            )
        else:
            seg_valid = True

        assert frames_per_batch, "Video Lisa require frames_per_batch !!!"
        # print('frmaes_per_batch: ', frames_per_batch)
        ori_size_list = []
        for i_bs, mask in enumerate(gt_masks):
            mask_shape = mask.shape[-2:]
            ori_size_list += [mask_shape] * frames_per_batch[i_bs]

        seg_token_mask = input_ids == self.seg_token_idx

        hidden_states = output.hidden_states
        hidden_states = self.text_hidden_fcs(hidden_states[-1])

        _zero = hidden_states.mean() * 0.0
        if seg_valid:
            pred_embeddings = hidden_states[seg_token_mask] + _zero
        else:
            pred_embeddings = hidden_states[:, :5].flatten(0, 1) + _zero

        seg_token_counts = seg_token_mask.int().sum(-1)
        if not seg_valid:
            seg_token_counts += 5

        pred_embeddings_list_ = torch.split(pred_embeddings, seg_token_counts.tolist(), dim=0)
        pred_embeddings_list = []
        for item in pred_embeddings_list_:
            if len(item) != 0:
                pred_embeddings_list.append(item)
        pred_embeddings_list_video, success = self.genetate_video_pred_embeddings(
            pred_embeddings_list, frames_per_batch)
        if not success:
            raise NotImplementedError

        if self.use_fast_supervision and fast_exists is not None:
            # gt_exists = []
            # for id_x, _fast_exists in enumerate(fast_exists):
            #     num_tot = _fast_exists.shape[0]
            #     num_conv = gt_masks[id_x].shape[0] // frames_per_batch[id_x]
            #     assert num_tot % num_conv == 0
            #     gt_exists.append(_fast_exists.reshape(num_conv, num_tot // num_conv))
            fast_flag = input_ids == self.fast_token_idx
            fast_tokens = output.hidden_states[-1][fast_flag]
            exists_logit = self.text_exist_fcs(fast_tokens[self.fast_pool_size ** 2 - 1::self.fast_pool_size ** 2])
            gt_exists = torch.cat(fast_exists)
            loss_exists = self.loss_exists(exists_logit, gt_exists)
        else:
            loss_exists = None

        gt_masks_video = self.process_video_gt_masks(gt_masks, frames_per_batch)
        pred_embeddings_list_video, gt_masks_video = self.check_obj_number(
            pred_embeddings_list_video, gt_masks_video
        )
        g_pixel_values = torch.stack([
            self.grounding_encoder.preprocess_image(pixel) for pixel in g_pixel_values
        ])
        num_objs = pred_embeddings_list_video[0].shape[0]
        num_frames = len(pred_embeddings_list_video)
        language_embeddings = torch.cat(pred_embeddings_list_video, dim=0)[:, None]
        sam_states = self.grounding_encoder.get_sam2_embeddings(g_pixel_values, expand_size=num_objs)
        pred_masks = self.grounding_encoder.inject_language_embd(sam_states, language_embeddings, nf_nobj=(num_frames, num_objs))

        gt_masks = [F.interpolate(gt_mask.unsqueeze(0), size=pred_masks[0].shape[-2:], mode='nearest').squeeze(0) for gt_mask in gt_masks_video]
        gt_masks = torch.cat(gt_masks, dim=0)
        pred_masks = pred_masks.flatten(0, 1)

        loss_mask, loss_dice = 0, 0
        if len(pred_masks) != len(gt_masks):
            # drop this data
            print(f"Pred mask shape {pred_masks.shape} is not equal to gt_mask shape {gt_masks.shape} !!!")
            min_num = min(len(pred_masks), len(gt_masks))
            pred_masks = pred_masks[:min_num]
            gt_masks = gt_masks[:min_num]
            seg_valid = False

        if self.loss_sample_points:
            sampled_pred_mask, sampled_gt_mask = self.sample_points(pred_masks, gt_masks)
            sam_loss_dice = self.loss_dice(
                sampled_pred_mask,
                sampled_gt_mask, avg_factor=(len(gt_masks) + 1e-4))
            sam_loss_mask = self.loss_mask(
                sampled_pred_mask.reshape(-1),
                sampled_gt_mask.reshape(-1),
                avg_factor=(pred_masks.shape[0] * sampled_pred_mask.shape[1] + 1e-4))
        else:
            sam_loss_mask = self.loss_mask(pred_masks, gt_masks)
            sam_loss_dice = self.loss_dice(pred_masks, gt_masks)
        loss_mask += sam_loss_mask
        loss_dice += sam_loss_dice

        if not seg_valid:
            _scale = 0.0
        else:
            _scale = 1.0
        loss_mask = loss_mask * _scale
        loss_dice = loss_dice * _scale

        loss_dict = {
            'loss_mask': loss_mask,
            'loss_dice': loss_dice,
            'llm_loss': output.loss,
        }
        if loss_exists is not None:
            loss_dict['loss_exists'] = loss_exists
        return loss_dict

    def sample_points(self, mask_pred, gt_masks):
        gt_masks = gt_masks.unsqueeze(1)
        gt_masks = gt_masks.to(mask_pred)
        mask_pred = mask_pred.unsqueeze(1)
        # (N, 1, h, w)

        with torch.no_grad():
            points_coords = get_uncertain_point_coords_with_randomness(
                mask_pred.to(torch.float32), None, self.num_points,
                self.oversample_ratio, self.importance_sample_ratio)
            # shape (num_total_gts, h, w) -> (num_total_gts, num_points)
            mask_point_targets = point_sample(
                gt_masks.float(), points_coords).squeeze(1)
        # shape (num_queries, h, w) -> (num_queries, num_points)
        mask_point_preds = point_sample(
            mask_pred.to(torch.float32), points_coords.to(torch.float32)).squeeze(1)
        return mask_point_preds.to(mask_pred.dtype), mask_point_targets.to(mask_pred.dtype)

    def genetate_video_pred_embeddings(self, pred_embeddings_list, frames_per_batch):
        if len(pred_embeddings_list) == len(frames_per_batch):
            success = True
        else:
            success = False
            print("len(pred_embeddings_list):{} is not equal to len(frames_per_batch):{} !!!".format(len(pred_embeddings_list), len(frames_per_batch)))
        pred_embeddings_list_video = []
        for pred_embedding_batch, frame_nums in zip(pred_embeddings_list, frames_per_batch):
            pred_embeddings_list_video += [pred_embedding_batch] * frame_nums
        return pred_embeddings_list_video, success

    def process_video_gt_masks(self, gt_masks, frames_per_batch):
        gt_masks_video = []

        assert len(gt_masks) == len(frames_per_batch)
        for gt_masks_batch, frames_num in zip(gt_masks, frames_per_batch):
            N, H, W = gt_masks_batch.shape
            assert N % frames_num == 0
            gt_masks_batch = gt_masks_batch.reshape(
                N // frames_num, frames_num, H, W)
            for i in range(frames_num):
                gt_masks_video.append(gt_masks_batch[:, i])
        return gt_masks_video

    def preparing_for_generation(self, metainfo, **kwargs):
        # set stop criteria and generation configs for model
        assert hasattr(self, 'tokenizer'), "The Model does not have the tokenizer!!!"
        self.bot_name = 'BOT'
        if 'template' in metainfo.keys():
            template = metainfo['template']
        else:
            template = PROMPT_TEMPLATE['phi3_chat']
        if self.template is None:
            self.template = template
        stop_words = []
        stop_words += self.template.get('STOP_WORDS', [])
        stop_criteria = get_stop_criteria(
            tokenizer=self.tokenizer, stop_words=stop_words)
        self.stop_criteria = stop_criteria

        default_generation_kwargs = dict(
            max_new_tokens=512,
            do_sample=False,
            eos_token_id=self.tokenizer.eos_token_id,
            pad_token_id=(
                self.tokenizer.pad_token_id
                if self.tokenizer.pad_token_id is not None
                else self.tokenizer.eos_token_id
            ),
        )
        default_generation_kwargs.update(metainfo.get('generation_kwargs', {}))
        self.gen_config = GenerationConfig(**default_generation_kwargs)
        self.init_prediction_config = True

        self.mllm.to(self.torch_dtype)
        self.text_hidden_fcs.to(self.torch_dtype)
        # if getattr(self, 'text_exist_fcs', None) is not None:
        #     self.text_exist_fcs.to(self.torch_dtype)

        # for sam image processor
        self.extra_image_processor = DirectResize(target_length=1024, )
        # for multi image process
        self.min_dynamic_patch = 1
        if 'max_dynamic_patch' in metainfo.keys():
            self.max_dynamic_patch = metainfo['max_dynamic_patch']
        else:
            self.max_dynamic_patch = 12
        self.downsample_ratio = 0.5
        self.image_size = 448
        self.use_thumbnail = True
        patch_size = 14
        self.patch_size = patch_size

        self.patch_token = int((self.image_size // patch_size) ** 2 * (self.downsample_ratio ** 2))
        self.IMAGENET_MEAN = (0.485, 0.456, 0.406)
        self.IMAGENET_STD = (0.229, 0.224, 0.225)
        self.IMG_CONTEXT_TOKEN = '<IMG_CONTEXT>'
        self.IMG_START_TOKEN = '<img>'
        self.IMG_END_TOKEN = '</img>'
        if self.arch_type == 'qwen':
            self.IMG_CONTEXT_TOKEN = '<|image_pad|>'
            self.IMG_START_TOKEN = ''
            self.IMG_END_TOKEN = ''

        if self.preprocessor is None:
            self.transformer = T.Compose([
                T.Lambda(lambda img: img.convert('RGB') if img.mode != 'RGB' else img),
                T.Resize((self.image_size, self.image_size), interpolation=InterpolationMode.BICUBIC),
                T.ToTensor(),
                T.Normalize(mean=self.IMAGENET_MEAN, std=self.IMAGENET_STD)
            ])
            self.preprocessor = None
        else:
            self.transformer = None
            # self.preprocessor = BUILDER.build(self.preprocessor)

        self.VP_START_TOKEN = '<vp>'
        self.VP_END_TOKEN = '</vp>'

        # change phi3 prepare for generation fuction
        if self.phi3:
            self.mllm.model.language_model.prepare_inputs_for_generation = MethodType(prepare_inputs_for_generation, self.mllm.model.language_model)
        return

    def predict_video(self, pixel_values, text_prompts, **kwargs):
        ori_h, ori_w = kwargs['ori_height'], kwargs['ori_width']

        _input_ids = kwargs['input_ids']

        g_pixel_values = kwargs.pop('g_pixel_values', None)
        g_pixel_values = torch.stack([
            self.grounding_encoder.preprocess_image(pixel) for pixel in g_pixel_values
        ])

        fast_pixel_values = kwargs.pop('fast_pixel_values', None)
        if fast_pixel_values is None:
            fast_token_idx = None
        else:
            fast_token_idx = self.fast_token_idx

        predictions = []
        pred_masks = []
        is_exists_list = []
        for input_ids in _input_ids:
            input_ids = torch.tensor(input_ids).unsqueeze(0)
            attention_mask = torch.ones_like(input_ids, dtype=torch.bool)
            pixel_values = pixel_values.to(dtype=self.torch_dtype)
            if fast_pixel_values is not None:
                fast_pixel_values = fast_pixel_values.to(dtype=self.torch_dtype)
            mm_inputs = {
                'pixel_values': pixel_values,
                'input_ids': input_ids,
                'attention_mask': attention_mask,
                'position_ids': None,
                'past_key_values': None,
                'labels': None,
                'fast_pixel_values': fast_pixel_values,
                'fast_token_idx': fast_token_idx,
            }
            if kwargs.get('image_grid_thw', None) is not None:
                mm_inputs['image_grid_thw'] = kwargs['image_grid_thw']

            generate_output = self.mllm.generate(
                **mm_inputs,
                generation_config=self.gen_config,
                streamer=None,
                bos_token_id=self.tokenizer.bos_token_id,
                stopping_criteria=self.stop_criteria,
                output_hidden_states=True,
                return_dict_in_generate=True
            )

            predict = self.tokenizer.decode(generate_output.sequences[0], skip_special_tokens=False).strip()

            # input_text = self.tokenizer.decode(mm_inputs['input_ids'][0], skip_special_tokens=False)
            # print(input_text, generate_output.sequences[0], '\n', predict, self.tokenizer("[SEG]", add_special_tokens=False).input_ids[0])

            predictions.append(predict)

            hidden_states = generate_output.hidden_states
            last_hidden_states = [item[-1][0] for item in hidden_states]
            last_hidden_states = torch.cat(last_hidden_states, dim=0)
            seg_hidden_states = get_seg_hidden_states(
                last_hidden_states, generate_output.sequences[0][:-1],
                seg_id=self.seg_token_idx
            )

            if len(seg_hidden_states) == 0:
                print("Warning, no [SEG] tokens !!!")
                pred_masks.append(torch.zeros((g_pixel_values.shape[0], ori_h, ori_w), dtype=torch.int))
                continue
            elif len(seg_hidden_states) > 1:
                print("Warning, {} [SEG] tokens !!!".format(len(seg_hidden_states)))
                seg_hidden_states = seg_hidden_states[:1]
            seg_hidden_states = self.text_hidden_fcs(seg_hidden_states)

            seg_hidden_states = seg_hidden_states.to(dtype=torch.float32)

            sam_states = self.grounding_encoder.get_sam2_embeddings(g_pixel_values)
            # TODO: change 5
            if len(pixel_values) < 5:
                pred_mask = self.grounding_encoder.language_embd_inference(sam_states, [seg_hidden_states] * pixel_values.shape[0])
            else:
                pred_mask = self.grounding_encoder.language_embd_inference(sam_states, [seg_hidden_states] * 5)
            pred_mask = F.interpolate(
                pred_mask,
                size=(ori_h, ori_w),
                mode='bilinear',
                align_corners=False,
            )
            pred_mask = pred_mask[:, 0]
            pred_mask = pred_mask.sigmoid() > 0.5
            pred_mask = pred_mask.int()
            # supervision
            if self.use_fast_supervision and (input_ids == self.fast_token_idx).sum() > 0:
                fast_flag = input_ids.squeeze(0) == self.fast_token_idx
                len_out = generate_output.sequences[0][:-1].shape[0]
                fast_tokens = last_hidden_states[:-len_out][fast_flag].to(dtype=torch.float32)
                exists_logit = self.text_exist_fcs(fast_tokens[self.fast_pool_size ** 2 - 1::self.fast_pool_size ** 2])
                is_exists = exists_logit.squeeze(-1).sigmoid() > 0.5
                is_exists_list.append(is_exists)
                not_exists = torch.logical_not(is_exists)
                if torch.any(not_exists):
                    pred_mask[not_exists] = pred_mask[not_exists] * 0

            pred_masks.append(pred_mask)
        assert len(pred_masks) == len(text_prompts)
        ret_dict = {
            'prediction': predictions,
            'prediction_masks': [mask_to_rle(_item.cpu().numpy()) for _item in pred_masks],
        }
        if 'id' in kwargs.keys():
            ret_dict['id'] = kwargs['id']

        if len(is_exists_list) > 0:
            ret_dict['is_exists'] = is_exists_list
        return ret_dict

def get_seg_hidden_states(hidden_states, output_ids, seg_id):
    seg_mask = output_ids == seg_id
    n_out = len(seg_mask)
    return hidden_states[-n_out:][seg_mask]

def mask_to_rle(mask):
    rle = []
    for m in mask:
        rle.append(_mask.encode(np.asfortranarray(m.astype(np.uint8))))
        rle[-1]['counts'] = rle[-1]['counts'].decode()
    return rle

from transformers.cache_utils import Cache, DynamicCache

def prepare_inputs_for_generation(
        self, input_ids, past_key_values=None, attention_mask=None, inputs_embeds=None, **kwargs
):
    if past_key_values is not None:
        if isinstance(past_key_values, Cache):
            cache_length = past_key_values.get_seq_length()
            past_length = past_key_values.seen_tokens
            max_cache_length = past_key_values.get_max_length()
        else:
            cache_length = past_length = past_key_values[0][0].shape[2]
            max_cache_length = None

        # Keep only the unprocessed tokens:
        # 1 - If the length of the attention_mask exceeds the length of input_ids, then we are in a setting where
        # some of the inputs are exclusively passed as part of the cache (e.g. when passing input_embeds as
        # input)
        if attention_mask is not None and attention_mask.shape[1] > input_ids.shape[1]:
            input_ids = input_ids[:, -(attention_mask.shape[1] - past_length):]
        # 2 - If the past_length is smaller than input_ids', then input_ids holds all input tokens. We can discard
        # input_ids based on the past_length.
        elif past_length < input_ids.shape[1]:
            input_ids = input_ids[:, past_length:]
        # 3 - Otherwise (past_length >= input_ids.shape[1]), let's assume input_ids only has unprocessed tokens.

        # If we are about to go beyond the maximum cache length, we need to crop the input attention mask.
        if (
                max_cache_length is not None
                and attention_mask is not None
                and cache_length + input_ids.shape[1] > max_cache_length
        ):
            attention_mask = attention_mask[:, -max_cache_length:]

    position_ids = kwargs.get('position_ids', None)
    if attention_mask is not None and position_ids is None:
        # create position_ids on the fly for batch generation
        position_ids = attention_mask.long().cumsum(-1) - 1
        position_ids.masked_fill_(attention_mask == 0, 1)
        if past_key_values:
            position_ids = position_ids[:, -input_ids.shape[1]:]

    # if `inputs_embeds` are passed, we only want to use them in the 1st generation step
    if inputs_embeds is not None and (past_key_values is None or len(past_key_values)==0):
        model_inputs = {'inputs_embeds': inputs_embeds}
    else:
        model_inputs = {'input_ids': input_ids}

    model_inputs.update(
        {
            'position_ids': position_ids,
            'past_key_values': past_key_values,
            'use_cache': kwargs.get('use_cache'),
            'attention_mask': attention_mask,
        }
    )
    return model_inputs
import torch
import torch.nn.functional as F





def setup_ddp():
    if not dist.is_initialized():
        dist.init_process_group(backend="nccl")

    local_rank = int(os.environ["LOCAL_RANK"])
    torch.cuda.set_device(local_rank)

    return local_rank


# --- initialize DDP ---


class VideoLLaVASAMModel_zero3(VideoLLaVASAMModel):
    def __init__(self,
                 mllm,
                 tokenizer,
                 grounding_encoder=None,
                 loss_mask=None,
                 loss_dice=None,
                 torch_dtype=torch.bfloat16,
                 pretrained_pth=None,
                 frozen_sam2_decoder=True,
                 special_tokens=['[SEG]', ],
                 loss_sample_points=False,
                 num_points=12544,
                 # for slow fast arch
                 fast_pool=False,
                 fast_pool_size=4,
                 arch_type='intern_vl',
                 # zero3
                 bs=1,
                 ):
        super(VideoLLaVASAMModel_zero3, self).__init__(
            mllm=mllm,
            tokenizer=tokenizer,
            grounding_encoder=grounding_encoder,
            loss_mask=loss_mask,
            loss_dice=loss_dice,
            torch_dtype=torch_dtype,
            pretrained_pth=pretrained_pth,
            frozen_sam2_decoder=frozen_sam2_decoder,
            special_tokens=special_tokens,
            loss_sample_points=loss_sample_points,
            num_points=num_points,
            # for slow fast arch
            fast_pool=fast_pool,
            fast_pool_size=fast_pool_size,
            arch_type=arch_type,
        )
        self.bs = bs
        self.iteration=0
        with torch.amp.autocast(device_type='cuda', dtype=torch.float32):
        
            self.segmenter = get_model("/home/vipradas/Thesis/scannet200_val.ckpt")
            local_rank = 0#setup_ddp()
        
            print(local_rank)
            self.device = torch.device(f"cuda:{local_rank}")
            # self.segmenter= self.segmenter.to(self.device)
            # self.segmenter = torch.nn.parallel.DistributedDataParallel(
            #     self.segmenter,
            #     device_ids=[local_rank],
            #     output_device=local_rank
            # )

        self.criterion=nn.BCEWithLogitsLoss()
        #self.pointencoder = nn.Linear(1091, 256)
        self.projector=nn.Linear(1408,2048)
        self.dice= diceloss()
        self.bbox_loss=nn.MSELoss()
        self.lang_projector = nn.Sequential(
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, 128),
            nn.LayerNorm(128),
        )

        
    def state_dict(self, *args, **kwargs):
        return get_state_dict(self)

    def _get_pesudo_data(self, dtype, device):
        g_pixel_values = torch.zeros((3, 1024, 1024), dtype=dtype, device=device)
        g_pixel_values = [g_pixel_values] * self.bs
        frames_per_batch = [1] * self.bs
        gt_masks = torch.zeros((5, 256, 256), dtype=torch.uint8, device=device)
        gt_masks = [gt_masks] * self.bs
        return g_pixel_values, frames_per_batch, gt_masks

    def forward(self, data, data_samples=None, mode='loss'):
        bbx=data.pop("bboxes")
        gt=data.pop("gt")
        point=data.pop('features_3d')
        g_pixel_values = data.pop('g_pixel_values', None)
        gt_masks = data.pop('masks', None)
        frames_per_batch = data.pop('frames_per_batch', None)
        input_ids = data['input_ids']
        obj_ids=data.pop('obj_ids', None)
        scene_id=data.pop('scene_id',None)
        projected=data.pop('projected',None)
        data['projected']=F.normalize(self.projector(projected), p=2, dim=-1)

        if self.fast_pool:
            output = self.mllm(data, data_samples, mode, fast_token_idx=self.fast_token_idx)
        else:
            output = self.mllm(data, data_samples, mode)

        if gt_masks is None:
            # require zero seg datas
            seg_valid = False
            g_pixel_values, frames_per_batch, gt_masks = self._get_pesudo_data(
                dtype=self.torch_dtype,
                device=input_ids.device,
            )
        else:
            seg_valid = True

        assert frames_per_batch, "Video Lisa require frames_per_batch !!!"
        # print('frmaes_per_batch: ', frames_per_batch)
        ori_size_list = []
        for i_bs, mask in enumerate(gt_masks):
            mask_shape = mask.shape[-2:]
            ori_size_list += [mask_shape] * frames_per_batch[i_bs]

        seg_token_mask = input_ids == self.seg_token_idx

        hidden_states = output.hidden_states
        hidden_states = self.text_hidden_fcs(hidden_states[-1])

        _zero = hidden_states.mean() * 0.0
        if seg_valid:
            pred_embeddings = hidden_states[seg_token_mask] + _zero
        else:
            pred_embeddings = hidden_states[:, :5].flatten(0, 1) + _zero
        seg_token_counts = seg_token_mask.int().sum(-1)
        if not seg_valid:
            seg_token_counts += 5

        #print(pred_embeddings.shape)

        # pred_embeddings_list_ = torch.split(pred_embeddings, seg_token_counts.tolist(), dim=0)
        # #print(type(pred_embeddings_list_))
        # pred_embeddings_list = []
        # for item in pred_embeddings_list_:
        #     if len(item) != 0:
        #         pred_embeddings_list.append(item)
        # pred_embeddings_list_video, success = self.genetate_video_pred_embeddings(
        #     pred_embeddings_list, frames_per_batch)
        # if not success:
        #     raise NotImplementedError
            # return {'llm_loss': output.loss, 'loss_mask': output.loss * 0.0, 'loss_dice': output.loss * 0.0}
        #gt_masks_video = self.process_video_gt_masks(gt_masks, frames_per_batch)
        #pred_embeddings_list_video, gt_masks_video = self.check_obj_number(
        #    pred_embeddings_list_video, gt_masks_video
        #)
        #g_pixel_values = torch.stack([
        #    self.grounding_encoder.preprocess_image(pixel) for pixel in g_pixel_values
        #])
        # print(f"Done, {g_pixel_values.device} !!!\n\n")
        #num_objs = pred_embeddings_list_video[0].shape[0]
        #num_frames = len(pred_embeddings_list_video)
        language_embeddings = pred_embeddings#torch.cat(pred_embeddings_list_video, dim=0)
        # print("DIFERENCES:  ",(pred_embeddings[0]-pred_embeddings[1]).sum(), (pred_embeddings[2]-pred_embeddings[1]).sum(), (pred_embeddings[3]-pred_embeddings[1]).sum(), (pred_embeddings[4]-pred_embeddings[1]).sum())
        # print(f"Done, {g_pixel_values.device} !!! {num_frames}---{num_objs}, {language_embeddings.shape}\n\n")
        #sam_states = self.grounding_encoder.get_sam2_embeddings(g_pixel_values, expand_size=num_objs)
        
#        pred_masks = self.grounding_encoder.inject_language_embd(sam_states, language_embeddings, nf_nobj=(num_frames, num_objs))
#        self.iteration+=1


#        print("PRED MASK LEN",pred_masks.shape)
#        gt_masks = [F.interpolate(gt_mask.unsqueeze(0), size=pred_masks[0].shape[-2:], mode='nearest').squeeze(0) for gt_mask in gt_masks_video]
#        print("GT MASK LEN",len(gt_masks))
#        gt_masks = torch.cat(gt_masks, dim=0)
#        print("GT MASK LEN",gt_masks.shape)
#        torch.save(pred_masks,f"/home/vipradas/Thesis/viz/{self.iteration}/gt.pth")
#        pred_masks = pred_masks.flatten(0, 1)
        # pred_masks = torch.cat(pred_masks, dim=0)
#        print("GT MASK LEN",gt_masks.shape)


#        bs = len(pred_masks)



        # if self.iteration==0:
        #     self.proj.weight = nn.Parameter(self.proj.weight.to(dtype=torch.bfloat16, device=torch.device('cpu')))
        #     self.proj.bias = nn.Parameter(self.proj.bias.to(dtype=torch.bfloat16, device=torch.device('cpu')))


        self.iteration+=1
        # for _ in range(2):
        #     assert "pooling_parent" in point.keys()
        #     assert "pooling_inverse" in point.keys()
        #     parent = point.pop("pooling_parent")
        #     inverse = point.pop("pooling_inverse")
        #     parent["feat"] = torch.cat([parent["feat"], point["feat"][inverse]], dim=-1)
        #     point = parent
        # while "pooling_parent" in point.keys():
        #     assert "pooling_inverse" in point.keys()
        #     parent = point.pop("pooling_parent")
        #     inverse = point.pop("pooling_inverse")
        #     parent["feat"] = point["feat"][inverse]
        #     point = parent
        # feat = point["feat"][point["inverse"]]
        # locs = point["coord"][point["inverse"]]
        #feat=torch.cat([feat, locs],dim=-1)
        

        # embds=language_embeddings.to(dtype=torch.bfloat16)
        embds=self.lang_projector(language_embeddings).to(dtype=torch.bfloat16)
        #embds=self.channel_reducer(embds).to(dtype=torch.float32)
        #feat = feat.to(dtype=torch.bfloat16)
        # pred_masks = torch.matmul(self.pointencoder(feat),embds.T).to(dtype=torch.bfloat16)
        
        #point=self.lang_projector(point)
        #print(point.shape, embds.shape)

        #pred_masks = torch.matmul(point, embds)
        #print(pred_masks.shape)
        
        sparse_tensor, pts_3d, clrs, fts, unique_map, inverse_map_m3d = prepare_data(point['vertices'], point['colors'], self.device)
            
        with torch.amp.autocast(device_type='cuda', dtype=torch.float32):
            mask3d_outputs = self.segmenter(sparse_tensor, raw_coordinates=fts.to(dtype=torch.float32), lang_query=embds[0]) 

        pred_masks = mask3d_outputs['pred_masks'][0][inverse_map_m3d][:,-1]
        bbox_loss = self.bbox_loss(mask3d_outputs["pred_bboxs"].squeeze(0).squeeze(0), bbx[0][0])
        # logits = outputs["pred_logits"][0]
        # queries = outputs['queries'].squeeze(0)

        # labels = []
        # confidences = []
        # masks_binary = []
        # valid_queries = []

        # for i in range(len(logits)):
        #     p_labels = torch.softmax(logits[i], dim=-1)
        #     p_masks = torch.sigmoid(seg_masks[:, i])
        #     l = torch.argmax(p_labels, dim=-1)
        #     c_label = torch.max(p_labels)
        #     m = p_masks > 0.5
        #     c_m = p_masks[m].sum() / (m.sum() + 1e-8)
        #     c = c_label * c_m
        #     if l < 200 and c > 0.9:
        #         labels.append(l.item())
        #         confidences.append(c.item())
        #         valid_queries.append(queries[i])
        #         masks_binary.append(
        #             seg_masks[:,i][inverse_map_m3d])  
        # valid_queries=torch.stack(valid_queries)

        # similarity = F.cosine_similarity(valid_queries, embds)
        # best_match = torch.argmax(similarity)
        # #print(best_match)
        # #matched_query = queries[0,best_match,:].squeeze(0)
        # matched_query=valid_queries[best_match]
        # pred_masks=masks_binary[best_match]

        # masks=torch.stack(masks_binary)
        # print(masks.shape)
        #pred_masks = pred_masks.repeat(5,1)

        gt_masks = torch.from_numpy(gt).to(device=pred_masks.device,dtype=pred_masks.dtype)
#        np.save(f"/nodes/faxe/work/vipradas/preds/gt_{self.iteration}.npy", gt_masks[0].detach().cpu().numpy())
        # inside_mass= (pred*gt_masks[0]).sum()
        # total_mass=pred.sum()+1e-6
        # purity=inside_mass/total_mass
        # purity_loss=1-purity
        #print(gt_masks.shape)   
        #class_loss , sam_loss_mask, sam_loss_dice, pm = compute_loss(logits, pred_masks, gt_masks[0],self.iteration)
#        if len(pred_masks) != len(gt_masks):
            # drop this data
#            print(f"Pred mask shape {pred_masks.shape} is not equal to gt_mask shape {gt_masks.shape} !!!")
#            min_num = min(len(pred_masks), len(gt_masks))
#            pred_masks = pred_masks[:min_num]
#            gt_masks = gt_masks[:min_num]
#            seg_valid = False

#        if False:
#            sampled_pred_mask, sampled_gt_mask = self.sample_points(pred_masks, gt_masks)
#            sam_loss_dice = self.loss_dice(
#                sampled_pred_mask,
#                sampled_gt_mask, avg_factor=(len(gt_masks) + 1e-4))
#            sam_loss_mask = self.loss_mask(
#                sampled_pred_mask.reshape(-1),
#                sampled_gt_mask.reshape(-1),
#                avg_factor=(pred_masks.shape[0] * sampled_pred_mask.shape[1] + 1e-4))
#        else:
#            sam_loss_mask = self.loss_mask(pred_masks, gt_masks)
#            sam_loss_dice = self.loss_dice(pred_masks, gt_masks)

        


        sam_loss_dice=self.dice(pred_masks,gt_masks[0])
        sam_loss_mask=self.criterion(pred_masks,gt_masks[0])

        #print(pred_mask_probs.shape, gt_masks[0].shape)
        # pred_mask_probs=torch.sigmoid(masks)
        # intersection = (pred_mask_probs.T * gt_masks[0].unsqueeze(1)).sum(dim=0)
        # print(intersection.shape)
        # union = pred_mask_probs.sum(dim=1) + gt_masks[0].sum()
        
        # dice_cost = 1 - (2 * intersection + 1e-3) / (union + 1e-3)
        
        # best_query_idx=torch.argmin(dice_cost)
        # best_query=valid_queries[ best_query_idx]
        # #print(best_query_idx, matched_query.shape, best_query.shape)
        # query_loss=torch.nn.MSELoss()(matched_query, best_query)


        loss_mask=0
        loss_dice=0
        loss_mask += sam_loss_mask
        loss_dice += sam_loss_dice #* purity_loss


        if self.training==False:
            preds=torch.sigmoid(pred_masks)
            preds=preds>0.6
            
            target=gt_masks[0]
            # print(preds,target)
            target= target.bool()
            np.save(f"/home/vipradas/preds_val/pred{self.iteration}.npy", pred_masks.detach().cpu().numpy())
            #torch.save(preds,f"/home/vipradas/gt{self.iteration}.pth")
            intersection = (preds & target).sum()
            union = (preds | target).sum()
            iou = intersection / (union + 1e-7)
            # iou=bbox_iou_3d(target,preds)
            # print(iou_3d(mask3d_outputs["pred_bboxs"].squeeze(0).squeeze(0), bbx[0][0]))

            print(iou)

        if not seg_valid:
            _scale = 0.0
        else:
            _scale = 1.0
        loss_mask = loss_mask * _scale
        loss_dice = loss_dice * _scale

        

        # if self.iteration >= 0 and self.iteration % 10==0:
            
        #     np.save(f"/nodes/faxe/work/vipradas/preds/gtp_{self.iteration}.npy", gt_masks[0].detach().cpu().numpy())
        #     np.save(f"/nodes/faxe/work/vipradas/preds/pred_{self.iteration}.npy", pred_masks.detach().cpu().numpy())
            # np.save(f"/nodes/faxe/work/vipradas/preds/best_{self.iteration}.npy", pred_mask_probs[:,best_query_idx].detach().cpu().numpy())

        if self.iteration % 18332==0:
            torch.save(get_state_dict(self), f"/nodes/faxe/work/vipradas/mask3d_{self.iteration}.pth")

        loss_dict = {
            'loss_mask': loss_mask,
            'loss_dice': loss_dice,
            'llm_loss': output.loss,
            'bbox_loss': bbox_loss
        }

        
        return loss_dict
