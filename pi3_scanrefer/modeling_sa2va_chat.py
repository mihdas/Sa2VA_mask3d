# --------------------------------------------------------
# InternVL
# Copyright (c) 2024 OpenGVLab
# Licensed under The MIT License [see LICENSE for details]
# --------------------------------------------------------

from timm.layers import LayerNorm2d
from torch import nn
import math
from einops import rearrange

import re
import warnings
from typing import Any, List, Optional, Tuple, Union

import torchvision.transforms as T
from torchvision.transforms.functional import InterpolationMode

import torch.utils.checkpoint
import transformers

from .modeling_internlm2 import InternLM2ForCausalLM
from .modeling_phi3 import Phi3ForCausalLM
from peft import LoraConfig, get_peft_model
from torch import nn
from torch.nn import CrossEntropyLoss
from transformers import (AutoModel, GenerationConfig, LlamaForCausalLM,
                          LlamaTokenizer, Qwen2ForCausalLM)
from transformers.modeling_outputs import CausalLMOutputWithPast
from transformers.modeling_utils import PreTrainedModel
from transformers.utils import ModelOutput, logging
from transformers import StoppingCriteriaList, StoppingCriteria

from .configuration_sa2va_chat import Sa2VAChatConfig
from .modeling_intern_vit import InternVisionModel, has_flash_attn

# from .sam2 import SAM2
from .templates import PROMPT_TEMPLATE
from pi3.models.pi3 import Pi3

import numpy as np
from torchvision.transforms.functional import resize, to_pil_image

from types import MethodType
import torch.nn.functional as F


try:
    from .flash_attention import FlashAttention
    has_flash_attn = True
except:
    print('FlashAttention is not installed.')
    has_flash_attn = False

logger = logging.get_logger(__name__)

def version_cmp(v1, v2, op='eq'):
    import operator

    from packaging import version
    op_func = getattr(operator, op)
    return op_func(version.parse(v1), version.parse(v2))

class StopWordStoppingCriteria(StoppingCriteria):
    """StopWord stopping criteria."""

    def __init__(self, tokenizer, stop_word):
        self.tokenizer = tokenizer
        self.stop_word = stop_word
        self.length = len(self.stop_word)

    def __call__(self, input_ids, *args, **kwargs) -> bool:
        cur_text = self.tokenizer.decode(input_ids[0])
        cur_text = cur_text.replace('\r', '').replace('\n', '')
        return cur_text[-self.length:] == self.stop_word

def get_stop_criteria(
    tokenizer,
    stop_words=[],
):
    stop_criteria = StoppingCriteriaList()
    for word in stop_words:
        stop_criteria.append(StopWordStoppingCriteria(tokenizer, word))
    return stop_criteria

class DirectResize:
    def __init__(self, target_length: int) -> None:
        self.target_length = target_length

    def apply_image(self, image: np.ndarray) -> np.ndarray:
        """
        Expects a numpy array with shape HxWxC in uint8 format.
        """
        img = to_pil_image(image, mode='RGB')
        return np.array(img.resize((self.target_length, self.target_length)))


class ScaleBlock(nn.Module):
    def __init__(self, embed_dim, embed_dim_in=None, conv1_layer=nn.ConvTranspose2d):
        super().__init__()

        if embed_dim_in is None:
            embed_dim_in = embed_dim

        self.conv1 = conv1_layer(
            embed_dim_in,
            embed_dim,
            kernel_size=2,
            stride=2,
        )
        self.act = nn.GELU()
        self.conv2 = nn.Conv2d(
            embed_dim,
            embed_dim,
            kernel_size=3,
            padding=1,
            groups=embed_dim,
            bias=False,
        )
        self.norm = LayerNorm2d(embed_dim)

    def forward(self, x):
        x = self.conv1(x)
        x = self.act(x)
        x = self.conv2(x)
        x = self.norm(x)

        return x

class Sa2VAChatModel(PreTrainedModel):
    config_class = Sa2VAChatConfig
    main_input_name = 'pixel_values'
    base_model_prefix = 'language_model'
    _no_split_modules = ['InternVisionModel', 'LlamaDecoderLayer', 'InternLM2DecoderLayer',
                         'Phi3DecoderLayer', 'Qwen2DecoderLayer', 'SAM2']
    _supports_flash_attn_2 = True
    supports_gradient_checkpointing = True

    def __init__(self, config: Sa2VAChatConfig, vision_model=None, language_model=None, use_flash_attn=True):
        super().__init__(config)

        assert version_cmp(transformers.__version__, '4.37.0', 'ge')
        image_size = config.force_image_size or config.vision_config.image_size
        patch_size = config.vision_config.patch_size
        self.patch_size = patch_size
        self.select_layer = config.select_layer
        self.template = config.template
        self.template = self.template.replace('-', '_')
        self.num_image_token = int((image_size // patch_size) ** 2 * (config.downsample_ratio ** 2))
        self.downsample_ratio = config.downsample_ratio
        self.ps_version = config.ps_version
        self.llm_arch_name = config.llm_config.architectures[0]

        use_flash_attn = use_flash_attn if has_flash_attn else False
        config.vision_config.use_flash_attn = True if use_flash_attn else False
        config.llm_config._attn_implementation = 'flash_attention_2' if use_flash_attn else 'eager'

        logger.info(f'num_image_token: {self.num_image_token}')
        logger.info(f'ps_version: {self.ps_version}')
        if vision_model is not None:
            self.vision_model = vision_model
        else:
            self.vision_model = InternVisionModel(config.vision_config)
        if language_model is not None:
            self.language_model = language_model
        else:
            if config.llm_config.architectures[0] == 'LlamaForCausalLM':
                self.language_model = LlamaForCausalLM(config.llm_config)
            elif config.llm_config.architectures[0] == 'InternLM2ForCausalLM':
                self.language_model = InternLM2ForCausalLM(config.llm_config)
            elif config.llm_config.architectures[0] == 'Phi3ForCausalLM':
                self.language_model = Phi3ForCausalLM(config.llm_config)
            elif config.llm_config.architectures[0] == 'Qwen2ForCausalLM':
                self.language_model = Qwen2ForCausalLM(config.llm_config)
            else:
                raise NotImplementedError(f'{config.llm_config.architectures[0]} is not implemented.')

        vit_hidden_size = config.vision_config.hidden_size
        llm_hidden_size = config.llm_config.hidden_size

        self.mlp1 = nn.Sequential(
            nn.LayerNorm(vit_hidden_size * int(1 / self.downsample_ratio) ** 2),
            nn.Linear(vit_hidden_size * int(1 / self.downsample_ratio) ** 2, llm_hidden_size),
            nn.GELU(),
            nn.Linear(llm_hidden_size, llm_hidden_size)
        )

        self.img_context_token_id = None
        self.conv_template = PROMPT_TEMPLATE[self.template]
        self.template = self.conv_template
        if hasattr(config, 'system_message'):
            self.system_message = config.system_message
        self.num_samples = 0

        if config.use_backbone_lora:
            self.wrap_backbone_lora(r=config.use_backbone_lora, lora_alpha=2 * config.use_backbone_lora)

        if config.use_llm_lora:
            self.wrap_llm_lora(r=config.use_llm_lora, lora_alpha=2 * config.use_llm_lora)

        self.mask_decoder = None
        # self.grounding_encoder = SAM2()
        # self.grounding_encoder = SAM2(ckpt_path="/p/scratch/llmvidseg/alexey/saved/sam2/sam2_hiera_large.pt")
        # out_dim = self.grounding_encoder.hidden_dim
        out_dim = 256
        in_dim = llm_hidden_size
        self.text_hidden_fcs = nn.Sequential(
            nn.Linear(in_dim, in_dim), nn.ReLU(inplace=True),
            nn.Linear(in_dim, out_dim), nn.Dropout(0.0)
        )
        # '''
        # self.mask_head = nn.Sequential(
        #     nn.Linear(256, 512),
        #     nn.GELU(),
        #     nn.Linear(512, 512),
        #     nn.GELU(),
        #     nn.Linear(512, 256),
        # )
        # #print("MASK_HEAD: 512x512")
        # '''
        # self.mask_head = nn.Sequential(
        #     nn.Linear(896, 256),
        #     nn.GELU(),
        #     nn.Linear(256, 256),
        #     nn.GELU(),
        #     nn.Linear(256, 256),
        # )
        #
        # patch_size = (28, 28)
        # num_upscale = 4
        # # self.upscale = nn.Sequential(
        # #     *[ScaleBlock(256, 896) for _ in range(num_upscale)],
        # # )
        #
        # upscale_blocks = [ScaleBlock(256, embed_dim_in=896)]
        # upscale_blocks.extend([ScaleBlock(256) for _ in range(num_upscale - 1)])
        # self.upscale = nn.Sequential(*upscale_blocks)

        self.init_prediction_config = False

        self.mask_head = nn.Sequential(
            nn.Linear(llm_hidden_size, 256),
            nn.GELU(),
            nn.Linear(256, 256),
            nn.LayerNorm(256),
            # nn.GELU(),
            # nn.Linear(256, 256),
        )

        self.spatial_model = Pi3().load_model(
            mode="eval", path="/nodes/cristal/fastwork/nekrasov/saved/pi3/model.safetensors",
        )
        for p in self.spatial_model.parameters():
            p.requires_grad_(False)
        self.pi3_upscale = ScaleBlock(
            self.spatial_model.dec_embed_dim * 2, embed_dim_in=llm_hidden_size
        )
        self.mlp2 = nn.Sequential(
            nn.LayerNorm(self.spatial_model.dec_embed_dim * 2),
            nn.Linear(self.spatial_model.dec_embed_dim * 2, llm_hidden_size),
            nn.GELU(),
            nn.Linear(llm_hidden_size, llm_hidden_size),
        )
        # self.mlp2.to(device=emb.device, dtype=emb.dtype)

        # patch_size = (28, 28)
        # max_patch_size = max(patch_size[0], patch_size[1])
        # num_upscale = max(1, int(math.log2(max_patch_size)) - 2)
        num_upscale = 3
        self.num_tokens_per_expression = 4

        upscale_blocks = [
            ScaleBlock(256, embed_dim_in=self.spatial_model.dec_embed_dim * 2)
        ]
        upscale_blocks.extend([ScaleBlock(256) for _ in range(num_upscale - 1)])
        self.upscale = nn.Sequential(*upscale_blocks)
        # self.step = 0
        # self.img_token_idx = self.tokenizer(
        #     "<IMG_CONTEXT>", add_special_tokens=False
        # ).input_ids[0]

    # def _get_pi3(self):
    # 	#download checkpoints from `https://huggingface.co/yyfz233/Pi3/resolve/main/model.safetensors`, and `--ckpt ckpts/model.safetensors
    #     self.spatial_model = Pi3().to("cuda").eval()
    #     from safetensors.torch import load_file
    #     weight = load_file("/home/burdorf/Downloads/model.safetensors")
    #     self.spatial_model.load_state_dict(weight)
    #     for p in self.spatial_model.parameters():
    #         p.requires_grad_(False)
    #     return self.spatial_model

    def wrap_backbone_lora(self, r=128, lora_alpha=256, lora_dropout=0.05):
        lora_config = LoraConfig(
            r=r,
            target_modules=['attn.qkv', 'attn.proj', 'mlp.fc1', 'mlp.fc2'],
            lora_alpha=lora_alpha,
            lora_dropout=lora_dropout,
        )
        self.vision_model = get_peft_model(self.vision_model, lora_config)
        self.vision_model.print_trainable_parameters()

    def wrap_llm_lora(self, r=128, lora_alpha=256, lora_dropout=0.05):
        # Determine the target modules based on the architecture of the language model
        if self.llm_arch_name == 'InternLM2ForCausalLM':
            target_modules = ['attention.wqkv', 'attention.wo', 'feed_forward.w1', 'feed_forward.w2', 'feed_forward.w3']
        elif self.llm_arch_name == 'Phi3ForCausalLM':
            target_modules = ['mlp.down_proj', 'mlp.gate_up_proj', 'self_attn.o_proj', 'self_attn.qkv_proj']
        elif self.llm_arch_name in ['Qwen2ForCausalLM', 'LlamaForCausalLM']:
            target_modules = ['self_attn.q_proj', 'self_attn.k_proj', 'self_attn.v_proj', 'self_attn.o_proj',
                              'mlp.gate_proj', 'mlp.down_proj', 'mlp.up_proj']
        else:
            raise NotImplemented
        lora_config = LoraConfig(
            r=r,
            target_modules=target_modules,
            lora_alpha=lora_alpha,
            lora_dropout=lora_dropout,
            task_type='CAUSAL_LM'
        )
        self.language_model = get_peft_model(self.language_model, lora_config)
        self.language_model.enable_input_require_grads()
        self.language_model.print_trainable_parameters()

    def pixel_shuffle(self, x, scale_factor=0.5):
        n, w, h, c = x.size()
        # N, W, H, C --> N, W, H * scale, C // scale
        x = x.view(n, w, int(h * scale_factor), int(c / scale_factor))
        # N, W, H * scale, C // scale --> N, H * scale, W, C // scale
        x = x.permute(0, 2, 1, 3).contiguous()
        # N, H * scale, W, C // scale --> N, H * scale, W * scale, C // (scale ** 2)
        x = x.view(n, int(h * scale_factor), int(w * scale_factor),
                   int(c / (scale_factor * scale_factor)))
        if self.ps_version == 'v1':
            warnings.warn("In ps_version 'v1', the height and width have not been swapped back, "
                          'which results in a transposed image.')
        else:
            x = x.permute(0, 2, 1, 3).contiguous()
        return x

    def extract_feature(self, pixel_values):
        if self.select_layer == -1:
            vit_embeds = self.vision_model(
                pixel_values=pixel_values,
                output_hidden_states=False,
                return_dict=True).last_hidden_state
        else:
            vit_embeds = self.vision_model(
                pixel_values=pixel_values,
                output_hidden_states=True,
                return_dict=True).hidden_states[self.select_layer]
        vit_embeds = vit_embeds[:, 1:, :]

        h = w = int(vit_embeds.shape[1] ** 0.5)
        vit_embeds = vit_embeds.reshape(vit_embeds.shape[0], h, w, -1)
        vit_embeds = self.pixel_shuffle(vit_embeds, scale_factor=self.downsample_ratio)
        vit_embeds = vit_embeds.reshape(vit_embeds.shape[0], -1, vit_embeds.shape[-1])
        vit_embeds = self.mlp1(vit_embeds)
        return vit_embeds

    @torch.no_grad()
    def extract_3d_feature(self, pixel_values):
        _, _, H, W = pixel_values.shape
        # pixel_values = (pixel_values - self.spatial_model.image_mean) / self.spatial_model.image_std
        hidden = self.spatial_model.encoder(pixel_values, is_training=True)
        if isinstance(hidden, dict):
            hidden = hidden["x_norm_patchtokens"]
        embeds_3d, pos = self.spatial_model.decode(hidden, N=1, H=H, W=W)
        B, _, C = embeds_3d.shape
        embeds_3d = embeds_3d[:, 5:, :].permute(0, 2, 1).reshape(B, C, H // 14, W // 14)
        return embeds_3d

    @property
    def lm_head(self):
        return self.language_model.get_output_embeddings()

    def get_input_embeddings(self):
        return self.language_model.get_input_embeddings()

    def get_output_embeddings(self):
        return self.language_model.get_output_embeddings()

    def forward(self, data, data_samples=None, mode='loss'):
        pixel_values = data['pixel_values']

        if type(pixel_values) is list or pixel_values.ndim == 5:
            if type(pixel_values) is list:
                pixel_values = [
                    x.unsqueeze(0) if x.ndim == 3 else x for x in pixel_values
                ]
            # b*n, c, h, w
            concat_images = torch.cat(
                [image.to(self.vision_model.dtype) for image in pixel_values], dim=0)
        else:
            raise NotImplementedError()

        input_ids = data['input_ids']
        position_ids = data['position_ids']
        attention_mask = data['attention_mask']
        # sum is 0 are text
        image_flags = torch.sum(concat_images, dim=(1, 2, 3)) != 0
        image_flags = image_flags.long()

        labels = data['labels']
        use_cache = False

        if 'vp_overall_mask' not in data.keys():
            vp_overall_mask = None
        else:
            vp_overall_mask = data['vp_overall_mask']

        if 'prompt_masks' in data.keys():
            prompt_masks = data['prompt_masks']
        else:
            prompt_masks = None

        outputs = self._llm_forward(
            input_ids=input_ids,
            position_ids=position_ids,
            attention_mask=attention_mask,
            image_flags=image_flags,
            pixel_values=concat_images,
            labels=labels,
            use_cache=use_cache,
            output_hidden_states=True,
            vp_overall_mask=vp_overall_mask,
            prompt_masks=prompt_masks,
        )

        return outputs

    def _llm_forward(
            self,
            pixel_values: torch.FloatTensor,
            input_ids: torch.LongTensor = None,
            attention_mask: Optional[torch.Tensor] = None,
            position_ids: Optional[torch.LongTensor] = None,
            image_flags: Optional[torch.LongTensor] = None,
            past_key_values: Optional[List[torch.FloatTensor]] = None,
            labels: Optional[torch.LongTensor] = None,
            use_cache: Optional[bool] = None,
            output_attentions: Optional[bool] = None,
            output_hidden_states: Optional[bool] = None,
            return_dict: Optional[bool] = None,
            vp_overall_mask=None,
            prompt_masks=None,
    ) -> Union[Tuple, CausalLMOutputWithPast]:
        return_dict = return_dict if return_dict is not None \
            else self.config.use_return_dict

        image_flags = image_flags.squeeze(-1)
        # We only added the clone code here to avoid the error.
        input_embeds = self.language_model.get_input_embeddings()(
            input_ids).clone()

        vit_embeds = self.extract_feature(pixel_values)
        vit_embeds = vit_embeds.to(input_embeds.dtype)  # FIXME: why vit_embeds is float16?

        B, N, C = input_embeds.shape
        input_embeds = input_embeds.reshape(B * N, C)

        self._count += 1

        if vp_overall_mask is not None and prompt_masks is not None:
            vp_embeds = []
            vp_overall_mask = vp_overall_mask.to(vit_embeds.device).bool()
            prompt_masks = [item.to(vit_embeds.device).bool() for item in prompt_masks]

            vp_overall_mask = vp_overall_mask[image_flags == 1]
            overall_tile_vit_embeds = vit_embeds[vp_overall_mask]  # (n_img, hw, c)

            i_vp_img = 0
            for i_img in range(len(vit_embeds)):
                vp_embeds.append(vit_embeds[i_img].reshape(-1, C))
                if vp_overall_mask[i_img]:
                    tile_vit_embeds = overall_tile_vit_embeds[i_vp_img].reshape(-1, C)  # (hw, C)
                    objects_prompt_masks = prompt_masks[i_vp_img]
                    n_obj = len(objects_prompt_masks)
                    tile_vit_embeds = tile_vit_embeds.unsqueeze(0).repeat(n_obj, 1, 1)
                    objects_prompt_masks = objects_prompt_masks.reshape(n_obj, -1)
                    vp_embeds.append(tile_vit_embeds[objects_prompt_masks])
                    i_vp_img += 1
            vp_embeds = torch.cat(vp_embeds, dim=0)
        else:
            vp_embeds = None

        input_ids = input_ids.reshape(B * N)
        selected = (input_ids == self.img_context_token_id)

        if vp_embeds is None:
            try:
                input_embeds[selected] = vit_embeds.reshape(-1, C)
            except Exception as e:
                vit_embeds = vit_embeds.reshape(-1, C)
                print(f'warning: {e}, input_embeds[selected].shape='
                      f'{input_embeds[selected].shape}, '
                      f'vit_embeds.shape={vit_embeds.shape}')
                n_token = selected.sum()
                if n_token > len(vit_embeds):
                    print(f"Wrong !!! {n_token} image tokens in text but only {len(vit_embeds)} vit embeds !!!")
                    expand_ratio = n_token // len(vit_embeds) + 1
                    vit_embeds = torch.cat([vit_embeds] * expand_ratio, dim=0)

                input_embeds[selected] = vit_embeds[:n_token]
        else:
            try:
                input_embeds[selected] = vp_embeds.reshape(-1, C)
            except Exception as e:
                vp_embeds = vp_embeds.reshape(-1, C)
                print(f'warning: {e}, input_embeds[selected].shape='
                      f'{input_embeds[selected].shape}, '
                      f'vp_embeds.shape={vp_embeds.shape}')
                n_token = selected.sum()
                if n_token > len(vp_embeds):
                    print(f"Wrong !!! {n_token} image tokens in text but only {len(vp_embeds)} vit embeds !!!")
                    expand_ratio = n_token // len(vp_embeds) + 1
                    vp_embeds = torch.cat([vp_embeds] * expand_ratio, dim=0)

                input_embeds[selected] = vp_embeds[:n_token]

        input_embeds = input_embeds.reshape(B, N, C)

        outputs = self.language_model(
            inputs_embeds=input_embeds,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )
        logits = outputs.logits

        loss = None
        if labels is not None:
            # Shift so that tokens < n predict n
            shift_logits = logits[..., :-1, :].contiguous()
            shift_labels = labels[..., 1:].contiguous()
            # Flatten the tokens
            loss_fct = CrossEntropyLoss()
            shift_logits = shift_logits.view(
                -1, self.language_model.config.vocab_size)
            shift_labels = shift_labels.view(-1)
            # Enable model parallelism
            shift_labels = shift_labels.to(shift_logits.device)
            loss = loss_fct(shift_logits, shift_labels)

        if not return_dict:
            output = (logits,) + outputs[1:]
            return (loss,) + output if loss is not None else output

        return CausalLMOutputWithPast(
            loss=loss,
            logits=logits,
            past_key_values=outputs.past_key_values,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )

    @torch.no_grad()
    def generate(
            self,
            pixel_values: Optional[torch.FloatTensor] = None,
            input_ids: Optional[torch.FloatTensor] = None,
            attention_mask: Optional[torch.LongTensor] = None,
            visual_features: Optional[torch.FloatTensor] = None,
            generation_config: Optional[GenerationConfig] = None,
            output_hidden_states: Optional[bool] = None,
            return_dict: Optional[bool] = None,
            prompt_masks=None,
            vp_overall_mask=None,
            **generate_kwargs,
    ) -> torch.LongTensor:
        device = self.device
        assert self.img_context_token_id is not None

        if pixel_values is not None:
            # else:
            if type(pixel_values) is list or pixel_values.ndim == 5:
                if type(pixel_values) is list:
                    pixel_values = [
                        x.unsqueeze(0) if x.ndim == 3 else x for x in pixel_values
                    ]
                # b*n, c, h, w
                pixel_values = torch.cat(
                    [image.to(self.vision_model.dtype) for image in pixel_values], dim=0)

            vit_embeds = self.extract_feature(pixel_values.to(device))

            image_flags = torch.sum(pixel_values, dim=(1, 2, 3)) != 0
            image_flags = image_flags.long()
            vit_embeds = vit_embeds[image_flags == 1]

            if visual_features is not None:
                vit_embeds = vit_embeds + visual_features

            input_embeds = self.language_model.get_input_embeddings()(input_ids.to(device))
            B, N, C = input_embeds.shape
            input_embeds = input_embeds.reshape(B * N, C)

            if vp_overall_mask is not None and prompt_masks is not None:
                vp_embeds = []
                vp_overall_mask = vp_overall_mask.to(vit_embeds.device).bool()
                prompt_masks = [item.to(vit_embeds.device).bool() for item in prompt_masks]

                vp_overall_mask = vp_overall_mask[image_flags == 1]
                overall_tile_vit_embeds = vit_embeds[vp_overall_mask]  # (n_img, hw, c)

                i_vp_img = 0
                for i_img in range(len(vit_embeds)):
                    vp_embeds.append(vit_embeds[i_img].reshape(-1, C))
                    if vp_overall_mask[i_img]:
                        tile_vit_embeds = overall_tile_vit_embeds[i_vp_img].reshape(-1, C)  # (hw, C)
                        objects_prompt_masks = prompt_masks[i_vp_img]
                        n_obj = len(objects_prompt_masks)
                        tile_vit_embeds = tile_vit_embeds.unsqueeze(0).repeat(n_obj, 1, 1)
                        objects_prompt_masks = objects_prompt_masks.reshape(n_obj, -1)
                        vp_embeds.append(tile_vit_embeds[objects_prompt_masks])
                        i_vp_img += 1

                vp_embeds = torch.cat(vp_embeds, dim=0)
            else:
                vp_embeds = None

            input_ids = input_ids.reshape(B * N)
            selected = (input_ids == self.img_context_token_id)
            assert selected.sum() != 0
            if vp_embeds is None:
                input_embeds[selected] = vit_embeds.reshape(-1, C).to(input_embeds.device)
            else:
                if len(input_embeds[selected]) != len(vp_embeds.reshape(-1, C)):
                    print("Shape mismatch, selected is {}, vp embeds is {} !!!" \
                          .format(len(input_embeds[selected]), len(vp_embeds.reshape(-1, C))))
                    min_tokens = min(len(input_embeds[selected]), len(vp_embeds.reshape(-1, C)))
                    input_embeds[selected][:min_tokens] = vp_embeds.reshape(-1, C)[:min_tokens].to(input_embeds.device)
                else:
                    input_embeds[selected] = vp_embeds.reshape(-1, C).to(input_embeds.device)

            input_embeds = input_embeds.reshape(B, N, C)
        else:
            input_embeds = self.language_model.get_input_embeddings()(input_ids)

        outputs = self.language_model.generate(
            inputs_embeds=input_embeds,
            attention_mask=attention_mask.to(device),
            generation_config=generation_config,
            output_hidden_states=output_hidden_states,
            # return_dict=return_dict,
            use_cache=True,
            **generate_kwargs,
        )

        return outputs

    def preparing_for_generation(self, tokenizer, max_new_tokens=2048, torch_dtype=torch.bfloat16):
        # set stop criteria and generation configs for model
        if not hasattr(self, 'tokenizer'):
            self.tokenizer = tokenizer
        self.bot_name = 'BOT'
        stop_words = []
        stop_words += self.template.get('STOP_WORDS', [])
        stop_criteria = get_stop_criteria(
            tokenizer=self.tokenizer, stop_words=stop_words)
        self.stop_criteria = stop_criteria

        default_generation_kwargs = dict(
            max_new_tokens=max_new_tokens,
            do_sample=False,
            eos_token_id=self.tokenizer.eos_token_id,
            pad_token_id=(
                self.tokenizer.pad_token_id
                if self.tokenizer.pad_token_id is not None
                else self.tokenizer.eos_token_id
            ),
        )

        self.gen_config = GenerationConfig(**default_generation_kwargs)
        self.init_prediction_config = True
        self.torch_dtype = torch_dtype
        self.to(torch_dtype)
        self.extra_image_processor = DirectResize(target_length=1024, )
        # for multi image process
        self.min_dynamic_patch = 1
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

        self.transformer = T.Compose([
            T.Lambda(lambda img: img.convert('RGB') if img.mode != 'RGB' else img),
            T.Resize((self.image_size, self.image_size), interpolation=InterpolationMode.BICUBIC),
            T.ToTensor(),
            T.Normalize(mean=self.IMAGENET_MEAN, std=self.IMAGENET_STD)
        ])
        self.VP_START_TOKEN = '<vp>'
        self.VP_END_TOKEN = '</vp>'

        # change phi3 prepare for generation fuction
        if self.config.llm_config.architectures[0] == 'Phi3ForCausalLM':
            self.language_model.prepare_inputs_for_generation = MethodType(prepare_inputs_for_generation_phi3, self.language_model)

        img_context_token_id = tokenizer.convert_tokens_to_ids('<IMG_CONTEXT>')
        self.img_context_token_id = img_context_token_id
        self.seg_token_idx = tokenizer.convert_tokens_to_ids('[SEG]')
        return

    def predict_forward(
            self,
            image=None,
            video=None,
            text=None,
            past_text='',
            mask_prompts=None,
            tokenizer=None,
            uniform_sampling=False,
            training_like=True,
            sample_num_frames=8,
    ):
        if not self.init_prediction_config:
            assert tokenizer
            self.preparing_for_generation(tokenizer=tokenizer)

        if image is None and video is None and '<image>' not in past_text:
            text = text.replace('<image>', "")
            input_text = ''
            input_text += self.template['INSTRUCTION'].format(
                input=text, round=1, bot_name=self.bot_name)
            input_text = past_text + input_text
            ids = self.tokenizer.encode(input_text)
            ids = torch.tensor(ids).cuda().unsqueeze(0)

            attention_mask = torch.ones_like(ids, dtype=torch.bool)

            mm_inputs = {
                'pixel_values': None,
                'input_ids': ids,
                'attention_mask': attention_mask,
                'position_ids': None,
                'past_key_values': None,
                'labels': None,
                'prompt_masks': None,
                'vp_overall_mask': None,
            }
            ret_masks = []
        else:
            input_dict = {}
            if video is not None:

                pixel_values = []
                # extra_pixel_values = []
                ori_image_size = video[0].size
                # Number of frames in the video
                num_frames = len(video)
                # Generate 5 (or fewer, if the video is short) uniformly spaced indices
                # if num_frames <= sample_num_frames:
                #     sample_idx = list(range(num_frames))
                # else:
                sample_idx = np.linspace(0, num_frames - 1, sample_num_frames, dtype=int).tolist()
                for frame_idx, frame_image in enumerate(video):
                    # Ensure all frames have the same size
                    assert ori_image_size == frame_image.size
                    # Convert frame to a NumPy array for extra processing
                    # g_image = np.array(frame_image)  # grounding image
                    # g_image = self.extra_image_processor.apply_image(g_image)
                    # g_image = torch.from_numpy(g_image).permute(2, 0, 1).contiguous()
                    # extra_pixel_values.append(g_image)
                    # Instead of `if frame_idx < 5:`, use uniform sampling
                    if frame_idx in sample_idx:
                        # Apply your transformer (e.g. CLIP transform)
                        img = self.transformer(frame_image)
                        pixel_values.append(img)

                pixel_values = torch.stack(pixel_values, dim=0).to(self.torch_dtype)  # (n_f, 3, h, w)
                # g_pixel_values = torch.stack([
                #     self.grounding_encoder.preprocess_image(pixel) for pixel in extra_pixel_values
                # ]).to(self.torch_dtype)
                num_image_tokens = self.patch_token
                num_frames = len(pixel_values)

                input_dict['vp_overall_mask'] = None
            else:
                ori_image_size = image.size

                # prepare grounding images
                g_image = np.array(image)  # for grounding
                g_image = self.extra_image_processor.apply_image(g_image)
                # g_pixel_values = torch.from_numpy(g_image).permute(2, 0, 1).contiguous().to(self.torch_dtype)
                # extra_pixel_values = [g_pixel_values]
                # g_pixel_values = torch.stack([
                #     self.grounding_encoder.preprocess_image(pixel) for pixel in extra_pixel_values
                # ]).to(self.torch_dtype)

                images = dynamic_preprocess(image, self.min_dynamic_patch,
                                            self.max_dynamic_patch,
                                            self.image_size, self.use_thumbnail)

                if mask_prompts is not None:
                    vp_overall_mask = torch.Tensor([False] * (len(images) - 1) + [True])
                    input_dict['vp_overall_mask'] = vp_overall_mask
                else:
                    input_dict['vp_overall_mask'] = None

                pixel_values = [self.transformer(image) for image in images]
                pixel_values = torch.stack(pixel_values).to(self.torch_dtype)
                num_image_tokens = pixel_values.shape[0] * self.patch_token
                num_frames = 1
                sample_idx = [0,]
            # input_dict['g_pixel_values'] = g_pixel_values
            input_dict['pixel_values'] = pixel_values

            if mask_prompts is not None:
                # reshape mask prompts to feature size
                mask_prompts = [torch.Tensor(item).to(pixel_values.device) for item in mask_prompts]
                mask_prompts = [F.interpolate(
                    item.unsqueeze(0),
                    size=(int(self.image_size // self.patch_size * self.downsample_ratio),
                          int(self.image_size // self.patch_size * self.downsample_ratio)),
                    mode='nearest').squeeze(0) for item in mask_prompts]
                region_pixels = []
                for mask_prompt in mask_prompts[0]:
                    region_pixels.append(mask_prompt.bool().to(torch.int64).sum())

                vp_token_str = '\nThere are {} part regions in the picture: '.format(len(mask_prompts[0]))
                for i in range(len(mask_prompts[0])):
                    vp_token_str = vp_token_str + \
                                   f"region{i + 1}" + self.VP_START_TOKEN + \
                                   self.IMG_CONTEXT_TOKEN * region_pixels[i] + \
                                   self.VP_END_TOKEN
                    if i == len(mask_prompts[0]) - 1:
                        vp_token_str = vp_token_str + '.\n'
                    else:
                        vp_token_str = vp_token_str + ', '
            else:
                vp_token_str = ''

            image_token_str = f'{self.IMG_START_TOKEN}' \
                              f'{self.IMG_CONTEXT_TOKEN * num_image_tokens}' \
                              f'{self.IMG_END_TOKEN}'
            image_token_str = image_token_str + '\n'
            image_token_str = image_token_str * num_frames
            image_token_str = image_token_str.strip()

            ret_masks = []

            if '<image>' in text or mask_prompts is not None:
                assert past_text is None or len(past_text) == 0
            text = text.replace('<image>', image_token_str + vp_token_str)
            input_text = ''
            input_text += self.template['INSTRUCTION'].format(
                input=text, round=1, bot_name=self.bot_name)
            input_text = past_text + input_text
            ids = self.tokenizer.encode(input_text)
            ids = torch.tensor(ids).cuda().unsqueeze(0)

            attention_mask = torch.ones_like(ids, dtype=torch.bool)

            mm_inputs = {
                'pixel_values': input_dict['pixel_values'],
                'input_ids': ids,
                'attention_mask': attention_mask,
                'position_ids': None,
                'past_key_values': None,
                'labels': None,
                'prompt_masks': mask_prompts,
                'vp_overall_mask': input_dict['vp_overall_mask'],
            }

        pixel_values = mm_inputs.get("pixel_values", None)
        if pixel_values is not None:
            if type(pixel_values) is list or pixel_values.ndim == 5:
                if type(pixel_values) is list:
                    pixel_values = [
                        x.unsqueeze(0) if x.ndim == 3 else x for x in pixel_values
                    ]
                pixel_values = torch.cat([image for image in pixel_values], dim=0)
            mm_inputs["pixel_values"] = pixel_values.to(device=self.device, dtype=self.torch_dtype)

            spatial_features = self.extract_3d_feature(
                mm_inputs["pixel_values"]
            )
            B, C, _, _ = spatial_features.shape
            vit_height = vit_width = 16
            resized_spatial = torch.nn.functional.interpolate(
                spatial_features,
                size=(vit_height, vit_width),
                mode="bilinear",
                align_corners=False,
            )
            # Reshape back to patch format: [B, vit_patch_count, C]
            embeds_3d = resized_spatial.reshape(B, C, 256).permute(0, 2, 1)
            embeds_3d = self.mlp2(embeds_3d)
            mm_inputs["visual_features"] = embeds_3d

        generate_output = self.generate(
            **mm_inputs,
            generation_config=self.gen_config,
            streamer=None,
            bos_token_id=self.tokenizer.bos_token_id,
            stopping_criteria=self.stop_criteria,
            output_hidden_states=True,
            return_dict_in_generate=True
        )
        predict = self.tokenizer.decode(
            generate_output.sequences[0], skip_special_tokens=False).strip()

        if image is None and video is None and '<image>' not in past_text:
            return {'prediction': predict, 'prediction_masks': ret_masks, }

        # if have seg result, find the seg hidden states
        hidden_states = generate_output.hidden_states

        last_hidden_states = [item[-1][0] for item in hidden_states]
        last_hidden_states = torch.cat(last_hidden_states, dim=0)
        seg_hidden_states = get_seg_hidden_states(
            last_hidden_states, generate_output.sequences[0][:-1],
            seg_id=self.seg_token_idx
        )
        g_pixel_values = hidden_states[0][-1][mm_inputs["input_ids"] == self.img_context_token_id]
        #g_pixel_values = self.text_hidden_fcs(g_pixel_values)
        num_frames_llm = max(1, int(len(g_pixel_values) / 256))


        g_pixel_values = g_pixel_values.reshape(1, len(mm_inputs["pixel_values"]), 16, 16, -1)
        g_pixel_values = rearrange(g_pixel_values, "b t h w c -> (b t) c h w")
        g_pixel_values = self.pi3_upscale(g_pixel_values)
        g_pixel_values = torch.nn.functional.interpolate(
            g_pixel_values,
            size=(spatial_features.shape[-2], spatial_features.shape[-1]),
            mode="bilinear",
            align_corners=False,
        )
        if g_pixel_values.shape == spatial_features.shape:
            g_pixel_values = g_pixel_values + spatial_features
        g_pixel_values = self.upscale(g_pixel_values)
        g_pixel_values = rearrange(
            g_pixel_values, "(b t) c h w -> b c t h w", b=1, t=num_frames_llm
        )
        pred_embeddings_list_ = seg_hidden_states.unsqueeze(1)
        # pred_embeddings_list_ = pred_embeddings_list_[0].transpose(0, 1)
        pred_masks = torch.einsum(
            "bqc,bcthw->qthw", self.mask_head(pred_embeddings_list_), g_pixel_values
        )
        masks = pred_masks[0] / self.num_tokens_per_expression

        # g_pixel_values = g_pixel_values.reshape(
        #     1, num_frames_llm, 16, 16, -1
        # )
        # g_pixel_values = self.upscale(rearrange(g_pixel_values, "b t h w c -> (b t) c h w"))
        # g_pixel_values = rearrange(g_pixel_values, "(b t) c h w -> b c t h w", b=1, t=num_frames_llm)
        # #pred_embeddings_list_ = self.text_hidden_fcs(seg_hidden_states).unsqueeze(0)
        # pred_embeddings_list_ = seg_hidden_states
        # print(seg_hidden_states.shape)
        # masks = torch.einsum(
        #     "bqc,bcthw->bqthw", self.mask_head(pred_embeddings_list_), g_pixel_values
        # )[0, 0, -num_frames:] # b and q are 1

        W, H = ori_image_size
        # masks = []
        # for mask in mask_tokens:
        #     mask = ids_to_mask_256(mask, self.mask_decoder)
        #     masks.append(mask)
        #
        # if len(pixel_values) != len(masks):
        #     return {'prediction': predict, 'prediction_masks': [] }

        # masks = torch.stack(masks)# if len(masks) > 1 else masks

        # if masks.shape[-2:] != (H, W):
        #     prob_M = F.interpolate(masks, size=(H,W), mode="bilinear", align_corners=False)[0,0].detach().cpu().numpy().astype(np.uint8)
        #     # M = max(H,W)
        #     # prob_M = F.interpolate(masks, size=(M,M), mode="bilinear", align_corners=False)[0,0].detach().cpu().numpy().astype(np.uint8)
        #     masks = prob_M[:H,:W]

        if num_frames != num_frames_llm:
            # masks_for_viz = pred_masks.clone().detach().float().cpu().numpy()
            # with open("notebooks/prompt.txt", "w") as file:
            #     file.write(
            #         self.tokenizer.decode(mm_inputs["input_ids"][0]).replace(
            #             "<IMG_CONTEXT>", ""
            #         )
            #     )
            # np.save("notebooks/mask.npy", masks_for_viz)
            # np.save(
            #     "notebooks/image.npy", input_dict["pixel_values"].float().cpu().numpy()
            # )
            masks = combine_masks(masks.cpu(), ori_image_size)
            # np.save("notebooks/refined_mask.npy", masks.cpu().numpy())
        else:
            # processing for video
            W, H = ori_image_size
            masks = F.interpolate(
                masks.unsqueeze(1), size=(H, W), mode="bilinear", align_corners=False
            )

        # pred_masks = self.grounding_encoder.train_postprocess(g_pixel_values, masks, sample_idx)
        # masks = F.interpolate(pred_masks, size=(H,W), mode='bilinear', align_corners=False)
        # masks = masks[:, 0]
        masks = masks.sigmoid() > 0.5
        # masks = masks.cpu().numpy()
        # ret_masks.append(masks)
        ret_masks = masks.cpu().numpy()

        return_dict = {'prediction': predict, 'prediction_masks': ret_masks}
        if training_like:
            return_dict["sample_idx"] = sample_idx
        return return_dict

def combine_masks(
    processed_masks,
    original_size,
    image_size=448,
    min_num=1,
    max_num=12,
    use_thumbnail=True,
):
    """
    Combine cropped masks back into a single mask.

    Args:
        processed_masks: List of binary masks as numpy arrays
        original_size: Tuple of (width, height) of the original mask
        image_size: Size of each processed mask
        min_num: Minimum number of output masks (used to determine grid)
        max_num: Maximum number of output masks (used to determine grid)
        use_thumbnail: Whether the last mask is a thumbnail

    Returns:
        Combined mask as numpy array
    """
    # If use_thumbnail is True, the last mask is a thumbnail and should be excluded
    if use_thumbnail and len(processed_masks) > 1:
        cropped_masks = processed_masks[:-1]
        thumbnail = processed_masks[-1]
    else:
        cropped_masks = processed_masks

    mask_image_width, mask_image_height = processed_masks.shape[-2:]

    # Determine the grid layout
    num_masks = len(cropped_masks)

    # Find the closest aspect ratio to determine the grid layout
    orig_width, orig_height = original_size
    aspect_ratio = orig_width / orig_height

    # Calculate the target ratios (same logic as dynamic_preprocess_mask)
    target_ratios = {
        (i, j)
        for n in range(min_num, max_num + 1)
        for i in range(1, n + 1)
        for j in range(1, n + 1)
        if i * j <= max_num and i * j >= min_num
    }
    target_ratios = sorted(target_ratios, key=lambda x: x[0] * x[1])

    # Find the closest aspect ratio to the target
    target_aspect_ratio = find_closest_aspect_ratio(
        aspect_ratio, target_ratios, orig_width, orig_height, image_size
    )

    # Calculate the target width and height
    target_width = mask_image_width * target_aspect_ratio[0]
    target_height = mask_image_height * target_aspect_ratio[1]
    blocks = target_aspect_ratio[0] * target_aspect_ratio[1]

    # Verify that the number of masks matches the expected blocks
    if num_masks != blocks:
        raise ValueError(
            f"Number of masks ({num_masks}) doesn't match expected blocks ({blocks})"
        )

    # Create an empty canvas for the combined mask
    combined_mask = torch.zeros((target_height, target_width))

    # Place each cropped mask back into its position
    for i, mask in enumerate(cropped_masks):
        # Calculate the position of this mask in the grid
        row = i // target_aspect_ratio[0]
        col = i % target_aspect_ratio[0]

        # Calculate the coordinates in the combined mask
        y_start = row * mask.shape[-2]
        y_end = y_start + mask.shape[-2]
        x_start = col * mask.shape[-1]
        x_end = x_start + mask.shape[-1]

        # Place the mask in the combined mask
        combined_mask[y_start:y_end, x_start:x_end] = mask

    # Convert back to numpy array and binarize
    combined_mask = F.interpolate(
        combined_mask[None, None],
        size=(orig_height, orig_width),
        mode="bilinear",
        align_corners=False,
    )
    thumbnail = F.interpolate(
        thumbnail[None, None],
        size=(orig_height, orig_width),
        mode="bilinear",
        align_corners=False,
    )
    combined_mask = (combined_mask + thumbnail) / 2
    return combined_mask




def get_tokens(full_input_string, pad_value=0):
    """
    Extracts sequences of integer tokens from within each <|mt_start|>...<|mt_end|> block.
    Pads these sequences to the maximum length and returns a single 2D torch.LongTensor.

    Args:
        full_input_string (str): The complete string containing one or more
                                 <|mt_start|>...<|mt_end|> blocks.
        pad_value (int): The value to use for padding shorter sequences. Defaults to 0.

    Returns:
        torch.LongTensor: A 2D tensor of shape (num_blocks, max_sequence_length),
                          where each row represents the integer tokens from a block,
                          padded to the maximum length.
                          Returns an empty tensor of shape (0, 0) if no blocks are found.
    """
    # 1. Extract the content between <|mt_start|> and <|mt_end|> for each block
    # The pattern uses non-greedy matching (.*?) to ensure it captures content
    # between *each* start/end pair, not the first start and last end.
    block_pattern = r"<\|mt_start\|>(.*?)<\|mt_end\|>"
    extracted_blocks_content = re.findall(block_pattern, full_input_string)

    list_of_1d_tensors = []

    # 2. For each extracted block, get its integer tokens
    for block_content in extracted_blocks_content:
        # Extract integer numbers (e.g., 447, 472) from the current block's content
        numbers_as_strings = re.findall(r'\d+', block_content)

        # Convert the list of string numbers to integers
        current_block_integers = [int(num) for num in numbers_as_strings]

        # Convert the list of integers for the current block into a 1D LongTensor
        if current_block_integers:
            list_of_1d_tensors.append(torch.tensor([current_block_integers], dtype=torch.long).cuda())
        else:
            # Handle cases where a block might be empty of tokens (e.g., <|mt_start|><|mt_end|>)
            list_of_1d_tensors.append(torch.empty(0, dtype=torch.long).cuda())

    # 3. Pad the sequences and return a single 2D tensor
    return list_of_1d_tensors

@torch.no_grad()
def ids_to_mask_256(ids: list, mask_decoder, thresh=0.5) -> np.ndarray:
    """Token ids -> (256,256) uint8 {0,1}."""
    T, V = len(ids[0]), mask_decoder.codebook_size
    probs = torch.zeros(1, T, V, device="cuda", dtype=mask_decoder.dtype)
    ar = torch.arange(T, device="cuda")
    probs[0, ar, ids] = 1
    rec = mask_decoder.decode_prob(probs).mean(dim=1, keepdim=False)# [0]  # (256,256)
    return (rec >= thresh).to(torch.uint8).cpu()

def get_seg_hidden_states(hidden_states, output_ids, seg_id):
    seg_mask = output_ids == seg_id
    n_out = len(seg_mask)
    if n_out == 0:
        return hidden_states[0:0]
    return hidden_states[-n_out:][seg_mask]

def get_img_hidden_states(hidden_states, input_ids, img_id):
    img_mask = input_ids == img_id
    return hidden_states[img_mask]

def find_closest_aspect_ratio(aspect_ratio, target_ratios, width, height,
                              image_size):
    best_ratio_diff = float('inf')
    best_ratio = (1, 1)
    area = width * height
    for ratio in target_ratios:
        target_aspect_ratio = ratio[0] / ratio[1]
        ratio_diff = abs(aspect_ratio - target_aspect_ratio)
        if ratio_diff < best_ratio_diff:
            best_ratio_diff = ratio_diff
            best_ratio = ratio
        elif ratio_diff == best_ratio_diff:
            if area > 0.5 * image_size * image_size * ratio[0] * ratio[1]:
                best_ratio = ratio
    return best_ratio

def dynamic_preprocess(image,
                       min_num=1,
                       max_num=6,
                       image_size=448,
                       use_thumbnail=False):
    orig_width, orig_height = image.size
    aspect_ratio = orig_width / orig_height

    # calculate the existing image aspect ratio
    target_ratios = {(i, j)
                     for n in range(min_num, max_num + 1)
                     for i in range(1, n + 1) for j in range(1, n + 1)
                     if i * j <= max_num and i * j >= min_num}
    target_ratios = sorted(target_ratios, key=lambda x: x[0] * x[1])

    # find the closest aspect ratio to the target
    target_aspect_ratio = find_closest_aspect_ratio(aspect_ratio,
                                                    target_ratios, orig_width,
                                                    orig_height, image_size)

    # calculate the target width and height
    target_width = image_size * target_aspect_ratio[0]
    target_height = image_size * target_aspect_ratio[1]
    blocks = target_aspect_ratio[0] * target_aspect_ratio[1]

    # resize the image
    resized_img = image.resize((target_width, target_height))
    processed_images = []
    for i in range(blocks):
        box = ((i % (target_width // image_size)) * image_size,
               (i // (target_width // image_size)) * image_size,
               ((i % (target_width // image_size)) + 1) * image_size,
               ((i // (target_width // image_size)) + 1) * image_size)
        # split the image
        split_img = resized_img.crop(box)
        processed_images.append(split_img)
    assert len(processed_images) == blocks
    if use_thumbnail and len(processed_images) != 1:
        thumbnail_img = image.resize((image_size, image_size))
        processed_images.append(thumbnail_img)
    return processed_images


from transformers.cache_utils import Cache, DynamicCache

def prepare_inputs_for_generation_phi3(
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

