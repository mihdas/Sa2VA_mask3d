
import os
import torch
from torch.utils.data import DataLoader
from xtuner.utils import PROMPT_TEMPLATE
from projects.llava_sam2.models.preprocess.image_resize import DirectResize   
from xtuner.dataset.map_fns import template_map_fn_factory
from transformers import AutoTokenizer
from xtuner.dataset.samplers import LengthGroupedSampler
#from mmengine.dataset import build_dataloader
#from xtuner.utils import build_dataloader
from projects.llava_sam2.datasets import  video_lisa_collate_fn, ScannetppDataset
from tqdm import tqdm
from projects.llava_sam2.models import VideoLLaVASAMModel, VideoLLaVASAMModel_zero3
from third_parts.mmdet.models.losses import DiceLoss, CrossEntropyLoss
from projects.llava_sam2.models.internvl import InternVL_Slowfast
from peft import LoraConfig
special_tokens = ['[SEG]', '<p>', '</p>', '<vp>', '</vp>']
prompt_template = PROMPT_TEMPLATE.phi3_chat
# template = "internlm2_chat"
# prompt_template = PROMPT_TEMPLATE.internlm2_chat
max_length = 8192
path = '/nodes/hoppiness/work/vipradas/InternVL2_5-4B'
# path = '/nodes/astra/work/vipradas/InternVL2_5-8B'
from collections.abc import Mapping, Sequence

from mmengine.runner import load_state_dict

def to_cuda_cast(obj, fp_dtype=torch.float16):
    """
    Recursively move tensors to CUDA and cast float32→fp_dtype.

    Args
    ----
    obj : any nested structure of dict / list / tuple / tensor / other
    fp_dtype : torch.dtype
        Usually torch.float16 on RTX 30‑series or torch.bfloat16 on A100/H100.
    """
    if torch.is_tensor(obj):
        # cast *only* float32 tensors
#        if obj.dtype == torch.float32:
#            return obj.to(device='cuda', dtype=fp_dtype, non_blocking=True)
#        else:                           # keep original dtype
        return obj.to(device='cuda', non_blocking=True)

    elif isinstance(obj, Mapping):      # dict‑like
        return {k: to_cuda_cast(v, fp_dtype) for k, v in obj.items()}

    elif isinstance(obj, Sequence) and not isinstance(obj, (str, bytes)):
        return type(obj)(to_cuda_cast(v, fp_dtype) for v in obj)

    else:                               # numbers, strings, None, …
        return obj

tokenizer = dict(
    type=AutoTokenizer.from_pretrained,
    pretrained_model_name_or_path=path,
    trust_remote_code=True,
    padding_side='right')

extra_image_processor = dict(
    type=DirectResize,
    target_length=1024,
)
template_map_fn=dict(
        type=template_map_fn_factory, template=prompt_template),

#from Thesis.Sa2VA_p.projects.llava_sam2.datasets import ScannetppDataset
dataset = ScannetppDataset(
    image_folder="/globalwork/vipradas/scannet_images",
    expression_file="/home/vipradas/Thesis/Sa2VA_p/expressions/scanrefer_val_24.json",
    mask_file="/home/vipradas/Thesis/Sa2VA_p/mask_dicts/mask_dict.json",
    tokenizer=tokenizer,    
    template_map_fn=dict(
        type=template_map_fn_factory, template=prompt_template),
    max_length=max_length,
    lazy=True,
    repeats=1,
    special_tokens=special_tokens,
    extra_image_processor=extra_image_processor,
    sampled_frames=24,
)


model = VideoLLaVASAMModel_zero3(
    special_tokens=special_tokens,
    frozen_sam2_decoder=False,
    mllm=dict(
        type=InternVL_Slowfast,
        model_path=path,
        freeze_llm=True,
        freeze_visual_encoder=True,
        llm_lora=dict(
            type=LoraConfig,
            r=128,
            lora_alpha=256,
            lora_dropout=0.05,
            bias='none',
            task_type='CAUSAL_LM'),
        special_tokens=special_tokens,
    ),
    tokenizer=tokenizer,
    grounding_encoder=None,
    loss_mask=dict(
        type=CrossEntropyLoss,
        use_sigmoid=True,
        reduction='mean',
        loss_weight=2.0),
    loss_dice=dict(
        type=DiceLoss,
        use_sigmoid=True,
        activate=True,
        reduction='mean',
        naive_dice=True,
        eps=1.0,
        loss_weight=0.5),
    pretrained_pth=None,
    loss_sample_points=True,
    # loss_sample_points=False,
    bs=1,
)


train_dataloader = DataLoader(
    batch_size=1,
    num_workers=0,
    dataset=dataset,
    sampler=dict(
        type=LengthGroupedSampler,
        length_property='modality_length',
        per_device_batch_size=1),
)

ckpt=torch.load("/nodes/faxe/work/vipradas/state_dict_no_proj_1t4580.pth", map_location='cpu')

load_state_dict(model,ckpt)
model=model.cuda()
model.eval()
print(len(dataset))
with torch.cuda.amp.autocast(dtype=torch.float16):
    with torch.inference_mode():
        for i in range(len(dataset)):
            
            imput=video_lisa_collate_fn([train_dataloader.dataset[i]])
            
            model(data=to_cuda_cast(imput["data"]))

