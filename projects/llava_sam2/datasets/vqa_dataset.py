import copy
import re

import barecat
from pathlib import Path
import random
import glob
import json
import logging
import os
from typing import Literal

import torch

from mmengine import print_log
from mmengine.config import Config, ConfigDict
from PIL import Image
from torch.utils.data import Dataset
import numpy as np
import torch.nn.functional as F
import torchvision.transforms as T
from torchvision.transforms.functional import InterpolationMode
from pycocotools.coco import COCO
from pycocotools import mask as mask_utils

from xtuner.registry import BUILDER
from xtuner.utils import IGNORE_INDEX
from xtuner.dataset.utils import encode_fn
from xtuner.dataset.map_fns import llava_map_fn

from projects.glamm.datasets.utils.utils import expand2square

from projects.glamm.datasets.utils.utils import SEG_QUESTIONS, ANSWER_LIST
from projects.glamm.utils import DEFAULT_IMAGE_TOKEN, DEFAULT_IM_START_TOKEN, DEFAULT_IM_END_TOKEN

from .utils import dynamic_preprocess


class InfinityMMDataset(Dataset):
    os.environ['TOKENIZERS_PARALLELISM'] = 'true'
    IMG_CONTEXT_TOKEN = '<IMG_CONTEXT>'
    IMG_START_TOKEN = '<img>'
    IMG_END_TOKEN = '</img>'

    IMAGENET_MEAN = (0.485, 0.456, 0.406)
    IMAGENET_STD = (0.229, 0.224, 0.225)

    def __init__(self,
                 tokenizer,
                 data_path,
                 prompt_template,
                 special_tokens=None,
                 max_length=8192,
                 offline_save_path='./work_dirs/infinityMM.json',
                 ):
        self.offline_save_path = offline_save_path
        self.tokenizer = BUILDER.build(tokenizer)
        if special_tokens is not None:
            self.tokenizer.add_tokens(special_tokens, special_tokens=True)
        self._system = ''

        self.template = prompt_template
        self.max_length = max_length

        self.min_dynamic_patch = 1
        self.max_dynamic_patch = 12
        self.downsample_ratio = 0.5
        self.image_size = 448
        self.use_thumbnail = True
        patch_size = 14
        self.patch_token = int(
            (self.image_size // patch_size) ** 2 * (self.downsample_ratio ** 2))

        self.transformer = T.Compose([
            T.Lambda(lambda img: img.convert('RGB')
            if img.mode != 'RGB' else img),
            T.Resize((self.image_size, self.image_size),
                     interpolation=InterpolationMode.BICUBIC),
            T.ToTensor(),
            T.Normalize(mean=self.IMAGENET_MEAN, std=self.IMAGENET_STD)
        ])

        self.data = self._load_annotations(data_path)
        self._max_refetch = 1000

    def _load_annotations(self, data_path):
        if os.path.exists(self.offline_save_path):
            with open(self.offline_save_path, 'r') as f:
                ret = json.load(f)
            print(f"Load InfinityMM file list from {self.offline_save_path}, {len(ret)} items !!!")
            return ret
        sub_folders = []
        for sub_folder in os.listdir(data_path):
            if '.' not in sub_folder:
                # a folder
                if "LVIS_111k" in sub_folder:
                    # special case, have subsub folder
                    subsub_folders = os.listdir(os.path.join(data_path, sub_folder))
                    for subsub_folder in subsub_folders:
                        sub_folders.append(os.path.join(data_path, sub_folder, subsub_folder))
                else:
                    sub_folders.append(os.path.join(data_path, sub_folder))

        all_jsons = []
        for sub_folder in sub_folders:
            print(f"Processing {sub_folder} !!!")
            _files = os.listdir(sub_folder)
            _num = 0
            for _file in _files:
                if '.json' in _file:
                    _json_path = os.path.join(sub_folder, _file)
                    _num += 1
                    all_jsons.append(os.path.join(sub_folder, _file))
            print(f"Finished {sub_folder} has {_num} items.")

        with open(self.offline_save_path, 'w') as f:
            json.dump(all_jsons, f)

        return all_jsons

    def __getitem__(self, index):
        for _ in range(self._max_refetch + 1):
            data = self.prepare_data(index)
            # Broken images may cause the returned data to be None
            if data is None:
                index = self._rand_another()
                continue
            return data

    def __len__(self):
        return len(self.data)

    @property
    def modality_length(self):
        self.group_length = []
        for data_dict in self.data:
            self.group_length.append(100)
        return self.group_length

    @property
    def length(self):
        group_length = np.array(self.group_length)
        group_length = np.abs(group_length).tolist()
        return group_length

    def prepare_data(self, index):
        data_path = self.data[index]

        with open(data_path, 'r') as f:
            data_dict = json.load(f)
        if 'image' in data_dict.keys():
            data_dict['image'] = data_path.replace('.json', '.jpg')

        if data_dict is None:
            return None

        out_data_dict = {}

        if data_dict.get('image', None) is not None:
            image_file = data_dict['image']
            try:
                import pudb; pudb.set_trace()
                image = Image.open(image_file).convert('RGB')
            except Exception as e:
                print(f'Error: {e}', flush=True)
                print_log(f'Error: {e}', logger='current')
                return None

            images = dynamic_preprocess(image, self.min_dynamic_patch,
                                        self.max_dynamic_patch,
                                        self.image_size, self.use_thumbnail)
            pixel_values = [self.transformer(image) for image in images]
            pixel_values = torch.stack(pixel_values)
            out_data_dict['pixel_values'] = pixel_values

            num_image_tokens = pixel_values.shape[0] * self.patch_token
            image_token_str = f'{self.IMG_START_TOKEN}' \
                              f'{self.IMG_CONTEXT_TOKEN * num_image_tokens}' \
                              f'{self.IMG_END_TOKEN}'
            token_dict = self.get_inputid_labels(
                data_dict['conversations'], image_token_str)
            out_data_dict.update(token_dict)
        else:
            token_dict = self.get_inputid_labels(
                data_dict['conversations'], None)
            out_data_dict.update(token_dict)
            out_data_dict['pixel_values'] = torch.zeros(
                1, 3, self.image_size, self.image_size)
        return out_data_dict

    def _rand_another(self) -> int:
        return np.random.randint(0, len(self.data))

    def _convert_coordinates_to_sa1b_format(self, text):
        """Convert normalized coordinates [0-1] to SA-1B format with <box> tags and [0-1000] range"""
        def convert_coord_match(match):
            coords_str = match.group(1)  # Get the coordinates inside brackets
            try:
                # Parse coordinates like [0.67, 0.63, 0.82, 0.74]
                coords = [float(x.strip()) for x in coords_str.split(',')]
                if len(coords) == 4:
                    # Convert from normalized [0-1] to [0-1000]
                    converted_coords = [int(coord * 1000) for coord in coords]
                    # Format with <box> tags and double brackets like SA1B
                    box_str = f"[{', '.join(map(str, converted_coords))}]"
                    return f"<box>[{box_str}]</box>"
                else:
                    return match.group(0)  # Return original if not 4 coordinates
            except:
                return match.group(0)  # Return original if parsing fails

        # Strict pattern to match normalized coordinates like [0.67, 0.63, 0.82, 0.74]
        # Only matches decimal numbers between 0 and 1, separated by commas
        pattern = r'\[(\s*(?:[01](?:\.\d+)?\s*,\s*){3}[01](?:\.\d+)?\s*)\]'
        return re.sub(pattern, convert_coord_match, text)

    def get_inputid_labels(self, conversations, image_token_str) -> dict:
        input = ''
        out_conversation = []
        while conversations and conversations[0]['from'] == 'gpt':
            # Skip the first one if it is from gpt
            conversations = conversations[1:]
        for i, msg in enumerate(conversations):
            if msg['from'] == 'human':
                # Convert coordinates in human questions from [0-1] to [0-1000] format
                converted_msg = self._convert_coordinates_to_sa1b_format(msg['value'])

                # change to 1 image
                if '<image>' in converted_msg:
                    converted_msg = converted_msg.replace('<image>\n', '').replace('<image>', '')
                    if i == 0:
                        converted_msg = "<image>\n" + converted_msg

                if image_token_str is None and '<image>' in converted_msg:
                    converted_msg = converted_msg.replace('<image>', '')
                if '<image>' in converted_msg:
                    converted_msg = converted_msg.replace('<image>', image_token_str).strip()
                input += converted_msg.strip()
            elif msg['from'] == 'gpt':
                # Convert coordinates in GPT responses from [0-1] to [0-1000] format
                converted_output = self._convert_coordinates_to_sa1b_format(msg['value'].strip())
                out_conversation.append({
                    'input': input,
                    'output': converted_output
                })
                input = ''
            else:
                raise NotImplementedError

        input_ids, labels = [], []
        for i, single_turn_conversation in enumerate(out_conversation):
            input = single_turn_conversation.get('input', '')
            if input is None:
                input = ''
            input_text = self.template.INSTRUCTION.format(
                input=input, round=i + 1)

            if i == 0:
                if self._system != '' and self._system is not None:
                    system = self.template.SYSTEM.format(system=self._system)
                    input_text = system + input_text
                input_encode = self.tokenizer.encode(
                    input_text, add_special_tokens=True)
            else:
                input_encode = self.tokenizer.encode(
                    input_text, add_special_tokens=False)
            input_ids += input_encode
            labels += [IGNORE_INDEX] * len(input_encode)

            output_text = single_turn_conversation.get('output', '')
            if self.template.get('SUFFIX', None):
                output_text += self.template.SUFFIX
            output_encode = self.tokenizer.encode(
                output_text, add_special_tokens=False)
            input_ids += output_encode
            labels += copy.deepcopy(output_encode)

        if len(input_ids) > self.max_length:
            input_ids = input_ids[:self.max_length]
            labels = labels[:self.max_length]
            print_log(
                f'Warning: input_ids length({len(input_ids)}) '
                f'is longer than max_length, cut to {self.max_length}',
                logger='current')
        return {'input_ids': input_ids, 'labels': labels}


class LLaVADataset(Dataset):
    os.environ['TOKENIZERS_PARALLELISM'] = 'true'
    IMG_CONTEXT_TOKEN = '<IMG_CONTEXT>'
    IMG_START_TOKEN = '<img>'
    IMG_END_TOKEN = '</img>'

    IMAGENET_MEAN = (0.485, 0.456, 0.406)
    IMAGENET_STD = (0.229, 0.224, 0.225)

    def __init__(self,
                 tokenizer,
                 data_path,
                 prompt_template,
                 special_tokens=None,
                 image_folder=None,
                 max_length=8192,
                 arch_type: Literal['intern_vl', 'qwen'] = 'qwen',
                 preprocessor=None,
                 skip_pure_text=False,
                 ):

        self.tokenizer = BUILDER.build(tokenizer)
        if special_tokens is not None:
            self.tokenizer.add_tokens(special_tokens, special_tokens=True)

        self.image_folder = image_folder
        self.template = prompt_template
        self.max_length = max_length

        self._system = ''

        self.arch_type = arch_type
        self.min_dynamic_patch = 1
        self.max_dynamic_patch = 12
        self.downsample_ratio = 0.5
        if self.arch_type == 'llava':
            self.downsample_ratio = 1
        self.image_size = 448
        if self.arch_type == 'llava':
            self.image_size = 336
        self.use_thumbnail = True
        patch_size = 14
        self.patch_token = int(
            (self.image_size // patch_size)**2 * (self.downsample_ratio**2))


        if self.arch_type == 'qwen':
            self.IMG_CONTEXT_TOKEN = '<|image_pad|>'
            self.IMG_START_TOKEN = '<|vision_start|>'
            self.IMG_END_TOKEN = '<|vision_end|>'
        elif self.arch_type == 'llava':
            self.IMG_CONTEXT_TOKEN = '<image>'
            self.IMG_START_TOKEN = ''
            self.IMG_END_TOKEN = ''

        if preprocessor is None:
            self.transformer = T.Compose([
                T.Lambda(lambda img: img.convert('RGB') if img.mode != 'RGB' else img),
                T.Resize((self.image_size, self.image_size), interpolation=InterpolationMode.BICUBIC),
                T.ToTensor(),
                T.Normalize(mean=self.IMAGENET_MEAN, std=self.IMAGENET_STD)
            ])
            self.preprocessor = None
        else:
            self.transformer = None
            self.preprocessor = BUILDER.build(preprocessor)

        self.data = self._load_annotations(data_path, image_folder)
        self._max_refetch = 1000

        self.skip_pure_text = skip_pure_text

    def _load_annotations(self, data_path, image_folder=None):
        data = json.load(open(data_path))
        return data

    def __getitem__(self, index):
        for _ in range(self._max_refetch + 1):
            data = self.prepare_data(index)
            # Broken images may cause the returned data to be None
            if data is None:
                index = self._rand_another()
                continue
            return data

    def __len__(self):
        return len(self.data)

    @property
    def modality_length(self):
        self.group_length = []
        for data_dict in self.data:
            self.group_length.append(100)
        return self.group_length

    @property
    def length(self):
        group_length = np.array(self.group_length)
        group_length = np.abs(group_length).tolist()
        return group_length
    
    def prepare_data(self, index):
        data_dict: dict = self.data[index]
        
        if data_dict is None:
            return None
        
        out_data_dict = {}

        if self.skip_pure_text and data_dict.get('image', None) is None:
            return None

        if data_dict.get('image', None) is not None:
            print(data_dict['image'])
            image_file = os.path.join(self.image_folder, data_dict['image'])
            # try:
            #     image_file = Path(self.image_folder) / data_dict['image']
            #     barecat_dir = Path(self.image_folder)
            #     barecat_filename = Path(self.image_folder).name.lower()
            #     if barecat_filename.endswith("_barecat"):
            #         barecat_filename = barecat_filename[:-8]+ ".barecat"
            #     with barecat.Barecat(barecat_dir / barecat_filename, readonly=True) as bc:
            #         with bc.open(Path("llava_images") / Path(image_file).relative_to(barecat_dir), 'rb') as f:
            #             image = Image.open(f).convert('RGB')
            image = Image.open(image_file).convert('RGB')
            # except Exception as e:
            #     print(f'Error: {e}', flush=True)
            #     print_log(f'Error: {e}', logger='current')
            #     return None
            if self.preprocessor is not None:
                # images = dynamic_preprocess(image, self.min_dynamic_patch,
                #                             self.max_dynamic_patch,
                #                             self.image_size, self.use_thumbnail)
                images = [image]
                if self.arch_type == 'qwen':
                    _data_dict = self.preprocessor(images, do_resize=True)
                    _data_dict['pixel_values'] = torch.tensor(_data_dict['pixel_values'], dtype=torch.float)
                    _data_dict['image_grid_thw'] = torch.tensor(_data_dict['image_grid_thw'], dtype=torch.int)
                    num_image_tokens = int(_data_dict['image_grid_thw'][0].prod() * (self.downsample_ratio ** 2))
                elif self.arch_type == 'llava':
                    _data_dict = self.preprocessor(images, do_resize=True, size=(self.image_size, self.image_size))
                    _data_dict['pixel_values'] = np.stack(_data_dict['pixel_values'], axis=0)
                    _data_dict['pixel_values'] = torch.tensor(_data_dict['pixel_values'], dtype=torch.float)
                    num_image_tokens = _data_dict['pixel_values'].shape[0] * self.patch_token
                else:
                    raise NotImplementedError
                out_data_dict.update(_data_dict)
            else:
                images = dynamic_preprocess(image, self.min_dynamic_patch,
                                            self.max_dynamic_patch,
                                            self.image_size, self.use_thumbnail)
                pixel_values = [self.transformer(image) for image in images]
                pixel_values = torch.stack(pixel_values)
                out_data_dict['pixel_values'] = pixel_values

                num_image_tokens = pixel_values.shape[0] * self.patch_token
            image_token_str = f'{self.IMG_START_TOKEN}' \
                              f'{self.IMG_CONTEXT_TOKEN * num_image_tokens}' \
                              f'{self.IMG_END_TOKEN}'
            token_dict = self.get_inputid_labels(
                data_dict['conversations'], image_token_str)
            out_data_dict.update(token_dict)
        else:
            #print('Here')
            token_dict = self.get_inputid_labels(
                data_dict['conversations'], None)
            out_data_dict.update(token_dict)
            out_data_dict['pixel_values'] = torch.zeros(
                1, 3, self.image_size, self.image_size)
        return out_data_dict

    def _rand_another(self) -> int:
        return np.random.randint(0, len(self.data))

    def _convert_coordinates_to_sa1b_format(self, text):
        """Convert normalized coordinates [0-1] to SA-1B format with <box> tags and [0-1000] range"""
        def convert_coord_match(match):
            coords_str = match.group(1)  # Get the coordinates inside brackets
            try:
                # Parse coordinates like [0.67, 0.63, 0.82, 0.74]
                coords = [float(x.strip()) for x in coords_str.split(',')]
                if len(coords) == 4:
                    # Convert from normalized [0-1] to [0-1000]
                    converted_coords = [int(coord * 1000) for coord in coords]
                    # Format with <box> tags and double brackets like SA1B
                    box_str = f"[{', '.join(map(str, converted_coords))}]"
                    return f"<box>[{box_str}]</box>"
                else:
                    return match.group(0)  # Return original if not 4 coordinates
            except:
                return match.group(0)  # Return original if parsing fails

        # Strict pattern to match normalized coordinates like [0.67, 0.63, 0.82, 0.74]
        # Only matches decimal numbers between 0 and 1, separated by commas
        pattern = r'\[(\s*(?:[01](?:\.\d+)?\s*,\s*){3}[01](?:\.\d+)?\s*)\]'
        return re.sub(pattern, convert_coord_match, text)

    def get_inputid_labels(self, conversations, image_token_str) -> dict:
        input = ''
        out_conversation = []
        while conversations and conversations[0]['from'] == 'gpt':
            # Skip the first one if it is from gpt
            conversations = conversations[1:]
        for msg in conversations:
            if msg['from'] == 'human':
                # Convert coordinates in human questions from [0-1] to [0-1000] format
                converted_input = self._convert_coordinates_to_sa1b_format(msg['value'])
                if image_token_str is None and '<image>' in converted_input:
                    converted_input = converted_input.replace('<image>', '')
                if '<image>' in converted_input:
                    converted_input = converted_input.replace('<image>', image_token_str).strip()
                input += converted_input.strip()
            elif msg['from'] == 'gpt':
                # Convert coordinates in GPT responses from [0-1] to [0-1000] format
                converted_output = self._convert_coordinates_to_sa1b_format(msg['value'].strip())
                out_conversation.append({
                    'input': input,
                    'output': converted_output
                })
                input = ''
            else:
                raise NotImplementedError

        input_ids, labels = [], []
        for i, single_turn_conversation in enumerate(out_conversation):
            input = single_turn_conversation.get('input', '')
            if input is None:
                input = ''
            input_text = self.template.INSTRUCTION.format(
                input=input, round=i + 1)

            if i == 0:
                if self._system != '' and self._system is not None:
                    system = self.template.SYSTEM.format(system=self._system)
                    input_text = system + input_text
                input_encode = self.tokenizer.encode(
                    input_text, add_special_tokens=True)
            else:
                input_encode = self.tokenizer.encode(
                    input_text, add_special_tokens=False)
            input_ids += input_encode
            labels += [IGNORE_INDEX] * len(input_encode)

            output_text = single_turn_conversation.get('output', '')
            if self.template.get('SUFFIX', None):
                output_text += self.template.SUFFIX
            output_encode = self.tokenizer.encode(
                output_text, add_special_tokens=False)
            input_ids += output_encode
            labels += copy.deepcopy(output_encode)

        if len(input_ids) > self.max_length:
            input_ids = input_ids[:self.max_length]
            labels = labels[:self.max_length]
            print_log(
                f'Warning: input_ids length({len(input_ids)}) '
                f'is longer than max_length, cut to {self.max_length}',
                logger='current')
        return {'input_ids': input_ids, 'labels': labels}


if __name__ == '__main__':
    from transformers import CLIPImageProcessor, AutoTokenizer
    # from third_parts.segment_anything.utils.transforms import ResizeLongestSide
    pretrained_model = 'MBZUAI/GLaMM-GranD-Pretrained'
    llm_name_or_path = 'lmsys/vicuna-7b-v1.5'

    tokenizer = dict(
        type=AutoTokenizer.from_pretrained,
        pretrained_model_name_or_path=llm_name_or_path)
    # image_processor = dict(
    #     type=CLIPImageProcessor.from_pretrained,
    #     pretrained_model_name_or_path='openai/clip-vit-large-patch14-336')
    # extra_image_processor = dict(
    #     type=ResizeLongestSide,
    #     target_length=1024,
    # )
    from xtuner.utils.templates import PROMPT_TEMPLATE
    prompt_template = PROMPT_TEMPLATE.vicuna
    # from xtuner.dataset.map_fns import llava_map_fn, template_map_fn_factory, template_map_fn
    # from projects.glamm.datasets.collate_fns.glamm_collate_fn import glamm_collate_fn

    dataset = LLaVADataset(
        tokenizer=tokenizer,
        data_path='/home/vipradas/Thesis/LLaVA-3D-Data/LLaVA-3D-Instruct-860K.json',
        prompt_template=prompt_template,
        special_tokens=['[SEG]'],
        image_folder='/globalwork/vipradas/scannet_images',
    )
    # for i in range(1000):
    #     dataset[i]
        #print(dataset[i])

    

    print(dataset[5])
