import json
import random
import re
from copy import deepcopy

import barecat
import numpy as np
import pycocotools.mask as maskUtils
import torch
from PIL import Image
from xtuner.utils import DEFAULT_IMAGE_TOKEN

from .encode_fn import video_lisa_encode_fn
from .ReVOS_Dataset import ANSWER_LIST, SEG_QUESTIONS, VideoReVOSDataset
from .utils import select_frames

# Constants for ViCaS dataset
VicasConfig = {
    "SPLIT_VERSION": "v1.0",
    "SPLITS_PATH": "splits/{version}/train.json",
    "ANNOTATIONS_PATH": "annotations/{version}/{video_id:06d}.json",
    "FRAME_PATH": "video_frames/{video_id:06d}/{frame_id:05d}.jpg",
}

GCG_QUESTIONS = [
    DEFAULT_IMAGE_TOKEN
    + "Could you please give me a brief description of the video? Please respond with interleaved segmentation masks for the corresponding parts of the answer.",
    DEFAULT_IMAGE_TOKEN
    + "Can you provide a brief description of the video? Please output with interleaved segmentation masks for the corresponding phrases.",
    DEFAULT_IMAGE_TOKEN
    + "Please briefly describe the contents of the video. Please respond with interleaved segmentation masks for the corresponding parts of the answer.",
    DEFAULT_IMAGE_TOKEN
    + "Could you give a brief explanation of what can be found within this video? Please output with interleaved segmentation masks for the corresponding phrases.",
    DEFAULT_IMAGE_TOKEN
    + "Could you give me an brief explanation of this video? Please respond with interleaved segmentation masks for the corresponding phrases.",
    DEFAULT_IMAGE_TOKEN
    + "Could you provide me with a briefly analysis of this video? Please output with interleaved segmentation masks for the corresponding parts of the answer.",
]


class VideoViCaSDataset(VideoReVOSDataset):
    """ViCaS dataset class for video referential segmentation with interleaved format.

    ViCaS dataset contains videos with:
    - Annotations in JSON format (one per video)
    - Captions with []<mask_N> placeholders for objects
    - Object referrals with prompts
    - Segmentation masks in RLE format
    - Multiple objects per video

    This class converts ViCaS format to interleaved segmentation format similar to GCG.
    """

    def json_file_preprocess(self, expression_file, mask_file):
        """Process ViCaS annotation files to match the ReVOS format.

        Args:
            expression_file: Path to the barecat archive containing splits and annotations
            mask_file: Path to the barecat archive (same as expression_file for ViCaS)

        Returns:
            tuple: (vid2metaid, metas, mask_dict)
        """
        # Store barecat archive path for later use (initialize per access)
        self.barecat_path = expression_file

        # Load train/test split using barecat
        splits_path = VicasConfig["SPLITS_PATH"].format(
            version=VicasConfig["SPLIT_VERSION"]
        )
        with barecat.Barecat(self.barecat_path).open(splits_path) as f:
            video_ids = json.load(f)  # List of video IDs

        # Create annotation dict for lazy loading (similar to ReSAM2 approach)
        metas = []
        vid2metaid = {}
        for video_id in video_ids:
            annotation_path = VicasConfig["ANNOTATIONS_PATH"].format(
                version=VicasConfig["SPLIT_VERSION"], video_id=video_id
            )
            vid2metaid[video_id] = 0
            metas.append(
                {
                    "annotation_path": annotation_path,
                    "video_id": video_id,
                }
            )

        return vid2metaid, metas, {}

    def process_caption(self, caption, num_seg_tokens, all_track_ids, track_masks):
        """
        Process a caption with masks to a new format and extract the corresponding masks.

        Args:
            caption: The original caption with masks like "[A man in fluorescent]<mask_1,4>"
            num_seg_tokens: Number of [SEG] tokens to add after each <p> tag
            all_track_ids: List of all available track IDs
            track_masks: Tensor of shape [len(all_track_ids), num_frames, H, W]

        Returns:
            processed_caption: The caption in the new format
            new_masks: Tensor of masks corresponding to the masks in the caption
        """

        # Find all mask references in the caption
        mask_pattern = r"<mask_(\d+(?:,\d+)*)>"
        mask_matches = re.findall(mask_pattern, caption)

        # Also check for unavailable masks
        unavailable_pattern = r"<mask_\?>"
        unavailable_matches = re.findall(unavailable_pattern, caption)

        # Create a list to store the new masks
        new_masks = []
        # Process each mask reference
        for mask_str in mask_matches:
            # Split the mask IDs
            mask_ids = [int(id) for id in mask_str.split(",")]
            # Get the masks for these indices, but only if they exist in track_masks
            masks_to_combine = []
            for mask_id in mask_ids:
                if mask_id in track_masks:
                    masks_to_combine.append(track_masks[mask_id])
                else:
                    # Skip this mask reference if any mask ID is missing
                    print(f"Warning: Mask ID {mask_id} not found in track_masks. Available IDs: {list(track_masks.keys())[:10]}...")
                    masks_to_combine = []
                    break

            # Only process if we found valid masks
            if masks_to_combine:
                # Combine them with OR operation
                combined_mask = np.logical_or.reduce(masks_to_combine)
                # Add to new_masks
                new_masks.append(combined_mask)

        # Process the caption to replace mask references with <p> tags and [SEG] tokens
        processed_caption = caption
        # Replace each mask reference with <p>...</p> and [SEG] tokens
        # Only process masks that were successfully added to new_masks
        for mask_str in mask_matches:
            # Split the mask IDs to check if they're valid
            mask_ids = [int(id) for id in mask_str.split(",")]
            # Check if all mask IDs exist in track_masks
            all_masks_exist = all(mask_id in track_masks for mask_id in mask_ids)

            if all_masks_exist:
                # Find the text before the mask reference
                pattern = r"\[([^\]]+)\]<mask_" + re.escape(mask_str) + r">"
                matches = list(re.finditer(pattern, processed_caption))
                # Process matches in reverse order to avoid changing indices
                for match in reversed(matches):
                    text = match.group(1)
                    seg_tokens = "".join(["[SEG]"] * num_seg_tokens)
                    replacement = f"<p>{text}</p> {seg_tokens}"
                    processed_caption = (
                        processed_caption[: match.start()]
                        + replacement
                        + processed_caption[match.end() :]
                    )
            else:
                # Remove the entire mask reference (text and <mask_...>) from caption
                pattern = r"\[([^\]]+)\]<mask_" + re.escape(mask_str) + r">"
                processed_caption = re.sub(pattern, "", processed_caption)
                print(f"Warning: Removed mask reference {mask_str} from caption due to missing masks")

        # Replace unavailable masks with just the text (no <p> tags or [SEG] tokens)
        for _ in unavailable_matches:
            pattern = r"\[([^\]]+)\]<mask_\?>"
            matches = list(re.finditer(pattern, processed_caption))
            # Process matches in reverse order to avoid changing indices
            for match in reversed(matches):
                text = match.group(1)
                replacement = text
                processed_caption = (
                    processed_caption[: match.start()]
                    + replacement
                    + processed_caption[match.end() :]
                )

        # Convert new_masks to a tensor if it's not empty
        if new_masks:
            new_masks = np.stack(new_masks)
        else:
            new_masks = np.array([])
        return processed_caption, new_masks

    def load_video_masks(
        self,
        processed_annotation,
        only_gt=True,
    ):
        segmentations = processed_annotation["segmentations"]
        all_track_ids = processed_annotation["all_track_ids"]

        frame_indices = list(range(len(segmentations)))
        if only_gt:
            frame_indices = [
                idx for idx, seg in enumerate(segmentations) if seg["is_gt"]
            ]

        object_masks = {track_id: list() for track_id in all_track_ids}
        for frame_idx, segmentation in enumerate(segmentations):
            if frame_idx not in frame_indices:
                continue
            for idx, track_id in enumerate(segmentation["track_ids"]):
                mask_rle = segmentation["mask_rles"][idx]
                mask_rle = deepcopy(mask_rle)
                mask_rle["counts"] = mask_rle["counts"].encode("utf-8")
                mask = maskUtils.decode(mask_rle)
                object_masks[track_id].append(mask)

        return object_masks, frame_indices

    def _load_frame_images(self, frame_paths):
        """Load specific frame images from barecat archive.

        Args:
            frame_paths: List of frame image paths in the barecat archive

        Returns:
            list: List of PIL.Image frames in RGB format
        """
        pil_frames = []
        bc = barecat.Barecat(self.barecat_path)  # Create new barecat instance for this access
        for frame_path in frame_paths:
            with bc.open(frame_path, "rb") as f:
                frame_image = Image.open(f).convert("RGB")
                pil_frames.append(frame_image)
        return pil_frames

    def dataset_map_fn(self, data_dict, only_gt=True, select_k=8, select_k_tarvis=8):
        """Override dataset mapping to handle ViCaS specific frame indexing with lazy loading.

        Args:
            data_dict
            select_k: Number of frames to select for context
            select_k_tarvis: Number of frames to select for TARViS

        Returns:
            dict: Mapped data dictionary
        """
        # Get video_id and process annotation on-demand
        video_id = data_dict["video_id"]
        annotation_path = data_dict["annotation_path"]
        # Create new barecat instance for this access
        bc = barecat.Barecat(self.barecat_path)
        with bc.open(annotation_path) as f:
            processed_annotation = json.load(f)

        # Oops, UVO_trainval, Kinetics, Oops_train, Oops_val
        # src_dataset = processed_annotation["src_dataset"]
        processed_annotation["video_id"] = video_id

        frames = sorted(
            [
                int(seg["filename"].split(".")[0])
                for seg in processed_annotation["segmentations"]
            ]
        )
        processed_annotation["frames"] = frames

        track_masks, frame_indices = self.load_video_masks(
            processed_annotation, only_gt=only_gt
        )

        # Use the same frame selection logic as parent class
        (
            selected_frame_indexes,
            tarvis_selected_frame_indexes,
            context_num_frames,
            tarvis_context_num_frames,
        ) = select_frames(
            vid_len=len(frame_indices),
            num_frames=select_k,
            tarvis_num_frames=select_k_tarvis,
            train_mode=True,
        )

        # Get selected frame indices
        selected_frames = [frame_indices[i] for i in selected_frame_indexes]
        tarvis_selected_frames = [
            frame_indices[i] for i in tarvis_selected_frame_indexes
        ]

        for track_id, track_mask in track_masks.items():
            track_masks[track_id] = [track_mask[idx] for idx in selected_frame_indexes]

        # Prepare frame paths for selected frames
        selected_frame_paths = [
            VicasConfig["FRAME_PATH"].format(video_id=video_id, frame_id=frame_idx)
            for frame_idx in selected_frames
        ]
        tarvis_selected_frame_paths = [
            VicasConfig["FRAME_PATH"].format(video_id=video_id, frame_id=frame_idx)
            for frame_idx in tarvis_selected_frames
        ]

        # Store frame paths for later loading
        images = selected_frame_paths
        tarvis_images = tarvis_selected_frame_paths

        processed_caption, object_masks = self.process_caption(
            processed_annotation["caption_raw_en"],
            num_seg_tokens=self.num_tokens_per_expression,
            all_track_ids=processed_annotation["all_track_ids"],
            track_masks=track_masks,
        )

        # Create video tokens (same as parent class)
        frame_token_str = (
            f"{self.IMG_START_TOKEN}"
            f"{self.IMG_CONTEXT_TOKEN * self.patch_token}"
            f"{self.IMG_END_TOKEN}"
        )

        # Create video tokens for all frames
        frame_tokens = frame_token_str + "\n"
        frame_tokens = frame_tokens * len(selected_frames)
        frame_tokens = frame_tokens.strip()

        question = random.choice(GCG_QUESTIONS).strip()
        # Add video tokens to the beginning of the question
        full_question = frame_tokens + " " + question
        formatted_conversation = [{"input": full_question, "output": processed_caption}]

        ret = {
            "images": images,
            "tarvis_images": tarvis_images,
            "masks": torch.from_numpy(object_masks.astype(np.uint8)),
            "conversation": formatted_conversation,
        }
        return ret

    def __getitem__(self, index):
        """Override __getitem__ to use barecat for image loading."""
        index = index % self.real_len()

        # Use index directly as annotation_key - annotation_dict was created with enumerate(video_ids)
        annotation_data = self.text_data[index]

        data_dict = self.dataset_map_fn(
            annotation_data,
            select_k=self.sampled_frames,
            select_k_tarvis=self.tarvis_sampled_frames,
            only_gt=True,
        )

        assert "images" in data_dict.keys()
        pixel_values = []
        extra_pixel_values = []
        num_video_tokens = None
        num_frame_tokens = None

        selected_frame_paths = data_dict["images"]
        tarvis_selected_frame_paths = data_dict["tarvis_images"]

        # Combine all unique frame paths to load them once
        all_frame_paths = list(set(selected_frame_paths + tarvis_selected_frame_paths))
        all_frame_paths_sorted = sorted(all_frame_paths)

        # Load all unique frames once
        extracted_frames = self._load_frame_images(all_frame_paths_sorted)

        # Create mapping from frame path to extracted frame
        frame_to_image = {
            frame_path: extracted_frames[i]
            for i, frame_path in enumerate(all_frame_paths_sorted)
        }

        # Get regular frames for processing
        regular_frames = [frame_to_image[frame_path] for frame_path in selected_frame_paths]

        # Get dimensions from first frame
        first_frame = regular_frames[0]
        ori_width, ori_height = first_frame.size

        # Process regular frames
        for frame_image in regular_frames:
            if self.preprocessor is not None:
                pixel_values.append(frame_image)
            else:
                frame_image = self.transformer(frame_image)
                pixel_values.append(frame_image)

        # # Get tarvis frames for grounding processing
        # tarvis_frames = [
        #     frame_to_image[frame_num] for frame_num in tarvis_selected_frames
        # ]
        #
        # # Process grounding frames
        # for frame_image in tarvis_frames:
        #     if self.extra_image_processor is not None:
        #         g_image = np.array(frame_image)  # for grounding
        #         g_image = self.extra_image_processor.apply_image(g_image)
        #         g_pixel_values = torch.from_numpy(g_image).permute(2, 0, 1).contiguous()
        #         extra_pixel_values.append(g_pixel_values)

        if self.preprocessor is not None:
            if self.arch_type == "qwen":
                _data_dict = self.preprocessor(
                    pixel_values,
                    do_resize=True,
                    size=(self.image_size, self.image_size),
                )
                _data_dict["pixel_values"] = torch.tensor(
                    _data_dict["pixel_values"], dtype=torch.float
                )
                _data_dict["image_grid_thw"] = torch.tensor(
                    _data_dict["image_grid_thw"], dtype=torch.int
                )
                num_frame_tokens = int(
                    _data_dict["image_grid_thw"][0].prod() * (self.downsample_ratio**2)
                )
                num_frames = _data_dict["image_grid_thw"].shape[0]
                num_video_tokens = num_frame_tokens * num_frames
            elif self.arch_type == "llava":
                _data_dict = self.preprocessor(
                    pixel_values,
                    do_resize=True,
                    size=(self.image_size, self.image_size),
                )
                _data_dict["pixel_values"] = np.stack(
                    _data_dict["pixel_values"], axis=0
                )
                _data_dict["pixel_values"] = torch.tensor(
                    _data_dict["pixel_values"], dtype=torch.float
                )
            else:
                raise NotImplementedError
            data_dict.update(_data_dict)
        else:
            pixel_values = torch.stack(pixel_values, dim=0)  # (n_f, 3, h, w)
            data_dict["pixel_values"] = pixel_values
        if self.extra_image_processor is not None:
            data_dict["g_pixel_values"] = extra_pixel_values

        # process and get masks
        if data_dict["masks"] is None:
            return self.__getitem__(random.randint(0, self.real_len()))

        if num_video_tokens is not None:
            assert self.patch_token == 1
            input_str = data_dict["conversation"][0]["input"]
            input_str = input_str.replace(
                self.IMG_CONTEXT_TOKEN, self.IMG_CONTEXT_TOKEN * num_frame_tokens
            )
            assert input_str.count(self.IMG_CONTEXT_TOKEN) == num_video_tokens
            data_dict["conversation"][0]["input"] = input_str

        result = self.template_map_fn(data_dict)
        data_dict.update(result)
        result = video_lisa_encode_fn(
            data_dict, tokenizer=self.tokenizer, max_length=self.max_length
        )
        data_dict.update(result)

        data_dict["type"] = "video"
        return data_dict


class VideoViCaSMultiTurnDataset(VideoViCaSDataset):
    """ViCaS dataset class for video referential segmentation with multi-turn conversation format.

    This class extends VideoViCaSDataset to create multi-turn conversations using object referrals
    from the ViCaS dataset, similar to ReVOS dataset format.
    """

    def prepare_text(
        self, n_frames, object_referrals, caption_raw_en=None, num_image_tokens=256
    ):
        """Prepare multi-turn conversation from object referrals.

        Args:
            n_frames: Number of frames in the video
            object_referrals: List of object referrals with prompts and track_ids
            num_image_tokens: Number of image tokens per frame

        Returns:
            dict: Conversation with multi-turn format
        """
        frame_token_str = (
            f"{self.IMG_START_TOKEN}"
            f"{self.IMG_CONTEXT_TOKEN * num_image_tokens}"
            f"{self.IMG_END_TOKEN}"
        )

        questions = []
        answers = []

        # Create frame tokens string for use in both main and fallback cases
        frame_tokens = frame_token_str + "\n"
        frame_tokens = frame_tokens * n_frames
        frame_tokens = frame_tokens.strip()

        # Extract expressions from referral prompts
        prompts = [
            (
                (
                    prompt.replace("Localize the", "")
                    .replace("Segment all", "all")
                    .replace("Segment the", "")
                    .replace("What is", "")
                    .replace("Where are", "")
                    .replace("Where is", "")
                    .replace("Which", "")
                    .replace("?", "")
                    .replace(".", "")
                    .strip()
                ),
                None,
            )
            for prompt in [referral["prompt"] for referral in object_referrals]
        ]

        for expression, _ in prompts:
            # Use SEG_QUESTIONS to format the expression into a question
            question_template = random.choice(SEG_QUESTIONS)
            questions.append(question_template.format(class_name=expression.lower()))
            answers.append(
                random.choice(ANSWER_LIST).replace(
                    "[SEG]", "[SEG]" * self.num_tokens_per_expression
                )
            )

        qa_list = []
        for i, (question, answer) in enumerate(zip(questions, answers)):
            if i == 0:
                qa_list.append(
                    {"from": "human", "value": frame_tokens + " " + question}
                )
            else:
                qa_list.append({"from": "human", "value": question})
            qa_list.append({"from": "gpt", "value": answer})

        input = ""
        conversation = []
        for msg in qa_list:
            if msg["from"] == "human":
                input += msg["value"]
            elif msg["from"] == "gpt":
                conversation.append({"input": input, "output": msg["value"]})
                input = ""
            else:
                raise NotImplementedError

        # add system information
        if len(conversation) > 0:
            conversation[0].update({"system": self._system})
        else:
            # Use a general description question
            fallback_question = (
                "Could you please describe what is happening in this video?"
            )
            full_question = frame_tokens + " " + fallback_question

            conversation = [{"input": full_question, "output": caption_raw_en}]
            conversation[0].update({"system": self._system})

        return {"conversation": conversation}

    def dataset_map_fn(self, data_dict, only_gt=True, select_k=8, select_k_tarvis=8):
        """Override dataset mapping to use multi-turn conversation format.

        Args:
            data_dict: Input data dictionary
            only_gt: Whether to use only ground truth frames
            select_k: Number of frames to select for context
            select_k_tarvis: Number of frames to select for TARViS

        Returns:
            dict: Mapped data dictionary with multi-turn conversation
        """
        # Get video_id and process annotation on-demand
        video_id = data_dict["video_id"]
        annotation_path = data_dict["annotation_path"]
        # Create new barecat instance for this access
        bc = barecat.Barecat(self.barecat_path)
        with bc.open(annotation_path) as f:
            processed_annotation = json.load(f)
        processed_annotation["video_id"] = video_id

        frames = sorted(
            [
                int(seg["filename"].split(".")[0])
                for seg in processed_annotation["segmentations"]
            ]
        )
        processed_annotation["frames"] = frames

        track_masks, frame_indices = self.load_video_masks(
            processed_annotation, only_gt=only_gt
        )

        # Use the same frame selection logic as parent class
        (
            selected_frame_indexes,
            tarvis_selected_frame_indexes,
            context_num_frames,
            tarvis_context_num_frames,
        ) = select_frames(
            vid_len=len(frame_indices),
            num_frames=select_k,
            tarvis_num_frames=select_k_tarvis,
            train_mode=True,
        )

        # Get selected frame indices
        selected_frames = [frame_indices[i] for i in selected_frame_indexes]
        tarvis_selected_frames = [
            frame_indices[i] for i in tarvis_selected_frame_indexes
        ]

        for track_id, track_mask in track_masks.items():
            track_masks[track_id] = [track_mask[idx] for idx in selected_frame_indexes]

        # Prepare frame paths for selected frames
        selected_frame_paths = [
            VicasConfig["FRAME_PATH"].format(video_id=video_id, frame_id=frame_idx)
            for frame_idx in selected_frames
        ]
        tarvis_selected_frame_paths = [
            VicasConfig["FRAME_PATH"].format(video_id=video_id, frame_id=frame_idx)
            for frame_idx in tarvis_selected_frames
        ]

        # Store frame paths for later loading
        images = selected_frame_paths
        tarvis_images = tarvis_selected_frame_paths

        # Prepare masks for each object referral
        object_referrals = processed_annotation["object_referrals"]

        # Apply the same sampling as in prepare_text to ensure consistency
        if len(object_referrals) > self.select_number:
            selected_referrals = random.sample(object_referrals, self.select_number)
        else:
            selected_referrals = object_referrals

        referral_masks = []
        for referral in selected_referrals:
            track_ids = referral["track_ids"]
            # Get the masks for these track IDs, but only if they exist in track_masks
            masks_to_combine = []
            for track_id in track_ids:
                if track_id in track_masks:
                    masks_to_combine.append(track_masks[track_id])
                else:
                    # Skip this referral if any track ID is missing
                    print(f"Warning: Track ID {track_id} not found in track_masks. Skipping referral.")
                    masks_to_combine = []
                    break

            # Only process if we found valid masks
            if masks_to_combine:
                # Combine them with OR operation across all frames
                combined_masks = []
                for frame_idx in range(
                    len(masks_to_combine[0])
                ):  # assume all have same frame count
                    frame_mask = np.logical_or.reduce(
                        [mask[frame_idx] for mask in masks_to_combine]
                    )
                    combined_masks.append(frame_mask)
                referral_masks.append(np.stack(combined_masks))

        # Create multi-turn conversation using selected object referrals
        conversation_dict = self.prepare_text(
            context_num_frames,
            selected_referrals,
            caption_raw_en=processed_annotation["caption_raw_en"],
            num_image_tokens=self.patch_token,
        )

        # If conversation is None, return None to sample a different entry
        if conversation_dict is None:
            return None

        # Handle case where referral_masks might be empty
        if referral_masks:
            masks_tensor = torch.from_numpy(np.stack(referral_masks).astype(np.uint8))
        else:
            # Create empty tensor with correct shape if no valid masks found
            masks_tensor = torch.empty((0, len(selected_frames), *next(iter(track_masks.values()))[0].shape), dtype=torch.uint8)
            print("Warning: No valid referral masks found, using empty tensor")

        ret = {
            "images": images,
            "tarvis_images": tarvis_images,
            "masks": masks_tensor,
            "conversation": conversation_dict["conversation"],
        }
        return ret
