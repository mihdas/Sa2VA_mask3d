import copy
import json
import os
import random
from pathlib import Path

import barecat
import cv2
import numpy as np
import torch
import torchvision.transforms as T
from PIL import Image
from pycocotools import mask as maskUtils
from torch.utils.data import Dataset
from torchvision.transforms.functional import InterpolationMode
from xtuner.registry import BUILDER
from xtuner.utils import DEFAULT_IMAGE_TOKEN, IGNORE_INDEX, PROMPT_TEMPLATE

from .encode_fn import video_lisa_encode_fn
from .utils import dynamic_preprocess, dynamic_preprocess_mask

ANSWER_LIST = [
    "It is [SEG].",
    "Sure, [SEG].",
    "Sure, it is [SEG].",
    "Sure, the segmentation result is [SEG].",
    "[SEG].",
]


class SA1BDataset(Dataset):
    """SA-1B dataset with 10 prompt templates for segmentation tasks using barecat storage"""

    os.environ["TOKENIZERS_PARALLELISM"] = "true"
    IMG_CONTEXT_TOKEN = "<IMG_CONTEXT>"
    IMG_START_TOKEN = "<img>"
    IMG_END_TOKEN = "</img>"

    IMAGENET_MEAN = (0.485, 0.456, 0.406)
    IMAGENET_STD = (0.229, 0.224, 0.225)

    # Original 10 prompt templates (backup - commented out due to tokenizer conflicts)
    ORIGINAL_PROMPT_TEMPLATES = [
        "Generate the segmentation mask for the object located within the bounding box <box>{}</box> and at the points <point>{}</point> in the provided image.",
        "Extract the mask for the object inside the bounding box <box>{}</box> and corresponding to the points <point>{}</point> in the given image.",
        "Create the mask for the object found in the region defined by bounding box <box>{}</box> and identified by the points <point>{}</point> in this image.",
        "Identify the object mask within the bounding box <box>{}</box> and marked by the points <point>{}</point> in the image.",
        "Acquire the segmentation mask of the object enclosed by bounding box <box>{}</box> and situated at the points <point>{}</point> in the image.",
        "Determine the mask for the object that is located within the area specified by bounding box <box>{}</box> and identified by the points <point>{}</point> in the image.",
        "Produce the mask for the object situated in the bounding box <box>{}</box> and at the points <point>{}</point> within the image.",
        "Locate the mask for the object that lies inside the boundaries <box>{}</box> and corresponds to the points <point>{}</point> in the image.",
        "Segment the mask for the object found within the bounds of bounding box <box>{}</box> and marked at the points <point>{}</point> in the given image.",
        "Outline the mask of the object residing within the box <box>{}</box> and corresponding to the points <point>{}</point> in the image.",
    ]

    # Simplified 10 prompt templates - bbox only to avoid tokenizer conflicts
    PROMPT_TEMPLATES = [
        "Generate the segmentation mask for the object located within the bounding box <box>{}</box> in the provided image.",
        "Extract the mask for the object inside the bounding box <box>{}</box> in the given image.",
        "Create the mask for the object found in the region defined by bounding box <box>{}</box> in this image.",
        "Identify the object mask within the bounding box <box>{}</box> in the image.",
        "Acquire the segmentation mask of the object enclosed by bounding box <box>{}</box> in the image.",
        "Determine the mask for the object that is located within the area specified by bounding box <box>{}</box> in the image.",
        "Produce the mask for the object situated in the bounding box <box>{}</box> within the image.",
        "Locate the mask for the object that lies inside the boundaries <box>{}</box> in the image.",
        "Segment the mask for the object found within the bounds of bounding box <box>{}</box> in the given image.",
        "Outline the mask of the object residing within the box <box>{}</box> in the image.",
    ]

    def __init__(
        self,
        image_folder,
        tokenizer=None,
        max_length=8196,
        special_tokens=None,
        template_map_fn=None,
        extra_image_processor=None,
        lazy=True,
        repeats=1,
        single_image_mode=False,
        select_number=5,
        num_tokens_per_expression=1,
        **kwargs,
    ):
        super().__init__()
        assert lazy
        self.lazy = lazy
        self.max_length = max_length
        self.num_tokens_per_expression = num_tokens_per_expression
        self.select_number = select_number

        # Set up barecat paths
        self.image_folder = image_folder
        self.barecat_path = None
        self.bc = None

        # Initialize barecat connection
        self._setup_barecat()

        # Load sample IDs using barecat listing
        self.sample_ids = self._get_sample_ids()
        self.samples = [
            (f"{sample_id}.jpg", f"{sample_id}.json") for sample_id in self.sample_ids
        ]

        self.tokenizer = BUILDER.build(tokenizer)

        if special_tokens is not None:
            self.tokenizer.add_tokens(special_tokens, special_tokens=True)

        self.template_map_fn = template_map_fn
        if isinstance(self.template_map_fn, dict) and self.lazy:
            _type = self.template_map_fn["type"]
            del self.template_map_fn["type"]
            self.template_map_fn = _type(**self.template_map_fn)

        if extra_image_processor is not None:
            self.extra_image_processor = BUILDER.build(extra_image_processor)

        self.repeats = repeats
        self._system = ""

        self.min_dynamic_patch = 1
        self.max_dynamic_patch = 12
        self.downsample_ratio = 0.5
        self.image_size = 448
        self.use_thumbnail = True
        patch_size = 14
        self.patch_token = int(
            (self.image_size // patch_size) ** 2 * (self.downsample_ratio**2)
        )

        self.transformer = T.Compose(
            [
                T.Lambda(lambda img: img.convert("RGB") if img.mode != "RGB" else img),
                T.Resize(
                    (self.image_size, self.image_size),
                    interpolation=InterpolationMode.BICUBIC,
                ),
                T.ToTensor(),
                T.Normalize(mean=self.IMAGENET_MEAN, std=self.IMAGENET_STD),
            ]
        )

        self.single_image_mode = single_image_mode
        self._max_refetch = 1000

        print(
            f"SA1B dataset initialized with {len(self.samples)} samples from barecat archive."
        )

    def _setup_barecat(self):
        """Set up barecat connection"""
        image_folder_path = Path(self.image_folder)

        # Check if the provided path is already a barecat file
        if image_folder_path.suffix == ".barecat":
            self.barecat_path = image_folder_path
        else:
            # Try to find barecat file in the directory
            barecat_filename = image_folder_path.name.lower()
            if barecat_filename.endswith("_barecat"):
                barecat_filename = barecat_filename[:-8] + ".barecat"

            potential_barecat_path = image_folder_path.parent / barecat_filename
            if potential_barecat_path.exists():
                self.barecat_path = potential_barecat_path
            else:
                # Look for any .barecat file in the parent directory
                for bc_file in image_folder_path.parent.glob("*.barecat"):
                    self.barecat_path = bc_file
                    break

        if self.barecat_path is None:
            raise FileNotFoundError(f"No barecat file found for {self.image_folder}")

        # Store barecat path for loading
        self.barecat_path = self.barecat_path

    def _get_sample_ids(self):
        """Get sample IDs by listing files in barecat archive"""
        sample_ids = set()

        # List all files in the barecat archive
        with barecat.Barecat(self.barecat_path, readonly=True) as bc:
            # Only check annotations directory - these are the valid samples with annotations
            annotations_files = bc.listdir("annotations")
            for file_path_str in annotations_files:
                file_path_str = str(file_path_str)
                if file_path_str.endswith(".json"):
                    sample_id = file_path_str.replace(".json", "")
                    sample_ids.add(sample_id)

        return sorted(list(sample_ids))

    def _load_annotation(self, annotation_path):
        """Load annotation from barecat archive"""
        with barecat.Barecat(self.barecat_path, readonly=True) as bc:
            # Try annotations/ prefix first
            with bc.open(f"annotations/{annotation_path}", "rb") as f:
                return json.load(f)

    def _load_image(self, image_path):
        """Load image from barecat archive"""
        with barecat.Barecat(self.barecat_path, readonly=True) as bc:
            with bc.open(f"images/{image_path}", "rb") as f:
                return Image.open(f).convert("RGB")

    def __len__(self):
        return len(self.samples) * self.repeats

    def annToRLE(self, ann, height, width):
        """Convert annotation to RLE format"""
        segm = ann["segmentation"]
        if isinstance(segm, list):
            rles = maskUtils.frPyObjects(segm, height, width)
            rle = maskUtils.merge(rles)
        elif isinstance(segm["counts"], list):
            # uncompressed RLE
            rle = maskUtils.frPyObjects(segm, height, width)
        else:
            rle = ann["segmentation"]
        return rle

    def annToMask(self, ann, height, width):
        """Convert annotation to binary mask"""
        rle = self.annToRLE(ann, height, width)
        m = maskUtils.decode(rle)
        return m

    def load_mask(self, annotation_path):
        """Load instance masks for the given image"""
        annotation_data = self._load_annotation(annotation_path)
        image_info = annotation_data["image"]
        annotations = annotation_data["annotations"]

        instance_masks = []
        class_ids = []
        boxes = []
        point_coords = []
        predicted_ious = []

        for annotation in annotations:
            m = self.annToMask(annotation, image_info["height"], image_info["width"])
            # Skip objects that are too small
            if m.sum() < 100:  # Minimum size threshold
                continue

            class_id = annotation["id"]
            instance_masks.append(m)
            class_ids.append(class_id)
            boxes.append(annotation.get("bbox", []))
            point_coords.append(annotation.get("point_coords", []))

            # Extract predicted IoU for sampling interesting masks
            # predicted_iou = annotation["area"] / (image_info["height"] * image_info["width"])
            predicted_iou = annotation["stability_score"]
            # predicted_iou = annotation["predicted_iou"]
            predicted_ious.append(predicted_iou)

        mask = np.stack(instance_masks)
        class_ids = np.array(class_ids, dtype=np.int32)
        boxes = np.array(boxes, dtype=np.float32)
        point_coords = np.array(point_coords, dtype=np.float32)
        predicted_ious = np.array(predicted_ious, dtype=np.float32)

        return mask, boxes, point_coords, class_ids, predicted_ious

    def prepare_text(
        self, n_frames, expressions, num_image_tokens=256, n_fast_images=0
    ):
        frame_token_str = (
            f"{self.IMG_START_TOKEN}"
            f"{self.IMG_CONTEXT_TOKEN * num_image_tokens * n_frames}"
            f"{self.IMG_END_TOKEN}"
        )
        # f"{self.IMG_CONTEXT_TOKEN * num_image_tokens}"
        # * n_frames Modification for image mode

        questions = []
        answers = []
        for i, exp in enumerate(expressions):
            # Use existing SA1B prompt templates - expressions contain formatted bbox/point strings
            prompt = (
                random.choice(self.ORIGINAL_PROMPT_TEMPLATES).format(exp[0], exp[1])
                if len(exp) == 2
                else random.choice(self.PROMPT_TEMPLATES).format(exp[0])
            )
            questions.append(prompt)
            answers.append(
                random.choice(ANSWER_LIST).replace(
                    "[SEG]", "[SEG]" * self.num_tokens_per_expression
                )
            )

        qa_list = []
        for i, (question, answer) in enumerate(zip(questions, answers)):
            if i == 0:
                frame_tokens = frame_token_str + "\n"
                frame_tokens = frame_tokens # * n_frames Modification for image mode
                frame_tokens = frame_tokens.strip()
                qa_list.append({"from": "human", "value": frame_tokens + question})
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
        conversation[0].update({"system": self._system})
        return {"conversation": conversation}

    def _calculate_mask_complexity(self, mask):
        """Calculate the contour complexity (number of points) of a binary mask"""
        mask_uint8 = mask.astype(np.uint8) * 255
        contours, _ = cv2.findContours(
            mask_uint8, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE
        )
        total_complexity = sum(len(contour) for contour in contours)
        return total_complexity

    def _calculate_iou(self, box1, box2):
        """Calculate IoU between two boxes in [x1, y1, x2, y2] format"""
        # Determine the coordinates of the intersection rectangle
        x1 = max(box1[0], box2[0])
        y1 = max(box1[1], box2[1])
        x2 = min(box1[2], box2[2])
        y2 = min(box1[3], box2[3])

        # Calculate area of intersection
        intersection = max(0, x2 - x1) * max(0, y2 - y1)

        # Calculate area of both boxes
        area1 = (box1[2] - box1[0]) * (box1[3] - box1[1])
        area2 = (box2[2] - box2[0]) * (box2[3] - box2[1])

        # Calculate union area
        union = area1 + area2 - intersection

        # Avoid division by zero
        if union == 0:
            return 0.0

        return intersection / union

    def _nms(self, boxes, scores, iou_threshold=0.5):
        """Non-Maximum Suppression implementation in NumPy"""
        if len(boxes) == 0:
            return []

        # Convert to numpy arrays if needed
        boxes = np.array(boxes)
        scores = np.array(scores)

        # Get coordinates of bounding boxes
        x1 = boxes[:, 0]
        y1 = boxes[:, 1]
        x2 = boxes[:, 2]
        y2 = boxes[:, 3]

        # Calculate areas
        areas = (x2 - x1) * (y2 - y1)

        # Sort by score (descending)
        order = np.argsort(scores)[::-1]

        keep = []
        while order.size > 0:
            # Pick the box with highest score
            i = order[0]
            keep.append(i)

            # Calculate IoU with remaining boxes
            xx1 = np.maximum(x1[i], x1[order[1:]])
            yy1 = np.maximum(y1[i], y1[order[1:]])
            xx2 = np.minimum(x2[i], x2[order[1:]])
            yy2 = np.minimum(y2[i], y2[order[1:]])

            w = np.maximum(0.0, xx2 - xx1)
            h = np.maximum(0.0, yy2 - yy1)
            intersection = w * h

            # Calculate IoU
            union = areas[i] + areas[order[1:]] - intersection
            iou = intersection / union

            # Keep boxes with IoU less than threshold
            inds = np.where(iou <= iou_threshold)[0]
            order = order[inds + 1]

        return keep

    def _sample_non_overlapping_masks(
        self, bboxes, stability_scores, num_to_select, masks
    ):
        """
        Sample masks based on combined complexity and size probability using NMS.

        Strategy:
        1. Filter out sky, ground, and too large masks
        2. Apply NMS to remove overlapping masks
        3. Sample based on complexity and mask area probabilities

        Args:
            bboxes: array of bounding boxes [x_min, y_min, width, height] in absolute pixels
            stability_scores: array of stability scores (unused)
            num_to_select: number of masks to select
            masks: array of binary masks for complexity calculation

        Returns:
            selected_indices: list of selected mask indices, or None if no valid masks found
        """
        if len(bboxes) == 0:
            return None
        if num_to_select >= len(bboxes):
            return list(range(len(bboxes)))

        # Get image dimensions from the first mask
        if len(masks) > 0:
            mask_height, mask_width = masks[0].shape
            image_size = (mask_width, mask_height)
        else:
            # Fallback - estimate from bboxes
            max_x = max(box[0] + box[2] for box in bboxes if len(box) >= 4)
            max_y = max(box[1] + box[3] for box in bboxes if len(box) >= 4)
            image_size = (max_x, max_y)

        image_width, image_height = image_size

        # Normalize boxes to [0,1] range for processing
        normalized_boxes = []
        for box in bboxes:
            if len(box) == 4:
                x_min, y_min, width, height = box
                # Convert to normalized coordinates
                x_min_norm = x_min / image_width
                y_min_norm = y_min / image_height
                width_norm = width / image_width
                height_norm = height / image_height
                x_max_norm = x_min_norm + width_norm
                y_max_norm = y_min_norm + height_norm
                normalized_boxes.append(
                    [x_min_norm, y_min_norm, x_max_norm, y_max_norm]
                )
            else:
                normalized_boxes.append([0, 0, 1, 1])

        normalized_boxes = np.array(normalized_boxes)

        # Filter out sky, ground, and too large masks
        valid_indices = []
        for i, box in enumerate(normalized_boxes):
            x_min, y_min, x_max, y_max = box
            box_width = x_max - x_min
            box_height = y_max - y_min
            area = box_width * box_height

            # Filter sky masks (top 20% with large width)
            is_sky = y_min < 0.2 and box_width > 0.6
            # Filter ground masks (bottom 20% with large width)
            is_ground = y_max > 0.8 and box_width > 0.6
            # Filter too large/small masks
            is_too_large = area > 0.8
            is_too_small = area < 0.001

            if not (is_sky or is_ground or is_too_large or is_too_small):
                valid_indices.append(i)

        # Calculate complexity scores and mask areas for all filtered masks
        complexities = []
        mask_areas = []

        for i in valid_indices:
            complexity = self._calculate_mask_complexity(masks[i])
            complexities.append(complexity)

            # Calculate mask area (number of pixels in mask)
            area = np.sum(masks[i])
            mask_areas.append(area)

        complexities = np.array(complexities)
        mask_areas = np.array(mask_areas)

        # Apply NMS using our NumPy implementation
        boxes_for_nms = []
        for i in valid_indices:
            box = bboxes[i]
            if len(box) == 4:
                x_min, y_min, width, height = box
                boxes_for_nms.append([x_min, y_min, x_min + width, y_min + height])
            else:
                boxes_for_nms.append([0, 0, 1, 1])

        # Apply NMS with IoU threshold
        nms_indices = self._nms(boxes_for_nms, complexities, iou_threshold=0.5)

        # Get the indices that survived NMS
        survived_indices = [valid_indices[i] for i in nms_indices]
        survived_complexities = [complexities[i] for i in nms_indices]
        survived_areas = [mask_areas[i] for i in nms_indices]

        # If NMS removed too many, fall back to valid_indices
        if len(survived_indices) < num_to_select:
            survived_indices = valid_indices
            survived_complexities = complexities
            survived_areas = mask_areas

        # If no valid indices after filtering, return None
        if len(survived_indices) == 0:
            return None

        # Normalize complexities and areas to probabilities
        survived_complexities = np.array(survived_complexities)
        survived_areas = np.array(survived_areas)

        # Avoid division by zero
        if np.sum(survived_complexities) == 0:
            complexity_probs = np.ones_like(survived_complexities) / len(
                survived_complexities
            )
        else:
            complexity_probs = survived_complexities / np.sum(survived_complexities)

        if np.sum(survived_areas) == 0:
            area_probs = np.ones_like(survived_areas) / len(survived_areas)
        else:
            area_probs = survived_areas / np.sum(survived_areas)

        # Combine complexity and area probabilities (you can adjust the weights)
        # 0.3 for complexity, 0.7 for area - emphasize larger masks
        combined_probs = 0.3 * complexity_probs + 0.7 * area_probs

        # Normalize combined probabilities to sum to 1 with proper error handling
        prob_sum = np.sum(combined_probs)
        if prob_sum > 0:
            combined_probs = combined_probs / prob_sum
            # Ensure probabilities sum to exactly 1 due to floating point precision
            combined_probs = combined_probs / np.sum(combined_probs)
        else:
            # Fallback to uniform distribution if all probabilities are zero
            combined_probs = np.ones_like(combined_probs) / len(combined_probs)

        # Sample without replacement using combined probabilities
        selected_indices = np.random.choice(
            survived_indices,
            size=min(num_to_select, len(survived_indices)),
            replace=False,
            p=combined_probs,
        ).tolist()

        return selected_indices

    def __getitem__(self, index):
        index = index % len(self.samples)

        img_path, annotation_path = self.samples[index]

        image = self._load_image(img_path)
        masks, bboxes, point_coords, class_ids, predicted_ious = self.load_mask(
            annotation_path
        )

        # Sample non-overlapping masks with balanced size-based probabilities
        num_to_select = min(self.select_number, masks.shape[0])
        selected_indices = self._sample_non_overlapping_masks(
            bboxes, predicted_ious, num_to_select, masks
        )

        # If no valid masks found, try a random sample
        if (selected_indices is None) or (len(selected_indices) == 0):
            random_index = np.random.randint(0, len(self.samples))
            return self.__getitem__(random_index)

        selected_masks = []
        expressions = []

        for i, obj_idx in enumerate(selected_indices):
            obj_mask = masks[obj_idx, :, :]

            # Process mask dynamically
            processed_masks = dynamic_preprocess_mask(
                obj_mask,
                self.min_dynamic_patch,
                self.max_dynamic_patch,
                self.image_size,
                self.use_thumbnail,
            )
            processed_masks = np.stack(processed_masks)
            selected_masks.append(processed_masks)

            box = bboxes[i]
            point_list = point_coords[i]

            # Convert bbox from [x_min, y_min, width, height] to [x_min, y_min, x_max, y_max]
            # and scale to [0-1000] range for regex parsing
            h, w = obj_mask.shape
            if len(box) == 4:
                x_min, y_min, width, height = box
                # Convert to [x_min, y_min, x_max, y_max] format and scale to 0-1000
                scaled_box = [
                    (x_min / w) * 1000,
                    (y_min / h) * 1000,
                    ((x_min + width) / w) * 1000,
                    ((y_min + height) / h) * 1000,
                ]
            else:
                scaled_box = [0, 0, 1000, 1000]  # fallback

            # Normalize point coordinates to [0-1] range for tensors
            normalized_points = []
            if len(point_list) > 0:
                for point in point_list:
                    if len(point) >= 2:
                        x, y = float(point[0]), float(point[1])  # Convert to float
                        normalized_points.append([(x / w) * 1000, (y / h) * 1000])

            # Format box for prompt - use double brackets for regex parsing, scale to 0-1000
            box_str = f"[{int(scaled_box[0])}, {int(scaled_box[1])}, {int(scaled_box[2])}, {int(scaled_box[3])}]"
            # Wrap in double brackets as expected by regex
            box_str = f"[{box_str}]"

            # Format points with clean decimal places
            if normalized_points:
                points_formatted = [[int(p[0]), int(p[1])] for p in normalized_points]
                points_str = f"{points_formatted}"
            else:
                points_str = "[]"

            # Store formatted expressions for prepare_text method
            expressions.append((box_str, points_str))

        # Stack masks
        masks_tensor = torch.stack(
            [torch.from_numpy(mask) for mask in selected_masks], dim=0
        )

        # Process image with dynamic_preprocess like other datasets
        images = dynamic_preprocess(
            image,
            self.min_dynamic_patch,
            self.max_dynamic_patch,
            self.image_size,
            self.use_thumbnail,
        )

        # Process images with transformer
        pixel_values = [self.transformer(image) for image in images]
        pixel_values = torch.stack(pixel_values)

        # Prepare text using the same method as reference datasets
        n_frames = len(images)
        text_dict = self.prepare_text(
            n_frames, expressions, num_image_tokens=self.patch_token
        )

        # Prepare return dictionary with conversation
        ret = {
            "pixel_values": pixel_values,
            "masks": masks_tensor,
            "conversation": text_dict["conversation"],
            "image_path": img_path,
        }

        # Apply template mapping function
        result = self.template_map_fn(ret)
        ret.update(result)

        # Apply video_lisa_encode_fn like reference datasets
        result = video_lisa_encode_fn(
            ret,
            tokenizer=self.tokenizer,
            max_length=self.max_length,
            with_image_token=True,
        )
        ret.update(result)

        if hasattr(self, "extra_image_processor"):
            g_image = np.array(image)  # for grounding
            g_image = self.extra_image_processor.apply_image(g_image)
            g_pixel_values = torch.from_numpy(g_image).permute(2, 0, 1).contiguous()
            ret["g_pixel_values"] = g_pixel_values

        return ret

    @property
    def modality_length(self):
        return [100] * len(self)  # Default modality length
