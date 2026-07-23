import argparse
import os

from PIL import Image
from transformers import AutoModelForCausalLM, AutoTokenizer

try:
    from mmengine.visualization import Visualizer
except ImportError:
    Visualizer = None
    print("Warning: mmengine is not installed, visualization is disabled.")


def parse_args():
    parser = argparse.ArgumentParser(description='Video Reasoning Segmentation')
    parser.add_argument('image_folder', help='Path to image file')
    parser.add_argument('--model_path', default="ByteDance/Sa2VA-8B")
    parser.add_argument('--work-dir', default=None, help='The dir to save results.')
    parser.add_argument('--text', type=str, default="<image>Please describe the video content.")
    parser.add_argument('--select', type=int, default=-1)
    args = parser.parse_args()
    return args


def visualize(pred_mask, image_path, work_dir):
    visualizer = Visualizer()
    img = cv2.imread(image_path)
    visualizer.set_image(img)
    visualizer.draw_binary_masks(pred_mask, colors='g', alphas=0.4)
    visual_result = visualizer.get_image()

    output_path = os.path.join(work_dir, os.path.basename(image_path))
    cv2.imwrite(output_path, visual_result)

if __name__ == "__main__":
    cfg = parse_args()
    torch.cuda.memory_summary()
    torch.cuda.empty_cache()
    torch.cuda.memory_summary()
    gc.collect()
    model_path = cfg.model_path
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        torch_dtype="auto",
        device_map="cuda",
        trust_remote_code=True
    )
    tokenizer = AutoTokenizer.from_pretrained(cfg.model_path, trust_remote_code=True)
    model.eval()

    # 2. Read the "expressions.json" from val folder
    val_path = Path(cfg.val_folder)
    ann_file = val_path / "meta_expressions.json"
    if not ann_file.exists():
        raise FileNotFoundError(
            f"Could not find {ann_file}. Is the val folder correct?"
        )

    with open(ann_file, "r") as f:
        annotation_data = json.load(f)

    # The JSON structure is typically: {"videos": {video_id: {frames: [...], expressions: {...}}}}
    videos_info = annotation_data["videos"]
    print(f"Found {len(videos_info)} videos in {ann_file}")

    # 3. Prepare output directory
    os.makedirs(cfg.work_dir, exist_ok=True)
    print(f"Results will be saved to: {cfg.work_dir}\n")

    # 4. Loop over each video, then each expression
    with torch.inference_mode():
        for video_id, vid_data in videos_info.items():
            # Frames for this video
            vid_frames = sorted(vid_data["frames"])  # e.g., ["0001", "0002", ...]

            # Load all frames as PIL
            # JPEGImages/{video_id}/{frame_name}.jpg
            frame_paths = [
                os.path.join(val_path, "JPEGImages", video_id, f"{fn}.jpg")
                for fn in vid_frames
            ]
            frames_pil = []
            for fp in frame_paths:
                if not os.path.exists(fp):
                    raise FileNotFoundError(f"Frame not found: {fp}")
                frames_pil.append(Image.open(fp).convert("RGB"))

            # Go through each expression in this video
            expressions_dict = vid_data["expressions"]
            for exp_id, exp_info in expressions_dict.items():
                out_dir = os.path.join(cfg.work_dir, video_id, exp_id)
                if os.path.exists(out_dir):
                    continue
                # We'll form a text prompt for the model
                # The default text is cfg.text, e.g., "<image>Please describe the video content."
                # Some code might append the expression, but you can adapt as needed.
                # For example:
                # text_prompt = f"<image> Describe: {exp_info['exp']}"
                # Or simply use cfg.text.
                text_prompt = f"<image>Please segment {exp_info['exp']}."

                print(
                    f"Processing video={video_id}, exp_id={exp_id} with {len(frames_pil)} frames."
                )

                # 5. Run the model for the entire video
                result = model.predict_forward(
                    video=frames_pil, text=text_prompt, tokenizer=tokenizer
                )

                # 6. Print or log the raw language model output
                prediction_text = result["prediction"]
                print(f"  Model output: {prediction_text}")

                # 7. Check for segmentations, overlay, and save
                if "[SEG]" in prediction_text and Visualizer is not None:
                    # The model might return multiple sets of masks,
                    # typically `result['prediction_masks']`
                    # Here we assume there's at least one set
                    seg_idx = 0
                    if "prediction_masks" not in result:
                        print(
                            "  prediction_masks missing from result; cannot visualize."
                        )
                        continue
                    pred_masks = result["prediction_masks"][seg_idx]

                    # Create output folder: {work_dir}/{video_id}/{exp_id}
                    out_dir = os.path.join(cfg.work_dir, video_id, exp_id)
                    os.makedirs(out_dir, exist_ok=True)

                    # For each frame, overlay mask and save
                    for frame_idx, frame_name in enumerate(vid_frames):
                        pred_mask = pred_masks[frame_idx]
                        out_path = os.path.join(out_dir, f"{frame_name}.png")
                        # visualize_and_save(pred_mask, frame_paths[frame_idx], out_path)
                        mask = pred_mask.astype(np.float32)
                        mask = Image.fromarray(mask * 255).convert("L")
                        mask.save(out_path)

                else:
                    print(
                        "  No segmentation or Visualizer unavailable; skipping mask overlay."
                    )

    print("\nDone processing all videos and expressions!")


if __name__ == "__main__":
    main()
