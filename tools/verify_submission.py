import argparse
import json
from pathlib import Path


def load_json(file_name):
    with open(Path(file_name)) as f:
        return json.load(f)


def get_annotation_map(ann_file):
    subset_expressions_by_video = load_json(ann_file)["videos"]
    annotation_map = {}
    for vid, vid_data in subset_expressions_by_video.items():
        vid_frames = vid_data["frames"]
        vid_len = len(vid_frames)
        for exp_id in vid_data["expressions"].keys():
            annotation_map[(vid, f"{int(exp_id):02d}")] = vid_len
    return annotation_map


def get_prediction_map(prediction_dir):
    prediction_dir = Path(prediction_dir)
    prediction_map = {}

    def count_files_in_folder(folder_path):
        path = Path(folder_path)
        return sum(1 for item in path.iterdir() if item.is_file())

    for video_key_path in prediction_dir.glob("*"):
        vid = video_key_path.name
        for expression_id_path in video_key_path.glob("*"):
            exp_id = f"{int(expression_id_path.name):02d}"
            vid_len = count_files_in_folder(expression_id_path)
            prediction_map[(vid, exp_id)] = vid_len
    return prediction_map


def parse_args():
    parser = argparse.ArgumentParser(
        description="Compare annotation and prediction directories."
    )
    parser.add_argument(
        "--annotation_file",
        required=True,
        default="/nodes/cristal/work/nekrasov/data/language-data/mevis/valid/meta_expressions.json",
        help="Path to the annotation file in JSON format.",
    )
    parser.add_argument(
        "--prediction_dir", required=True, help="Directory containing predictions."
    )
    args = parser.parse_args()
    return args


if __name__ == "__main__":
    args = parse_args()

    annotation_map = get_annotation_map(args.annotation_file)
    prediction_map = get_prediction_map(args.prediction_dir)

    # Extract sets of videos
    annotation_videos = {vid for (vid, _) in annotation_map.keys()}
    prediction_videos = {vid for (vid, _) in prediction_map.keys()}

    # 1. Find videos that are completely missing from predictions
    missing_videos = annotation_videos - prediction_videos

    # 2. Compare video_expressions
    annotation_keys = set(annotation_map.keys())
    prediction_keys = set(prediction_map.keys())

    # Video_expressions that exist in annotation but not in prediction
    missing_expressions = annotation_keys - prediction_keys

    # Video_expressions that exist in both but differ in frame count
    frame_mismatches = set()
    for key in annotation_keys.intersection(prediction_keys):
        if annotation_map[key] != prediction_map[key]:
            frame_mismatches.add((key, annotation_map[key], prediction_map[key]))

    # Print results
    if not missing_videos and not missing_expressions and not frame_mismatches:
        print("All videos and expressions match perfectly.")
        exit(0)
    else:
        if missing_videos:
            print("Videos missing from predictions:")
            for v in missing_videos:
                print(f"  - {v}")

        if missing_expressions:
            print("\nVideo_expressions missing from predictions:")
            for vid, exp in missing_expressions:
                print(f"  - {vid}_{exp} (expected {annotation_map[(vid, exp)]} frames)")

        if frame_mismatches:
            print("\nVideo_expressions with frame count mismatches:")
            for (vid, exp), ann_len, pred_len in frame_mismatches:
                print(f"  - {vid}_{exp}: annotation={ann_len}, prediction={pred_len}")
        exit(1)
