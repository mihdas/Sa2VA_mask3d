
import json
import torch
import numpy as np
from sklearn.cluster import DBSCAN
from scipy.spatial import ConvexHull
import sys

# Redirect stdout to file
sys.stdout = open('output.log', 'w')

def get_8_corners(min_xyz, max_xyz):
    points = np.array([
        [min_xyz[0], min_xyz[1], min_xyz[2]],
        [min_xyz[0], min_xyz[1], max_xyz[2]],
        [min_xyz[0], max_xyz[1], min_xyz[2]],
        [min_xyz[0], max_xyz[1], max_xyz[2]],
        [max_xyz[0], min_xyz[1], min_xyz[2]],
        [max_xyz[0], min_xyz[1], max_xyz[2]],
        [max_xyz[0], max_xyz[1], min_xyz[2]],
        [max_xyz[0], max_xyz[1], max_xyz[2]],
    ])
    return points

def iou(box1, box2):
    """
    Calculate IoU between two 3D bounding boxes defined by 8 corner points.
    
    Args:
        box1: (8, 3) array of corner points
        box2: (8, 3) array of corner points
    Returns:
        iou: float
    """
    def box_volume(box):
        try:
            hull = ConvexHull(box)
            return hull.volume
        except:
            return 0

    def intersection_volume(box1, box2):
        # Combine points and compute convex hull of intersection
        # via sampling points inside both boxes
        combined = np.vstack([box1, box2])
        
        try:
            hull1 = ConvexHull(box1)
            hull2 = ConvexHull(box2)
        except:
            return 0.0

        # Sample points inside the combined bounding region
        mins = combined.min(axis=0)
        maxs = combined.max(axis=0)
        
        num_samples = 100000
        samples = np.random.uniform(mins, maxs, size=(num_samples, 3))
        
        def in_hull(points, hull):
            return np.all(points @ hull.equations[:, :3].T + hull.equations[:, 3] <= 1e-10, axis=1)
        
        inside_both = in_hull(samples, hull1) & in_hull(samples, hull2)
        
        # Scale by the total sampled volume
        total_volume = np.prod(maxs - mins)
        return (inside_both.sum() / num_samples) * total_volume

    vol1 = box_volume(box1)
    vol2 = box_volume(box2)
    inter = intersection_volume(box1, box2)
    union = vol1 + vol2 - inter

    return inter / union if union > 0 else 0.0


with open("/home/vipradas/Thesis/Sa2VA_p/raw_jsons/ScanRefer_filtered_val.json") as file: #TODO Change dataset
    json_data=json.load(file)

for idx,line in enumerate(json_data):

    scene_id=line['scene_id']
    obj_id=int(line['object_id'])

    data=torch.load(f"/globalwork/vipradas/scannet_m3drefer/val/{scene_id}.pth")
    locs_in=data['xyz']
    gt=data["aabb_corner_xyz"][obj_id]


    pred=np.load(f"/home/vipradas/preds_val/pred{idx+1}.npy")
    pred=torch.from_numpy(pred)
    pred=torch.sigmoid(pred)>0.6
    masked_points=locs_in[pred]
    if np.count_nonzero(pred) ==0:
            pred_min_xyz = np.array([0,0,0])
            pred_max_xyz = np.array([0,0,0])
        # else:
        #     pred_min_xyz = masked_points.min(axis=0)
        #     pred_max_xyz = masked_points.max(axis=0)
    else:
        clustering = DBSCAN(eps=0.05, min_samples=10).fit(masked_points)
        labels = clustering.labels_
        unique_labels, counts = np.unique(labels[labels != -1], return_counts=True)
        cluster_labels = labels[labels != -1]
        if cluster_labels.size == 0:
            largest_label = None   # or handle accordingly
            pred_min_xyz = np.array([0,0,0])
            pred_max_xyz = np.array([0,0,0])
        else:
            #unique_labels, counts = np.unique(cluster_labels, return_counts=True)
            largest_label = unique_labels[np.argmax(counts)]
            largest_label = unique_labels[np.argmax(counts)]
            largest_cluster_mask = labels == largest_label
            largest_cluster_points = masked_points[largest_cluster_mask]
            pred_min_xyz = largest_cluster_points.min(axis=0)
            pred_max_xyz = largest_cluster_points.max(axis=0)
    pred_bbox=get_8_corners(pred_min_xyz, pred_max_xyz)

    bbox_iou=iou(gt,pred_bbox)
    print(bbox_iou)


