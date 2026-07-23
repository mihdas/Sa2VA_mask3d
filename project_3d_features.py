
import torch
import torch.nn.functional as F
import numpy as np
import cv2
import sonata
import json
import tqdm
import os

from mask3d import get_model, load_mesh, prepare_data

def read_matrix_txt(path):
    with open(path, 'r') as f:
        lines = f.readlines()
        lines = [line.strip() for line in lines]
        matrix = np.array([[float(num) for num in line.split()] for line in lines])
    return matrix

def read_axis_align_matrix(file_path):
    axis_align_matrix = None
    with open(file_path, "r") as f:
        for line in f:
            line_content = line.strip()
            if 'axisAlignment' in line_content:
                axis_align_matrix = [float(x) for x in line_content.strip('axisAlignment = ').split(' ')]
                axis_align_matrix = np.array(axis_align_matrix).reshape((4, 4))
                break
    return axis_align_matrix

with open("expressions/scanrefer_train_24_debug.json","r") as f:
    data=json.load(f)['videos']

dct={}
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
ln = torch.nn.LayerNorm([32,32], device=device) 

model = get_model("/home/vipradas/Thesis/scannet200_val.ckpt")

model.to(device)


for scene in tqdm.tqdm(data.keys()):
    tensor=[]
    scene_id=scene.split("/")[0]
    if scene_id in dct: 
        continue
    else:
        dct[scene_id]=1
    for frame in data[scene]['frames']:
        
        
        img_bgr = cv2.imread(f"/globalwork/vipradas/scannet_images/{scene_id}/color/{frame}.jpg")
        img = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
        #print(img.shape)
        pose=np.linalg.inv(read_matrix_txt(f"/globalwork/vipradas/scannet_images/{scene_id}/pose/{frame}.txt"))
        K=read_matrix_txt(f"/globalwork/vipradas/scannet_images/{scene_id}/intrinsics/intrinsic_color.txt")[:3,:3]
        alignment_matrix=read_axis_align_matrix(f"/globalwork/datasets/scannet/scannet/scans/{scene_id}/{scene_id}.txt")
        #aligned_xyz=torch.load(f"/globalwork/vipradas/scannet_m3drefer/val/{scene_id}.pth")['xyz']
        # aligned_xyz=torch.load(f"/home/vipradas/Thesis/coordinates.pth")[:,1:].numpy()
        # pcd_features=torch.load(f"/home/vipradas/Thesis/pcd_features.pth")
        # mask_features=torch.load(f"/home/vipradas/Thesis/mask_features.pth")
        # pos_encodings=torch.load(f"/home/vipradas/Thesis/pos_encodings.pth")
        pointcloud_file = f"/globalwork/datasets/scannet/scannet/scans/{scene_id}/{scene_id}_vh_clean_2.ply"
        mesh = load_mesh(pointcloud_file)
        sparse_tensor, pts_3d, clrs, fts, unique_map, inverse_map_m3d = prepare_data(np.asarray(mesh.vertices), np.asarray(mesh.vertex_colors), device)
        pcd_features, mask_features, pos_encodings, aligned_xyz = model(sparse_tensor, raw_coordinates=fts)
        feat=torch.cat((pcd_features, mask_features, pos_encodings), dim=-1)
        print(feat.shape)

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


        #inverse_matrix = np.linalg.inv(alignment_matrix)

        num_points = aligned_xyz.shape[0]
        ones_column = np.ones((num_points, 1))
        points_homogeneous = np.hstack([aligned_xyz, ones_column])
        #points_3d_h = (inverse_matrix @ points_homogeneous.T).T
        
        points_cam_h = (pose @ points_homogeneous.T).T  # (N, 4)
        points_cam = points_cam_h[:, :3] 

        mask =points_cam[:, 2] > 0
        #mask1.append(mask)

        points_cam = points_cam[mask]
        feat=feat[mask]
        aligned_xyz=aligned_xyz[mask]
        
        # Project: p_h = K * P_c
        points_cam_T = points_cam.T                      # (3, N)
        pixels_h = K @ points_cam_T                      # (3, N)

        # Perspective divide
        pixels = pixels_h[:2, :] / pixels_h[:2, :]        # (2, N)
        coords = pixels.T

        img_mask= (
            (coords[:, 0] >= 0) & (coords[:, 0] < img.shape[1]) &
            (coords[:, 1] >= 0) & (coords[:, 1] < img.shape[0])             
        )
        #mask2.append(img_mask)
        

        H, W, C = img.shape[0], img.shape[1], feat.shape[1]
        img_feat = torch.zeros(H, W, C, dtype=feat.dtype, device=feat.device, requires_grad=False)
        global_coord = torch.zeros(H, W, 3, dtype=feat.dtype, requires_grad=False)

        feat=feat[img_mask]
        aligned_xyz=aligned_xyz[img_mask]
        # print(feat.shape)
        coords=coords[img_mask]
        ys = coords[:, 0].astype(int)
        xs = coords[:, 1].astype(int)
        #locs.append(coords)
        #print(mask.shape, img_mask.shape, coords.shape)
        #print(np.min(ys),np.min(xs))
        #print(np.max(ys),np.max(xs))
    
        img_feat[xs,ys] = feat

        global_coord[xs,ys]= torch.from_numpy(aligned_xyz).float()
        global_coord=global_coord.to(device=feat.device)

        img_feat = img_feat.permute(2, 0, 1).unsqueeze(0)
        global_coord = global_coord.permute(2, 0, 1).unsqueeze(0)

        img_feat = F.adaptive_max_pool2d(img_feat, output_size=(48, 48))
        global_coord = F.adaptive_max_pool2d(global_coord, output_size=(16,16))
        global_coord = global_coord.permute(0,2,3,1).squeeze(0)
        

        #img_feat = ln(img_feat)
        img_feat = img_feat.reshape(1, 352, 16, 3, 16, 3)

        img_feat = img_feat.permute(0, 3, 5, 1, 2, 4)
        img_feat = img_feat.reshape(1, 3168, 16, 16)
        img_feat = img_feat.squeeze(0).permute(1, 2, 0)

        #img_feat=torch.cat((img_feat, global_coord), dim=2)        
        img_feat = img_feat.reshape(256, 3168)
        img_feat = img_feat.detach()
        tensor.append(img_feat)
        # img_feat_resized = F.interpolate(
        #     img_feat,
        #     size=(256, 2048),
        #     mode='bilinear',
        #     align_corners=False
        # )   # shape: ( 1088, 256, 2048)
        # # print(img_feat_resized.shape)
        # img_feat_resized=img_feat_resized.squeeze(0)
        # f2d.append(img_feat_resized)
    
    # mask1=np.stack(mask1)
    # mask2=np.stack(mask2)
    # locs=np.stack(locs)


    # print(mask1.shape, mask2.shape, locs.shape)
    tensor=torch.stack(tensor)
    print(tensor.shape)
    torch.save(tensor, f'/globalwork/vipradas/mask3dshuffle/{scene_id}.pth')
