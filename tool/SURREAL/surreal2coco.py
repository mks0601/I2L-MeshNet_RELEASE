import os
import os.path as osp
import glob
import scipy.io as sio
import numpy as np
import cv2
import json
import transforms3d
import math
import torch
import torch.nn as nn
from tqdm import tqdm
import matplotlib.pyplot as plt

# Returns intrinsic camera matrix
# Parameters are hard-coded since all SURREAL images use the same.
def get_intrinsic():
    # These are set in Blender (datageneration/main_part1.py)
    res_x_px = 320  # *scn.render.resolution_x
    res_y_px = 240  # *scn.render.resolution_y
    f_mm = 60  # *cam_ob.data.lens
    sensor_w_mm = 32  # *cam_ob.data.sensor_width
    sensor_h_mm = sensor_w_mm * res_y_px / res_x_px  # *cam_ob.data.sensor_height (function of others)

    scale = 1  # *scn.render.resolution_percentage/100
    skew = 0  # only use rectangular pixels
    pixel_aspect_ratio = 1

    # From similar triangles:
    # sensor_width_in_mm / resolution_x_inx_pix = focal_length_x_in_mm / focal_length_x_in_pix
    fx_px = f_mm * res_x_px * scale / sensor_w_mm
    fy_px = f_mm * res_y_px * scale * pixel_aspect_ratio / sensor_h_mm

    # Center of the image
    u = res_x_px * scale / 2
    v = res_y_px * scale / 2

    # Intrinsic camera matrix
    K = np.array([[fx_px, skew, u], [0, fy_px, v], [0, 0, 1]])

    focal = np.array([fx_px,fy_px])
    princpt = np.array([u,v])
    return focal, princpt

# Returns extrinsic camera matrix
#   T : translation vector from Blender (*cam_ob.location)
#   RT: extrinsic computer vision camera matrix
#   Script based on https://blender.stackexchange.com/questions/38009/3x4-camera-matrix-from-blender-camera
def get_extrinsic(T):
    # Take the first 3 columns of the matrix_world in Blender and transpose.
    # This is hard-coded since all images in SURREAL use the same.
    R_world2bcam = np.array([[0, 0, 1], [0, -1, 0], [-1, 0, 0]]).transpose()
    # *cam_ob.matrix_world = Matrix(((0., 0., 1, params['camera_distance']),
    #                               (0., -1, 0., -1.0),
    #                               (-1., 0., 0., 0.),
    #                               (0.0, 0.0, 0.0, 1.0)))

    # Convert camera location to translation vector used in coordinate changes
    T_world2bcam = -1 * np.dot(R_world2bcam, T)

    # Following is needed to convert Blender camera to computer vision camera
    R_bcam2cv = np.array([[1, 0, 0], [0, -1, 0], [0, 0, -1]])

    # Build the coordinate transform matrix from world to computer vision camera
    R_world2cv = np.dot(R_bcam2cv, R_world2bcam)
    T_world2cv = np.dot(R_bcam2cv, T_world2bcam)

    return R_world2cv, T_world2cv

def get_bbox(joint_img, joint_valid):

    x_img, y_img = joint_img[:,0], joint_img[:,1]
    x_img = x_img[joint_valid==1]; y_img = y_img[joint_valid==1];
    xmin = min(x_img); ymin = min(y_img); xmax = max(x_img); ymax = max(y_img);

    x_center = (xmin+xmax)/2.; width = xmax-xmin;
    xmin = x_center - 0.5*width*1.2
    xmax = x_center + 0.5*width*1.2
    
    y_center = (ymin+ymax)/2.; height = ymax-ymin;
    ymin = y_center - 0.5*height*1.2
    ymax = y_center + 0.5*height*1.2

    bbox = np.array([xmin, ymin, xmax - xmin, ymax - ymin]).astype(np.float32)
    return bbox

def rigid_transform_3D(A, B):
    centroid_A = np.mean(A, axis = 0)
    centroid_B = np.mean(B, axis = 0)
    H = np.dot(np.transpose(A - centroid_A), B - centroid_B)
    U, s, V = np.linalg.svd(H)
    R = np.dot(np.transpose(V), np.transpose(U))
    if np.linalg.det(R) < 0:
        print('a')
        V[2] = -V[2]
        R = np.dot(np.transpose(V), np.transpose(U))
    t = -np.dot(R, np.transpose(centroid_A)) + np.transpose(centroid_B)
    return R, t

def vis_keypoints(img, mesh_vertex, alpha=0.5, rad = 3):
    # Convert from plt 0-1 RGBA colors to 0-255 BGR colors for opencv.
    cmap = plt.get_cmap('rainbow')
    colors = [cmap(i) for i in np.linspace(0, 1, len(mesh_vertex))]
    colors = [(c[2] * 255, c[1] * 255, c[0] * 255) for c in colors]

    # Perform the drawing on a copy of the image, to allow for blending.
    mask = np.copy(img)

    # Draw the mesh
    for i in range(len(mesh_vertex)):
        p = mesh_vertex[i][0].astype(np.int32), mesh_vertex[i][1].astype(np.int32)
        cv2.circle(mask, p, radius=rad, color=colors[i], thickness=-1, lineType=cv2.LINE_AA)

    # Blend the keypoints.
    return cv2.addWeighted(img, 1.0 - alpha, mask, alpha, 0)

def save_obj(v, f, file_name='output.obj'):
    obj_file = open(file_name, 'w')
    for i in range(len(v)):
        obj_file.write('v ' + str(v[i][0]) + ' ' + str(v[i][1]) + ' ' + str(v[i][2]) + '\n')
    for i in range(len(f)):
        obj_file.write('f ' + str(f[i][0]+1) + '/' + str(f[i][0]+1) + ' ' + str(f[i][1]+1) + '/' + str(f[i][1]+1) + ' ' + str(f[i][2]+1) + '/' + str(f[i][2]+1) + '\n')
    obj_file.close()

import sys
smpl_path = '/home/gyeongsikmoon/workspace/smplpytorch'
sys.path.insert(0, smpl_path)
from smplpytorch.pytorch.smpl_layer import SMPL_Layer
smpl_layer = {k: SMPL_Layer(gender=k, model_root=smpl_path + '/smplpytorch/native/models') for k in ('male', 'female', 'neutral')}
flip_pairs = ( (1,2), (4,5), (7,8), (10,11), (13,14), (16,17), (18,19), (20,21), (22,23) ) 

root_path = '/mnt/hdd1/Data/SURREAL'
data_splits = glob.glob(osp.join(root_path, '*'))
data_splits = [name for name in data_splits if osp.isdir(name)]
data_splits = [name for name in data_splits if 'test' not in name] # exclude test. use val as test set following bodynet paper

with open('bodynet_val_set') as f:
    bodynet_val_set = f.readlines()
with open('bodynet_val_idx') as f:
    bodynet_val_idx = f.readlines()
    bodynet_val_idx = bodynet_val_idx[0].split(',')
    bodynet_val_idx = [int(idx)-1 for idx in bodynet_val_idx]
    bodynet_val_set = [bodynet_val_set[idx] for idx in bodynet_val_idx]

for data_split in tqdm(data_splits):

    images = []
    annotations = []
    img_id = 0; annot_id = 0;

    run_names = glob.glob(osp.join(data_split, '*'))
    for run_name in tqdm(run_names):

        seq_names = glob.glob(osp.join(run_name, '*'))
        for seq_name in tqdm(seq_names):

            clip_names = glob.glob(osp.join(seq_name, '*.mp4'))
            annot_names = [name[:-4] + '_info.mat' for name in clip_names]
            
            for i in range(len(clip_names)):
                """
                # ffmpeg-based video to image. It cannot know frame idxs :( Instead, use opencv.
                clip_name = clip_names[i]
                img_name = clip_name[:-4] + '_%03d.jpg' # couting start from 1
                cmd = 'ffmpeg -i ' + clip_name + ' -qscale:v 1 -vf fps=1 ' + img_name # video to images (1 fps)
                print(cmd)
                os.system(cmd)
                """
                clip_name = clip_names[i]

                # use bodynet val set for val set
                is_included = False
                if 'val' == data_split.split('/')[-1]:
                    for data in bodynet_val_set:
                        splitted = data.split('/')
                        _run_name, _seq_name, _clip_name = splitted[-3], splitted[-2], splitted[-1][:-1]
                        _clip_name = _clip_name[:_clip_name.find('mp4')+3]
                        if _run_name == run_name.split('/')[-1] and _seq_name == seq_name.split('/')[-1] and _clip_name == clip_name.split('/')[-1]:
                            is_included = True
                            break
                    if not is_included:
                        continue

                ### video to image part
                video = cv2.VideoCapture(clip_name)
                fps = video.get(cv2.CAP_PROP_FPS)
                frame_num = int(video.get(cv2.CAP_PROP_FRAME_COUNT))
                video_len = frame_num/fps
 
                # skip single frame videos
                if frame_num == 1:
                    continue
               
                target_fps = 1
                cur_frame_idx = 0
                frame_idx_save = []
                success = True
                while(success):
                    success, frame = video.read() 
                    
                    # extract middle frame in validation set
                    if 'val' == data_split.split('/')[-1]:
                        if cur_frame_idx != frame_num//2:
                            cur_frame_idx += 1
                            continue
                    # extract frames to be target fps
                    elif cur_frame_idx % (int(fps) // target_fps) != 0:
                        cur_frame_idx += 1
                        continue

                    if success: 
                        img_name = clip_name[:-4] + '_%03d.jpg' % cur_frame_idx
                        #cv2.imwrite(img_name, frame) 
                        frame_idx_save.append(cur_frame_idx)
                        cur_frame_idx += 1
                ###
                
                _run_name = run_name.split('/')[-1]
                _seq_name = seq_name.split('/')[-1]
                _clip_name = clip_name.split('/')[-1][:-4]

                annot_name = annot_names[i]
                annot = sio.loadmat(annot_name)
                for frame_idx in frame_idx_save:
                    img_dict = {}
                    img_dict['id'] = img_id
                    img_dict['width'] = 320
                    img_dict['height'] = 240
                    img_dict['file_name'] = osp.join(_run_name, _seq_name, _clip_name) + '_%03d.jpg' % frame_idx
                    focal, princpt = get_intrinsic()
                    camLoc = np.array(annot['camLoc'], dtype=np.float32)
                    cam_R, cam_t = get_extrinsic(camLoc)
                    img_dict['cam_param'] = {'focal': focal.tolist(), 'princpt': princpt.tolist()}

                    joint_img = annot['joints2D'].transpose(2,1,0)[frame_idx].reshape(-1,2) # shape: (joint_num, 2)
                    joint_world = annot['joints3D'].transpose(2,1,0)[frame_idx].reshape(-1,3) # shape: (joint_num, 3)

                    # joint_img and joint_world in the original annotation are flipped (don't know why)
                    for pair in flip_pairs:
                        joint_img[pair[0]], joint_img[pair[1]] = joint_img[pair[1]].copy(), joint_img[pair[0]].copy()
                        joint_world[pair[0]], joint_world[pair[1]] = joint_world[pair[1]].copy(), joint_world[pair[0]].copy()

                    joint_cam = np.dot(cam_R.reshape(3,3), joint_world.transpose(1,0)).transpose(1,0) + cam_t.reshape(1,3)
                    
                    """
                    # for debug
                    x = joint_cam[:,0] / joint_cam[:,2] * focal[0] + princpt[0]
                    y = joint_cam[:,1] / joint_cam[:,2] * focal[1] + princpt[1]
                    _joint_img = np.concatenate((x[:,None], y[:,None]),1)
                    _joint_img = _joint_img[5:6,:]
                    img = cv2.imread(osp.join(root_path, data_split, _run_name, _seq_name, _clip_name) + '_%03d.jpg' % frame_idx)
                    img = vis_keypoints(img, _joint_img, 1.0)
                    cv2.imwrite(str(img_id) + '_joint.jpg', img)
                    """
       
                    annot_dict = {}
                    annot_dict['id'] = annot_id
                    annot_dict['image_id'] = img_id
                    annot_dict['joint_img'] = joint_img.tolist()
                    annot_dict['joint_cam'] = joint_cam.tolist()

                    # bbox
                    joint_valid = ((joint_img[:,0] >= 0) * (joint_img[:,0] < img_dict['width']) * (joint_img[:,1] >= 0) * (joint_img[:,1] < img_dict['height'])).reshape(-1)
                    if np.sum(joint_valid) == 0: continue
                    bbox = get_bbox(joint_img, joint_valid)
                    annot_dict['bbox'] = bbox.tolist()
                    
                    # smpl parameters
                    smpl_pose = torch.from_numpy(annot['pose'].transpose(1,0)[frame_idx].reshape(1,-1)) # smpl pose param. shape: (72)
                    smpl_shape = torch.from_numpy(annot['shape'].transpose(1,0)[frame_idx].reshape(1,-1)) # smpl shape param. shape: (10)
                    gender = 'female' if int(annot['gender'][frame_idx][0]) == 0 else 'male'

                    # smpl parameters in the original annotation are flipped (don't know why)
                    smpl_pose = smpl_pose.view(-1,3)
                    for pair in flip_pairs:
                        smpl_pose[pair[0], :], smpl_pose[pair[1], :] = smpl_pose[pair[1], :].clone(), smpl_pose[pair[0], :].clone()
                    smpl_pose[:,1:3] *= -1; # multiply -1 to y and z axis of axis-angle
                    smpl_pose = smpl_pose.view(1,-1)

                    # transform smpl parameters to camera coordinate system
                    mesh_coord, joint_coord = smpl_layer[gender](smpl_pose, smpl_shape)
                    cam_R, cam_t = rigid_transform_3D(joint_coord[0].numpy(), joint_cam)
                    
                    """
                    # for debug
                    save_obj(mesh_coord.numpy().reshape(-1,3), smpl_layer[gender].th_faces.numpy(), str(img_id) + '.obj')
                    """
                    
                    joint_coord = joint_coord[0].numpy()
                    smpl_pose = smpl_pose.view(-1,3)
                    root_pose = smpl_pose[0].numpy()
                    angle = np.linalg.norm(root_pose)
                    root_pose = transforms3d.axangles.axangle2mat(root_pose / angle, angle)
                    root_pose = np.dot(cam_R, root_pose)
                    axis, angle = transforms3d.axangles.mat2axangle(root_pose)
                    root_pose = axis*angle
                    smpl_pose[0] = torch.from_numpy(root_pose)
 
                    smpl_trans = -joint_coord[0].reshape(1,3) + np.dot(cam_R, joint_coord[0].reshape(3,1)).reshape(1,3) + cam_t.reshape(1,3)
                   
                    """
                    # for debug
                    smpl_pose = smpl_pose.view(1,-1)
                    smpl_trans = torch.from_numpy(smpl_trans).view(1,-1)
                    mesh_coord, joint_coord = smpl_layer[gender](smpl_pose, smpl_shape, smpl_trans)
                    mesh_coord = mesh_coord[0].numpy()
                    mesh_coord[:,0] = mesh_coord[:,0] / mesh_coord[:,2] * focal[0] + princpt[0]
                    mesh_coord[:,1] = mesh_coord[:,1] / mesh_coord[:,2] * focal[1] + princpt[1]
                    img = cv2.imread(osp.join(root_path, data_split, _run_name, _seq_name, _clip_name) + '_%03d.jpg' % frame_idx)
                    img = vis_keypoints(img, mesh_coord, 1.0, rad=1)
                    cv2.imwrite(str(img_id) + '_mesh.jpg', img)
                    """

                    smpl_pose = smpl_pose.view(-1).numpy().tolist()
                    smpl_shape = smpl_shape.view(-1).numpy().tolist()
                    smpl_trans = smpl_trans.reshape(-1).tolist()
                    smpl_param = {'pose': smpl_pose, 'shape': smpl_shape, 'trans': smpl_trans, 'gender': gender}
                    annot_dict['smpl_param'] = smpl_param
                    
                    img_id += 1
                    annot_id += 1
                    images.append(img_dict)
                    annotations.append(annot_dict)
    
    output_dict = {'images': images, 'annotations': annotations}
    output_name = osp.join(root_path, data_split.split('/')[-1] + '.json')
    with open(output_name,'w') as f:
        json.dump(output_dict, f)
    print(output_name + ' is saved')
