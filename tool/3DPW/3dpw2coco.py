import os
import os.path as osp
import json
import pickle
import cv2
import numpy as np
import torch
from glob import glob
from tqdm import tqdm
import transforms3d
import matplotlib.pyplot as plt

def cam2pixel(cam_coord, f, c):
    x = cam_coord[:,0] / cam_coord[:,2] * f[0] + c[0]
    y = cam_coord[:,1] / cam_coord[:,2] * f[1] + c[1]
    z = cam_coord[:,2]
    return np.stack((x,y,z),1)

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

def vis_keypoints(img, kps, alpha=1):
    # Convert from plt 0-1 RGBA colors to 0-255 BGR colors for opencv.
    cmap = plt.get_cmap('rainbow')
    colors = [cmap(i) for i in np.linspace(0, 1, len(kps) + 2)]
    colors = [(c[2] * 255, c[1] * 255, c[0] * 255) for c in colors]

    # Perform the drawing on a copy of the image, to allow for blending.
    kp_mask = np.copy(img)

    # Draw the keypoints.
    for i in range(len(kps)):
        p = kps[i][0].astype(np.int32), kps[i][1].astype(np.int32)
        cv2.circle(kp_mask, p, radius=5, color=colors[i], thickness=-1, lineType=cv2.LINE_AA)

    # Blend the keypoints.
    return cv2.addWeighted(img, 1.0 - alpha, kp_mask, alpha, 0)

def vis_mesh(img, mesh_vertex, alpha=0.5):
    # Convert from plt 0-1 RGBA colors to 0-255 BGR colors for opencv.
    cmap = plt.get_cmap('rainbow')
    colors = [cmap(i) for i in np.linspace(0, 1, len(mesh_vertex))]
    colors = [(c[2] * 255, c[1] * 255, c[0] * 255) for c in colors]

    # Perform the drawing on a copy of the image, to allow for blending.
    mask = np.copy(img)

    # Draw the mesh
    for i in range(len(mesh_vertex)):
        p = mesh_vertex[i][0].astype(np.int32), mesh_vertex[i][1].astype(np.int32)
        cv2.circle(mask, p, radius=3, color=colors[i], thickness=-1, lineType=cv2.LINE_AA)

    # Blend the keypoints.
    return cv2.addWeighted(img, 1.0 - alpha, mask, alpha, 0)

import sys
smpl_path = '/home/gyeongsikmoon/workspace/smplpytorch'
sys.path.insert(0, smpl_path)
from smplpytorch.pytorch.smpl_layer import SMPL_Layer


# Create the SMPL layer
smpl_layer = SMPL_Layer(
    gender='neutral',
    model_root=smpl_path + '/smplpytorch/native/models')

root_path = '/mnt/hdd1/Data/3DPW/'
img_id = 0; annot_id = 0;

for data_split in tqdm(('train', 'validation', 'test')):

    images = []; annotations = [];
    annot_list = glob(osp.join(root_path, 'sequenceFiles', 'sequenceFiles', data_split, '*.pkl'))
    for annot in tqdm(annot_list):
        with open(annot, 'rb') as f:
            data = pickle.load(f, encoding='latin1')
        
        sequence_name = str(data['sequence'])
        smpl_betas = data['betas']
        smpl_poses = data['poses']
        smpl_trans = data['trans']
        joint_3d = data['jointPositions']
        gender = data['genders']
        campose_valid = data['campose_valid']
        extrinsics = data['cam_poses']
        intrinsics = data['cam_intrinsics']
        Rs, ts = extrinsics[:,:3,:3], extrinsics[:,:3,3]
        focal, princpt = [intrinsics[0,0], intrinsics[1,1]], [intrinsics[0,2], intrinsics[1,2]]
        joint_2d_coco = data['poses2d']
        person_num = len(smpl_betas)
        frame_num = len(Rs)
        img = cv2.imread(osp.join(root_path, 'imageFiles', sequence_name, ('image_%05d.jpg' % 0)))
        img_height, img_width, _ = img.shape

        for iid in tqdm(range(frame_num)):
            
            img_dict = {}
            img_dict['id'] = img_id
            img_dict['file_name'] = ('image_%05d.jpg' % int(iid))
            img_dict['sequence'] = sequence_name
            img_dict['frame_idx'] = int(iid)
            img_dict['width'] = img_width
            img_dict['height'] = img_height
            img_dict['cam_param'] = {'focal': focal, 'princpt': princpt}
            images.append(img_dict)

            for pid in range(person_num):
                
                if campose_valid[pid][iid] == 0:
                    continue
                if smpl_betas[pid].shape == (300,):
                    smpl_betas[pid] = smpl_betas[pid][:10]
 
                joint_cam = np.dot(Rs[iid].reshape(3,3), joint_3d[pid][iid].reshape(-1,3).transpose(1,0)).transpose(1,0) + ts[iid].reshape(-1,3)
                joint_img = cam2pixel(joint_cam, focal, princpt)
                
                # merge camera paraemter with smpl parameter (rotation)
                R = Rs[iid].reshape(3,3)
                pose = smpl_poses[pid][iid].reshape(-1,3)
                root_pose = pose[0,:]
                angle = np.linalg.norm(root_pose)
                root_pose = transforms3d.axangles.axangle2mat(root_pose / angle, angle)
                root_pose = np.dot(R, root_pose)
                axis, angle = transforms3d.axangles.mat2axangle(root_pose)
                root_pose = axis*angle
                pose[0] = root_pose
                pose = pose.reshape(-1)
                
                # merge camera parameter with smpl parameter (translation)
                pose = torch.from_numpy(pose).view(1,-1).float()
                shape = torch.from_numpy(smpl_betas[pid]).view(1,-1).float()
                mesh_coord, joint_coord = smpl_layer(pose, shape)
                root_joint_coord = joint_coord[0,0].numpy().reshape(1,3)
                t = ts[iid].reshape(3)
                trans = smpl_trans[pid][iid].reshape(3)
                trans = np.dot(R, trans[:,None]).reshape(1,3) + t.reshape(1,3)
                trans = trans - root_joint_coord + np.dot(R, root_joint_coord.transpose(1,0)).transpose(1,0)

                annot_dict = {}
                annot_dict['id'] = annot_id
                annot_dict['image_id'] = img_id
                annot_dict['person_id'] = pid
                annot_dict['joint_cam'] = joint_cam.tolist()
                annot_dict['joint_img'] = joint_img[:,:2].tolist()
                annot_dict['fitted_joint_cam'] = (joint_coord.numpy()[0] + trans.reshape(1,3)).tolist()
                annot_dict['openpose_result'] = joint_2d_coco[pid][iid].transpose(1,0).tolist()
                annot_dict['smpl_param'] = {'shape': shape.numpy().reshape(-1).tolist(), \
                                    'pose': pose.numpy().reshape(-1).tolist(), \
                                    'trans': trans.reshape(-1).tolist(), \
                                    'gender': 'female' if gender[pid] == 'f' else 'male'}
                
                # bbox
                joint_valid = ((joint_img[:,0] >= 0) * (joint_img[:,0] < img_width) * (joint_img[:,1] >= 0) * (joint_img[:,1] < img_height)).astype(np.float32)
                if np.sum(joint_valid) == 0: 
                    continue
                bbox = get_bbox(joint_img, joint_valid)
                annot_dict['bbox'] = bbox.tolist() # x, y, w, h

                annotations.append(annot_dict)
                annot_id += 1
                
                """
                # for debug
                img = cv2.imread(osp.join(root_path, 'imageFiles', img_dict['sequence'], img_dict['file_name']))
                #coord = joint_2d_coco[pid][iid].transpose(1,0)
                coord = joint_img[:,:2]
                for j in range(len(coord)):
                    cv2.circle(img, (int(coord[j][0]), int(coord[j][1])), radius=10, color=(255,0,0), thickness=-1)
                    cv2.imwrite(str(annot_id) + '_' + str(pid) + '_' + str(j) + '.jpg', img)
                """
                
                """
                # for debug
                pose = torch.FloatTensor(annot_dict['smpl_param']['pose']).reshape(1,-1)
                shape = torch.FloatTensor(annot_dict['smpl_param']['shape']).reshape(1,-1)
                trans = torch.FloatTensor(annot_dict['smpl_param']['trans']).reshape(1,-1)
                mesh_coord, joint_coord = smpl_layer(pose, shape, trans)
                mesh_coord = mesh_coord.numpy().reshape(-1,3)
                joint_coord = joint_coord.numpy().reshape(-1,3)

                #joint_cam = np.array(joint_cam).reshape(-1,3)
                #print(np.mean((joint_coord - joint_cam)))

                cam_param = {k: np.array(img_dict['cam_param'][k]) for k in img_dict['cam_param']}
                focal, princpt = cam_param['focal'], cam_param['princpt']

                mesh_coord = cam2pixel(mesh_coord, focal, princpt)
                joint_coord = cam2pixel(joint_coord, focal, princpt)
                
                img = cv2.imread(osp.join(root_path, 'imageFiles', img_dict['sequence'], img_dict['file_name']))
                mesh_img = vis_mesh(img, mesh_coord)
                joint_img = vis_keypoints(img, joint_coord)

                mesh_img = mesh_img[int(bbox[1]):int(bbox[1]+bbox[3]), int(bbox[0]):int(bbox[0]+bbox[2]), :]
                joint_img = joint_img[int(bbox[1]):int(bbox[1]+bbox[3]), int(bbox[0]):int(bbox[0]+bbox[2]), :]

                cv2.imwrite(str(img_id) + '_mesh.jpg', mesh_img)
                cv2.imwrite(str(img_id) + '_joint.jpg', joint_img)
                """


            img_id += 1
    
    output = {'images': images, 'annotations': annotations}
    output_path = osp.join(root_path, '3DPW_' + data_split + '.json')
    with open(output_path, 'w') as f:
        json.dump(output, f)
    print('Saved at ' + output_path)

    
