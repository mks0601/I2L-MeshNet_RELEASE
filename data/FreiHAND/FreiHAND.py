import os
import os.path as osp
import numpy as np
import torch
import cv2
import random
import json
import math
import copy
import transforms3d
from pycocotools.coco import COCO
from config import cfg
from utils.mano import MANO
from utils.preprocessing import load_img, get_bbox, process_bbox, generate_patch_image, augmentation
from utils.transforms import world2cam, cam2pixel, pixel2cam, rigid_align, transform_joint_to_other_db
from utils.vis import vis_keypoints, vis_mesh, save_obj
import transforms3d

class FreiHAND(torch.utils.data.Dataset):
    def __init__(self, transform, data_split):
        self.transform = transform
        self.data_split = data_split
        self.data_path = osp.join('..', 'data', 'FreiHAND', 'data')
        self.human_bbox_root_dir = osp.join('..', 'data', 'FreiHAND', 'rootnet_output', 'bbox_root_freihand_output.json')
        
        # MANO joint set
        self.mano = MANO()
        self.face = self.mano.face
        self.joint_regressor = self.mano.joint_regressor
        self.vertex_num = self.mano.vertex_num
        self.joint_num = self.mano.joint_num
        self.joints_name = self.mano.joints_name
        self.skeleton = self.mano.skeleton
        self.root_joint_idx = self.mano.root_joint_idx

        self.datalist = self.load_data()
        
    def load_data(self):
        if self.data_split == 'train':
            db = COCO(osp.join(self.data_path, 'freihand_train_coco.json'))
            with open(osp.join(self.data_path, 'freihand_train_data.json')) as f:
                data = json.load(f)
            
        else:
            db = COCO(osp.join(self.data_path, 'freihand_eval_coco.json'))
            with open(osp.join(self.data_path, 'freihand_eval_data.json')) as f:
                data = json.load(f)
            print("Get bounding box and root from " + self.human_bbox_root_dir)
            bbox_root_result = {}
            with open(self.human_bbox_root_dir) as f:
                annot = json.load(f)
            for i in range(len(annot)):
                bbox_root_result[str(annot[i]['image_id'])] = {'bbox': np.array(annot[i]['bbox']), 'root': np.array(annot[i]['root_cam'])}

        datalist = []
        for aid in db.anns.keys():
            ann = db.anns[aid]
            image_id = ann['image_id']
            img = db.loadImgs(image_id)[0]
            img_path = osp.join(self.data_path, img['file_name'])
            img_shape = (img['height'], img['width'])
            db_idx = str(img['db_idx'])

            if self.data_split == 'train':
                cam_param, mano_param, joint_cam = data[db_idx]['cam_param'], data[db_idx]['mano_param'], data[db_idx]['joint_3d']
                joint_cam = np.array(joint_cam).reshape(-1,3)
                bbox = process_bbox(np.array(ann['bbox']), img['width'], img['height'])
                if bbox is None: continue
                root_joint_depth = joint_cam[self.root_joint_idx][2]
                
            else:
                cam_param, scale = data[db_idx]['cam_param'], data[db_idx]['scale']
                joint_cam = np.ones((self.joint_num,3), dtype=np.float32) # dummy
                mano_param = {'pose': np.ones((48), dtype=np.float32), 'shape': np.ones((10), dtype=np.float32)}
                bbox = bbox_root_result[str(image_id)]['bbox'] # bbox should be aspect ratio preserved-extended. It is done in RootNet.
                root_joint_depth = bbox_root_result[str(image_id)]['root'][2]

            datalist.append({
                'img_path': img_path,
                'img_shape': img_shape,
                'bbox': bbox,
                'joint_cam': joint_cam,
                'cam_param': cam_param,
                'mano_param': mano_param,
                'root_joint_depth': root_joint_depth})
                
        return datalist

    def get_mano_coord(self, mano_param, cam_param):
        pose, shape, trans = mano_param['pose'], mano_param['shape'], mano_param['trans']
        mano_pose = torch.FloatTensor(pose).view(1,-1); mano_shape = torch.FloatTensor(shape).view(1,-1); # mano parameters (pose: 48 dimension, shape: 10 dimension)
        mano_trans = torch.FloatTensor(trans).view(1,3) # translation vector

        # get mesh and joint coordinates
        mano_mesh_coord, mano_joint_coord = self.mano.layer(mano_pose, mano_shape, mano_trans)
        mano_mesh_coord = mano_mesh_coord.numpy().reshape(self.vertex_num,3); mano_joint_coord = mano_joint_coord.numpy().reshape(self.joint_num,3)
        
        # milimeter -> meter
        mano_mesh_coord /= 1000; mano_joint_coord /= 1000;
        return mano_mesh_coord, mano_joint_coord, mano_pose[0].numpy(), mano_shape[0].numpy()

    def __len__(self):
        return len(self.datalist)

    def __getitem__(self, idx):
        data = copy.deepcopy(self.datalist[idx])
        img_path, img_shape, bbox, joint_cam, cam_param, mano_param = data['img_path'], data['img_shape'], data['bbox'], data['joint_cam'], data['cam_param'], data['mano_param']

        # img
        img = load_img(img_path)
        img, img2bb_trans, bb2img_trans, rot, _ = augmentation(img, bbox, self.data_split, exclude_flip=True) # FreiHAND dataset only contains right hands. do not perform flip aug.
        img = self.transform(img.astype(np.float32))/255.
        
        if self.data_split == 'train':
            # mano coordinates
            mano_mesh_cam, mano_joint_cam, mano_pose, mano_shape = self.get_mano_coord(mano_param, cam_param)
            mano_coord_cam = np.concatenate((mano_mesh_cam, mano_joint_cam))
            focal, princpt = cam_param['focal'], cam_param['princpt']
            mano_coord_img = cam2pixel(mano_coord_cam, focal, princpt)

            # affine transform x,y coordinates. root-relative depth
            mano_coord_img_xy1 = np.concatenate((mano_coord_img[:,:2], np.ones_like(mano_coord_img[:,:1])),1)
            mano_coord_img[:,:2] = np.dot(img2bb_trans, mano_coord_img_xy1.transpose(1,0)).transpose(1,0)[:,:2]
            root_joint_depth = mano_coord_cam[self.vertex_num + self.root_joint_idx][2]
            mano_coord_img[:,2] = mano_coord_img[:,2] - root_joint_depth
            mano_coord_img[:,0] = mano_coord_img[:,0] / cfg.input_img_shape[1] * cfg.output_hm_shape[2]
            mano_coord_img[:,1] = mano_coord_img[:,1] / cfg.input_img_shape[0] * cfg.output_hm_shape[1]
            mano_coord_img[:,2] = (mano_coord_img[:,2] / (cfg.bbox_3d_size / 2) + 1)/2. * cfg.output_hm_shape[0]

            # check truncation
            mano_trunc = ((mano_coord_img[:,0] >= 0) * (mano_coord_img[:,0] < cfg.output_hm_shape[2]) * \
                        (mano_coord_img[:,1] >= 0) * (mano_coord_img[:,1] < cfg.output_hm_shape[1]) * \
                        (mano_coord_img[:,2] >= 0) * (mano_coord_img[:,2] < cfg.output_hm_shape[0])).reshape(-1,1).astype(np.float32)
            
            # split mesh and joint coordinates
            mano_mesh_img = mano_coord_img[:self.vertex_num]; mano_joint_img = mano_coord_img[self.vertex_num:];
            mano_mesh_trunc = mano_trunc[:self.vertex_num]; mano_joint_trunc = mano_trunc[self.vertex_num:];
            
            # 3D data rotation augmentation
            rot_aug_mat = np.array([[np.cos(np.deg2rad(-rot)), -np.sin(np.deg2rad(-rot)), 0], 
            [np.sin(np.deg2rad(-rot)), np.cos(np.deg2rad(-rot)), 0],
            [0, 0, 1]], dtype=np.float32)
            # parameter
            mano_pose = mano_pose.reshape(-1,3)
            root_pose = mano_pose[self.root_joint_idx,:]
            root_pose, _ = cv2.Rodrigues(root_pose)
            root_pose, _ = cv2.Rodrigues(np.dot(rot_aug_mat,root_pose))
            mano_pose[self.root_joint_idx] = root_pose.reshape(3)
            mano_pose = mano_pose.reshape(-1)
            # mano coordinate
            mano_joint_cam = (mano_joint_cam - mano_joint_cam[self.root_joint_idx,None]) * 1000 # root-relative, meter to milimeter
            mano_joint_cam = np.dot(rot_aug_mat, mano_joint_cam.transpose(1,0)).transpose(1,0)

            orig_joint_img = np.zeros((self.joint_num,3), dtype=np.float32) # dummy
            orig_joint_cam = np.zeros((self.joint_num,3), dtype=np.float32) # dummy
            orig_joint_valid = np.zeros((self.joint_num,1), dtype=np.float32) # dummy
            orig_joint_trunc = np.zeros((self.joint_num,1), dtype=np.float32) # dummy

            inputs = {'img': img}
            targets = {'orig_joint_img': orig_joint_img, 'fit_joint_img': mano_joint_img, 'fit_mesh_img': mano_mesh_img, 'orig_joint_cam': orig_joint_cam, 'fit_joint_cam': mano_joint_cam, 'pose_param': mano_pose, 'shape_param': mano_shape}
            meta_info = {'orig_joint_valid': orig_joint_valid, 'orig_joint_trunc': orig_joint_trunc, 'fit_joint_trunc': mano_joint_trunc, 'fit_mesh_trunc': mano_mesh_trunc, 'is_valid_fit': float(True), 'is_3D': float(True)}
        else:
            inputs = {'img': img}
            targets = {}
            meta_info = {'bb2img_trans': bb2img_trans}

        return inputs, targets, meta_info

    def evaluate(self, outs, cur_sample_idx):

        annots = self.datalist
        sample_num = len(outs)
        eval_result = {'joint_out': [], 'mesh_out': []}
        for n in range(sample_num):
            annot = annots[cur_sample_idx + n]
            out = outs[n]
            
            # x,y: resize to input image space and perform bbox to image affine transform
            mesh_out_img = out['mesh_coord_img']
            mesh_out_img[:,0] = mesh_out_img[:,0] / cfg.output_hm_shape[2] * cfg.input_img_shape[1]
            mesh_out_img[:,1] = mesh_out_img[:,1] / cfg.output_hm_shape[1] * cfg.input_img_shape[0]
            mesh_out_img_xy1 = np.concatenate((mesh_out_img[:,:2], np.ones_like(mesh_out_img[:,:1])),1)
            mesh_out_img[:,:2] = np.dot(out['bb2img_trans'], mesh_out_img_xy1.transpose(1,0)).transpose(1,0)[:,:2]

            # z: devoxelize and translate to absolute depth
            root_joint_depth = annot['root_joint_depth']
            mesh_out_img[:,2] = (mesh_out_img[:,2] / cfg.output_hm_shape[0] * 2. - 1) * (cfg.bbox_3d_size / 2)
            mesh_out_img[:,2] = mesh_out_img[:,2] + root_joint_depth

            # camera back-projection
            cam_param = annot['cam_param']
            focal, princpt = cam_param['focal'], cam_param['princpt']
            mesh_out_cam = pixel2cam(mesh_out_img, focal, princpt)
            
            if cfg.stage == 'param':
                mesh_out_cam = out['mesh_coord_cam']
            joint_out_cam = np.dot(self.joint_regressor, mesh_out_cam)

            eval_result['mesh_out'].append(mesh_out_cam.tolist())
            eval_result['joint_out'].append(joint_out_cam.tolist())
 
            vis = False
            if vis:
                filename = annot['img_path'].split('/')[-1][:-4]

                img = load_img(annot['img_path'])[:,:,::-1]
                img = vis_mesh(img, mesh_out_img, 0.5)
                cv2.imwrite(filename + '.jpg', img)

                save_obj(mesh_out_cam, self.mano.face, filename + '.obj')

        return eval_result
    
    def print_eval_result(self, eval_result):
        output_save_path = osp.join(cfg.result_dir, 'pred.json')
        with open(output_save_path, 'w') as f:
            json.dump([eval_result['joint_out'], eval_result['mesh_out']], f)
        print('Saved at ' + output_save_path)

