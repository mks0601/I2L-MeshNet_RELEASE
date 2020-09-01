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
from utils.smpl import SMPL
from utils.preprocessing import load_img, get_bbox, process_bbox, generate_patch_image, augmentation
from utils.transforms import cam2pixel, pixel2cam, rigid_align
from utils.vis import vis_keypoints, vis_mesh, save_obj

class PW3D(torch.utils.data.Dataset):
    def __init__(self, transform, data_split):
        self.transform = transform
        self.data_split = data_split
        self.data_path = osp.join('..', 'data', 'PW3D', 'data')
        self.human_bbox_root_dir = osp.join('..', 'data', 'PW3D', 'rootnet_output', 'bbox_root_pw3d_output.json')
       
        # SMPL joint set
        self.smpl = SMPL()
        self.face = self.smpl.face
        self.joint_regressor = self.smpl.joint_regressor
        self.vertex_num = self.smpl.vertex_num
        self.joint_num = self.smpl.joint_num
        self.joints_name = self.smpl.joints_name
        self.skeleton = self.smpl.skeleton
        self.root_joint_idx = self.smpl.root_joint_idx
        self.face_kps_vertex = self.smpl.face_kps_vertex

        # H36M joint set
        self.h36m_joints_name = ('Pelvis', 'R_Hip', 'R_Knee', 'R_Ankle', 'L_Hip', 'L_Knee', 'L_Ankle', 'Torso', 'Neck', 'Nose', 'Head_top', 'L_Shoulder', 'L_Elbow', 'L_Wrist', 'R_Shoulder', 'R_Elbow', 'R_Wrist')
        self.h36m_root_joint_idx = self.h36m_joints_name.index('Pelvis')
        self.h36m_eval_joint = (1, 2, 3, 4, 5, 6, 8, 10, 11, 12, 13, 14, 15, 16)
        self.h36m_joint_regressor = np.load(osp.join('..', 'data', 'Human36M', 'J_regressor_h36m_correct.npy'))

        self.datalist = self.load_data()

    def load_data(self):

        db = COCO(osp.join(self.data_path, '3DPW_' + self.data_split + '.json'))
        if self.data_split == 'test' and not cfg.use_gt_info:
            print("Get bounding box and root from " + self.human_bbox_root_dir)
            bbox_root_result = {}
            with open(self.human_bbox_root_dir) as f:
                annot = json.load(f)
            for i in range(len(annot)):
                ann_id = str(annot[i]['ann_id'])
                bbox_root_result[ann_id] = {'bbox': np.array(annot[i]['bbox']), 'root': np.array(annot[i]['root_cam'])}
        else:
            print("Get bounding box and root from groundtruth")

        datalist = []
        for aid in db.anns.keys():
            ann = db.anns[aid]
            image_id = ann['image_id']
            img = db.loadImgs(image_id)[0]
            img_width, img_height = img['width'], img['height']
            sequence_name = img['sequence']
            img_name = img['file_name']
            img_path = osp.join(self.data_path, 'imageFiles', sequence_name, img_name)
            cam_param = {k: np.array(v, dtype=np.float32) for k,v in img['cam_param'].items()}
            smpl_param = ann['smpl_param']
            if self.data_split == 'test' and not cfg.use_gt_info:
                bbox = bbox_root_result[str(aid)]['bbox'] # bbox should be aspect ratio preserved-extended. It is done in RootNet.
                root_joint_depth = bbox_root_result[str(aid)]['root'][2]
            else:
                bbox = process_bbox(np.array(ann['bbox']), img['width'], img['height'])
                if bbox is None: continue
                root_joint_depth = None

            datalist.append({
                'img_path': img_path,
                'img_shape': (img_height, img_width),
                'bbox': bbox,
                'smpl_param': smpl_param,
                'cam_param': cam_param,
                'root_joint_depth': root_joint_depth})
            
        return datalist

    def get_smpl_coord(self, smpl_param):
        pose, shape, trans, gender = smpl_param['pose'], smpl_param['shape'], smpl_param['trans'], smpl_param['gender']
        smpl_pose = torch.FloatTensor(pose).view(1,-1); smpl_shape = torch.FloatTensor(shape).view(1,-1); # smpl parameters (pose: 72 dimension, shape: 10 dimension)
        smpl_trans = torch.FloatTensor(trans).view(-1,3) # translation vector from smpl coordinate to 3dpw camera coordinate
       
        # get mesh and joint coordinates
        smpl_mesh_coord, smpl_joint_coord = self.smpl.layer[gender](smpl_pose, smpl_shape, smpl_trans)

        return smpl_mesh_coord[0].numpy(), smpl_joint_coord[0].numpy()

    def __len__(self):
        return len(self.datalist)

    def __getitem__(self, idx):
        data = copy.deepcopy(self.datalist[idx])
        img_path, bbox, smpl_param = data['img_path'], data['bbox'], data['smpl_param']
        
        # img
        img = load_img(img_path)
        img, img2bb_trans, bb2img_trans, _, _ = augmentation(img, bbox, self.data_split)
        img = self.transform(img.astype(np.float32))/255.

        # smpl coordinates
        smpl_mesh_cam, smpl_joint_cam = self.get_smpl_coord(smpl_param)
        
        inputs = {'img': img}
        targets = {'fit_mesh_coord_cam': smpl_mesh_cam}
        meta_info = {'bb2img_trans': bb2img_trans}
        return inputs, targets, meta_info

    def evaluate(self, outs, cur_sample_idx):
        
        annots = self.datalist
        sample_num = len(outs)
        eval_result = {'mpjpe_lixel': [], 'pa_mpjpe_lixel': [], 'mpjpe_param': [], 'pa_mpjpe_param': []}
        for n in range(sample_num):
            annot = annots[cur_sample_idx + n]
            out = outs[n]
            
            # h36m joint from gt mesh
            mesh_gt_cam = out['mesh_coord_cam_target']
            pose_coord_gt_h36m = np.dot(self.h36m_joint_regressor, mesh_gt_cam)
            depth_gt_h36m = pose_coord_gt_h36m[self.h36m_root_joint_idx,2]
            pose_coord_gt_h36m = pose_coord_gt_h36m - pose_coord_gt_h36m[self.h36m_root_joint_idx,None] # root-relative
            pose_coord_gt_h36m = pose_coord_gt_h36m[self.h36m_eval_joint,:]
            
            # mesh from lixel
            # x,y: resize to input image space and perform bbox to image affine transform
            mesh_out_img = out['mesh_coord_img']
            mesh_out_img[:,0] = mesh_out_img[:,0] / cfg.output_hm_shape[2] * cfg.input_img_shape[1]
            mesh_out_img[:,1] = mesh_out_img[:,1] / cfg.output_hm_shape[1] * cfg.input_img_shape[0]
            mesh_out_img_xy1 = np.concatenate((mesh_out_img[:,:2], np.ones_like(mesh_out_img[:,:1])),1)
            mesh_out_img[:,:2] = np.dot(out['bb2img_trans'], mesh_out_img_xy1.transpose(1,0)).transpose(1,0)[:,:2]
            # z: devoxelize and translate to absolute depth
            if cfg.use_gt_info:
                root_joint_depth = depth_gt_h36m
            else:
                root_joint_depth = annot['root_joint_depth']
            mesh_out_img[:,2] = (mesh_out_img[:,2] / cfg.output_hm_shape[0] * 2. - 1) * (cfg.bbox_3d_size / 2)
            mesh_out_img[:,2] = mesh_out_img[:,2] + root_joint_depth
            # camera back-projection
            cam_param = annot['cam_param']
            focal, princpt = cam_param['focal'], cam_param['princpt']
            mesh_out_cam = pixel2cam(mesh_out_img, focal, princpt)
            


            # h36m joint from lixel mesh
            pose_coord_out_h36m = np.dot(self.h36m_joint_regressor, mesh_out_cam)
            pose_coord_out_h36m = pose_coord_out_h36m - pose_coord_out_h36m[self.h36m_root_joint_idx,None] # root-relative
            pose_coord_out_h36m = pose_coord_out_h36m[self.h36m_eval_joint,:]
            pose_coord_out_h36m_aligned = rigid_align(pose_coord_out_h36m, pose_coord_gt_h36m)
            eval_result['mpjpe_lixel'].append(np.sqrt(np.sum((pose_coord_out_h36m - pose_coord_gt_h36m)**2,1)).mean() * 1000) # meter -> milimeter
            eval_result['pa_mpjpe_lixel'].append(np.sqrt(np.sum((pose_coord_out_h36m_aligned - pose_coord_gt_h36m)**2,1)).mean() * 1000) # meter -> milimeter

            # h36m joint from parameter mesh
            if cfg.stage == 'param':
                mesh_out_cam = out['mesh_coord_cam']
                pose_coord_out_h36m = np.dot(self.h36m_joint_regressor, mesh_out_cam)
                pose_coord_out_h36m = pose_coord_out_h36m - pose_coord_out_h36m[self.h36m_root_joint_idx,None] # root-relative
                pose_coord_out_h36m = pose_coord_out_h36m[self.h36m_eval_joint,:]
                pose_coord_out_h36m_aligned = rigid_align(pose_coord_out_h36m, pose_coord_gt_h36m)
                eval_result['mpjpe_param'].append(np.sqrt(np.sum((pose_coord_out_h36m - pose_coord_gt_h36m)**2,1)).mean() * 1000) # meter -> milimeter
                eval_result['pa_mpjpe_param'].append(np.sqrt(np.sum((pose_coord_out_h36m_aligned - pose_coord_gt_h36m)**2,1)).mean() * 1000) # meter -> milimeter

            vis = False
            if vis:
                seq_name = annot['img_path'].split('/')[-2]
                img_name = annot['img_path'].split('/')[-1][:-4]
                filename = seq_name + '_' + img_name + '_' + str(n)

                img = load_img(annot['img_path'])[:,:,::-1]
                img = vis_mesh(img, mesh_out_img, 0.5)
                cv2.imwrite(filename + '.jpg', img)

                save_obj(mesh_out_cam, self.smpl.face, filename + '.obj')
                
        return eval_result

    def print_eval_result(self, eval_result):
        print('MPJPE from lixel mesh: %.2f mm' % np.mean(eval_result['mpjpe_lixel']))
        print('PA MPJPE from lixel mesh: %.2f mm' % np.mean(eval_result['pa_mpjpe_lixel']))

        print('MPJPE from param mesh: %.2f mm' % np.mean(eval_result['mpjpe_param']))
        print('PA MPJPE from param mesh: %.2f mm' % np.mean(eval_result['pa_mpjpe_param']))



