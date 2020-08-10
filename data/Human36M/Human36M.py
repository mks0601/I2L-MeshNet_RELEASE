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
from utils.transforms import world2cam, cam2pixel, pixel2cam, rigid_align, transform_joint_to_other_db
from utils.vis import vis_keypoints, vis_mesh, save_obj

class Human36M(torch.utils.data.Dataset):
    def __init__(self, transform, data_split):
        self.transform = transform
        self.data_split = data_split
        self.img_dir = osp.join('..', 'data', 'Human36M', 'images')
        self.annot_path = osp.join('..', 'data', 'Human36M', 'annotations')
        self.human_bbox_root_dir = osp.join('..', 'data', 'Human36M', 'rootnet_output', 'bbox_root_human36m_output.json')
        self.action_name = ['Directions', 'Discussion', 'Eating', 'Greeting', 'Phoning', 'Posing', 'Purchases', 'Sitting', 'SittingDown', 'Smoking', 'Photo', 'Waiting', 'Walking', 'WalkDog', 'WalkTogether']
        self.fitting_thr = 25 # milimeter

        # H36M joint set
        self.h36m_joint_num = 17
        self.h36m_joints_name = ('Pelvis', 'R_Hip', 'R_Knee', 'R_Ankle', 'L_Hip', 'L_Knee', 'L_Ankle', 'Torso', 'Neck', 'Nose', 'Head_top', 'L_Shoulder', 'L_Elbow', 'L_Wrist', 'R_Shoulder', 'R_Elbow', 'R_Wrist')
        self.h36m_flip_pairs = ( (1, 4), (2, 5), (3, 6), (14, 11), (15, 12), (16, 13) )
        self.h36m_skeleton = ( (0, 7), (7, 8), (8, 9), (9, 10), (8, 11), (11, 12), (12, 13), (8, 14), (14, 15), (15, 16), (0, 1), (1, 2), (2, 3), (0, 4), (4, 5), (5, 6) )
        self.h36m_root_joint_idx = self.h36m_joints_name.index('Pelvis')
        self.h36m_eval_joint = (1, 2, 3, 4, 5, 6, 8, 10, 11, 12, 13, 14, 15, 16)
        self.h36m_joint_regressor = np.load(osp.join('..', 'data', 'Human36M', 'J_regressor_h36m_correct.npy'))

        # SMPL joint set
        self.smpl = SMPL()
        self.face = self.smpl.face
        self.joint_regressor = self.smpl.joint_regressor
        self.vertex_num = self.smpl.vertex_num
        self.joint_num = self.smpl.joint_num
        self.joints_name = self.smpl.joints_name
        self.flip_pairs = self.smpl.flip_pairs
        self.skeleton = self.smpl.skeleton
        self.root_joint_idx = self.smpl.root_joint_idx
        self.face_kps_vertex = self.smpl.face_kps_vertex

        self.datalist = self.load_data()
        
    def get_subsampling_ratio(self):
        if self.data_split == 'train':
            return 5
        elif self.data_split == 'test':
            return 64
        else:
            assert 0, print('Unknown subset')

    def get_subject(self):
        if self.data_split == 'train':
            subject = [1,5,6,7,8]
        elif self.data_split == 'test':
            subject = [9,11]
        else:
            assert 0, print("Unknown subset")

        return subject
    
    def load_data(self):
        subject_list = self.get_subject()
        sampling_ratio = self.get_subsampling_ratio()
        
        # aggregate annotations from each subject
        db = COCO()
        cameras = {}
        joints = {}
        smpl_params = {}
        for subject in subject_list:
            # data load
            with open(osp.join(self.annot_path, 'Human36M_subject' + str(subject) + '_data.json'),'r') as f:
                annot = json.load(f)
            if len(db.dataset) == 0:
                for k,v in annot.items():
                    db.dataset[k] = v
            else:
                for k,v in annot.items():
                    db.dataset[k] += v
            # camera load
            with open(osp.join(self.annot_path, 'Human36M_subject' + str(subject) + '_camera.json'),'r') as f:
                cameras[str(subject)] = json.load(f)
            # joint coordinate load
            with open(osp.join(self.annot_path, 'Human36M_subject' + str(subject) + '_joint_3d.json'),'r') as f:
                joints[str(subject)] = json.load(f)
            # smpl parameter load
            with open(osp.join(self.annot_path, 'Human36M_subject' + str(subject) + '_smpl_param.json'),'r') as f:
                smpl_params[str(subject)] = json.load(f)
        db.createIndex()

        if self.data_split == 'test' and not cfg.use_gt_info:
            print("Get bounding box and root from " + self.human_bbox_root_dir)
            bbox_root_result = {}
            with open(self.human_bbox_root_dir) as f:
                annot = json.load(f)
            for i in range(len(annot)):
                bbox_root_result[str(annot[i]['image_id'])] = {'bbox': np.array(annot[i]['bbox']), 'root': np.array(annot[i]['root_cam'])}
        else:
            print("Get bounding box and root from groundtruth")

        datalist = []
        for aid in db.anns.keys():
            ann = db.anns[aid]
            image_id = ann['image_id']
            img = db.loadImgs(image_id)[0]
            img_path = osp.join(self.img_dir, img['file_name'])
            img_shape = (img['height'], img['width'])
            
            # check subject and frame_idx
            frame_idx = img['frame_idx'];
            if frame_idx % sampling_ratio != 0:
                continue

            # check smpl parameter exist
            subject = img['subject']; action_idx = img['action_idx']; subaction_idx = img['subaction_idx']; frame_idx = img['frame_idx'];
            try:
                smpl_param = smpl_params[str(subject)][str(action_idx)][str(subaction_idx)][str(frame_idx)]
            except KeyError:
                smpl_param = None

            # camera parameter
            cam_idx = img['cam_idx']
            cam_param = cameras[str(subject)][str(cam_idx)]
            R,t,f,c = np.array(cam_param['R'], dtype=np.float32), np.array(cam_param['t'], dtype=np.float32), np.array(cam_param['f'], dtype=np.float32), np.array(cam_param['c'], dtype=np.float32)
            cam_param = {'R': R, 't': t, 'focal': f, 'princpt': c}
            
            # only use frontal camera following previous works (HMR and SPIN)
            if self.data_split == 'test' and str(cam_idx) != '4':
                continue
                
            # project world coordinate to cam, image coordinate space
            joint_world = np.array(joints[str(subject)][str(action_idx)][str(subaction_idx)][str(frame_idx)], dtype=np.float32)
            joint_cam = world2cam(joint_world, R, t)
            joint_img = cam2pixel(joint_cam, f, c)
            joint_valid = np.ones((self.h36m_joint_num,1))
        
            if self.data_split == 'test' and not cfg.use_gt_info:
                bbox = bbox_root_result[str(image_id)]['bbox'] # bbox should be aspect ratio preserved-extended. It is done in RootNet.
                root_joint_depth = bbox_root_result[str(image_id)]['root'][2]
            else:
                bbox = process_bbox(np.array(ann['bbox']), img['width'], img['height'])
                if bbox is None: continue
                root_joint_depth = joint_cam[self.h36m_root_joint_idx][2]
    
            datalist.append({
                'img_path': img_path,
                'img_id': image_id,
                'img_shape': img_shape,
                'bbox': bbox,
                'joint_img': joint_img,
                'joint_cam': joint_cam,
                'joint_valid': joint_valid,
                'smpl_param': smpl_param,
                'root_joint_depth': root_joint_depth,
                'cam_param': cam_param})
            
        return datalist

    def get_smpl_coord(self, smpl_param, cam_param, do_flip, img_shape):
        pose, shape, trans = smpl_param['pose'], smpl_param['shape'], smpl_param['trans']
        smpl_pose = torch.FloatTensor(pose).view(-1,3); smpl_shape = torch.FloatTensor(shape).view(1,-1); # smpl parameters (pose: 72 dimension, shape: 10 dimension)
        R, t = np.array(cam_param['R'], dtype=np.float32).reshape(3,3), np.array(cam_param['t'], dtype=np.float32).reshape(3) # camera rotation and translation
        
        # merge root pose and camera rotation 
        root_pose = smpl_pose[self.root_joint_idx,:].numpy()
        root_pose, _ = cv2.Rodrigues(root_pose)
        root_pose, _ = cv2.Rodrigues(np.dot(R,root_pose))
        smpl_pose[self.root_joint_idx] = torch.from_numpy(root_pose).view(3)

        # flip smpl pose parameter (axis-angle)
        if do_flip:
            for pair in self.flip_pairs:
                if pair[0] < len(smpl_pose) and pair[1] < len(smpl_pose): # face keypoints are already included in self.flip_pairs. However, they are not included in smpl_pose.
                    smpl_pose[pair[0], :], smpl_pose[pair[1], :] = smpl_pose[pair[1], :].clone(), smpl_pose[pair[0], :].clone()
            smpl_pose[:,1:3] *= -1; # multiply -1 to y and z axis of axis-angle
        smpl_pose = smpl_pose.view(1,-1)
       
        # get mesh and joint coordinates
        smpl_mesh_coord, smpl_joint_coord = self.smpl.layer['neutral'](smpl_pose, smpl_shape)

        # incorporate face keypoints
        smpl_mesh_coord = smpl_mesh_coord.numpy().astype(np.float32).reshape(-1,3); 
        smpl_joint_coord = smpl_joint_coord.numpy().astype(np.float32).reshape(-1,3)
        smpl_face_kps_coord = smpl_mesh_coord[self.face_kps_vertex,:].reshape(-1,3)
        smpl_joint_coord = np.concatenate((smpl_joint_coord, smpl_face_kps_coord))
 
        # compenstate rotation (translation from origin to root joint was not cancled)
        smpl_trans = np.array(trans, dtype=np.float32).reshape(3) # translation vector from smpl coordinate to h36m world coordinate
        smpl_trans = np.dot(R, smpl_trans[:,None]).reshape(1,3) + t.reshape(1,3)/1000
        root_joint_coord = smpl_joint_coord[self.root_joint_idx].reshape(1,3)
        smpl_trans = smpl_trans - root_joint_coord + np.dot(R, root_joint_coord.transpose(1,0)).transpose(1,0)
        smpl_mesh_coord = smpl_mesh_coord + smpl_trans
        smpl_joint_coord = smpl_joint_coord + smpl_trans

        # flip translation
        if do_flip: # avg of old and new root joint should be image center.
            focal, princpt = cam_param['focal'], cam_param['princpt']
            flip_trans_x = 2 * (((img_shape[1] - 1)/2. - princpt[0]) / focal[0] * (smpl_joint_coord[self.root_joint_idx,2] * 1000)) / 1000 - 2 * smpl_joint_coord[self.root_joint_idx][0]
            smpl_mesh_coord[:,0] += flip_trans_x
            smpl_joint_coord[:,0] += flip_trans_x

        # change to mean shape if beta is too far from it
        smpl_shape[(smpl_shape.abs() > 3).any(dim=1)] = 0.

        # meter -> milimeter
        smpl_mesh_coord *= 1000; smpl_joint_coord *= 1000;
        return smpl_mesh_coord, smpl_joint_coord, smpl_pose[0].numpy(), smpl_shape[0].numpy()
    
    def get_fitting_error(self, h36m_joint, smpl_mesh, do_flip):
        h36m_joint = h36m_joint - h36m_joint[self.h36m_root_joint_idx,None,:] # root-relative
        if do_flip:
            h36m_joint[:,0] = -h36m_joint[:,0]
            for pair in self.h36m_flip_pairs:
                h36m_joint[pair[0],:] , h36m_joint[pair[1],:] = h36m_joint[pair[1],:].copy(), h36m_joint[pair[0],:].copy()

        h36m_from_smpl = np.dot(self.h36m_joint_regressor, smpl_mesh)
        h36m_from_smpl = h36m_from_smpl - np.mean(h36m_from_smpl,0)[None,:] + np.mean(h36m_joint,0)[None,:] # translation alignment

        error = np.sqrt(np.sum((h36m_joint - h36m_from_smpl)**2,1)).mean()
        return error

    def __len__(self):
        return len(self.datalist)

    def __getitem__(self, idx):
        data = copy.deepcopy(self.datalist[idx])
        img_path, img_shape, bbox, smpl_param, cam_param = data['img_path'], data['img_shape'], data['bbox'], data['smpl_param'], data['cam_param']
         
        # img
        img = load_img(img_path)
        img, img2bb_trans, bb2img_trans, rot, do_flip = augmentation(img, bbox, self.data_split)
        img = self.transform(img.astype(np.float32))/255.
        
        if self.data_split == 'train':
            # h36m gt
            h36m_joint_img = data['joint_img']
            h36m_joint_cam = data['joint_cam']
            h36m_joint_cam = h36m_joint_cam - h36m_joint_cam[self.h36m_root_joint_idx,None,:] # root-relative
            h36m_joint_valid = data['joint_valid']
            if do_flip:
                h36m_joint_cam[:,0] = -h36m_joint_cam[:,0]
                h36m_joint_img[:,0] = img_shape[1] - 1 - h36m_joint_img[:,0]
                for pair in self.h36m_flip_pairs:
                    h36m_joint_img[pair[0],:], h36m_joint_img[pair[1],:] = h36m_joint_img[pair[1],:].copy(), h36m_joint_img[pair[0],:].copy()
                    h36m_joint_cam[pair[0],:], h36m_joint_cam[pair[1],:] = h36m_joint_cam[pair[1],:].copy(), h36m_joint_cam[pair[0],:].copy()
                    h36m_joint_valid[pair[0],:], h36m_joint_valid[pair[1],:] = h36m_joint_valid[pair[1],:].copy(), h36m_joint_valid[pair[0],:].copy()

            h36m_joint_img_xy1 = np.concatenate((h36m_joint_img[:,:2], np.ones_like(h36m_joint_img[:,:1])),1)
            h36m_joint_img[:,:2] = np.dot(img2bb_trans, h36m_joint_img_xy1.transpose(1,0)).transpose(1,0)
            h36m_joint_img[:,0] = h36m_joint_img[:,0] / cfg.input_img_shape[1] * cfg.output_hm_shape[2]
            h36m_joint_img[:,1] = h36m_joint_img[:,1] / cfg.input_img_shape[0] * cfg.output_hm_shape[1]
            h36m_joint_img[:,2] = h36m_joint_img[:,2] - h36m_joint_img[self.h36m_root_joint_idx][2] # root-relative
            h36m_joint_img[:,2] = (h36m_joint_img[:,2] / (cfg.bbox_3d_size * 1000  / 2) + 1)/2. * cfg.output_hm_shape[0] # change cfg.bbox_3d_size from meter to milimeter
            
            # check truncation
            h36m_joint_trunc = h36m_joint_valid * ((h36m_joint_img[:,0] >= 0) * (h36m_joint_img[:,0] < cfg.output_hm_shape[2]) * \
                        (h36m_joint_img[:,1] >= 0) * (h36m_joint_img[:,1] < cfg.output_hm_shape[1]) * \
                        (h36m_joint_img[:,2] >= 0) * (h36m_joint_img[:,2] < cfg.output_hm_shape[0])).reshape(-1,1).astype(np.float32)

            # transform h36m joints to target db joints
            h36m_joint_img = transform_joint_to_other_db(h36m_joint_img, self.h36m_joints_name, self.joints_name)
            h36m_joint_cam = transform_joint_to_other_db(h36m_joint_cam, self.h36m_joints_name, self.joints_name)
            h36m_joint_valid = transform_joint_to_other_db(h36m_joint_valid, self.h36m_joints_name, self.joints_name)
            h36m_joint_trunc = transform_joint_to_other_db(h36m_joint_trunc, self.h36m_joints_name, self.joints_name)

            if smpl_param is not None:
                # smpl coordinates
                smpl_mesh_cam, smpl_joint_cam, smpl_pose, smpl_shape = self.get_smpl_coord(smpl_param, cam_param, do_flip, img_shape)
                smpl_coord_cam = np.concatenate((smpl_mesh_cam, smpl_joint_cam))
                focal, princpt = cam_param['focal'], cam_param['princpt']
                smpl_coord_img = cam2pixel(smpl_coord_cam, focal, princpt)

                # affine transform x,y coordinates, root-relative depth
                smpl_coord_img_xy1 = np.concatenate((smpl_coord_img[:,:2], np.ones_like(smpl_coord_img[:,:1])),1)
                smpl_coord_img[:,:2] = np.dot(img2bb_trans, smpl_coord_img_xy1.transpose(1,0)).transpose(1,0)[:,:2]
                smpl_coord_img[:,2] = smpl_coord_img[:,2] - smpl_coord_cam[self.vertex_num + self.root_joint_idx][2]
                # coordinates voxelize
                smpl_coord_img[:,0] = smpl_coord_img[:,0] / cfg.input_img_shape[1] * cfg.output_hm_shape[2]
                smpl_coord_img[:,1] = smpl_coord_img[:,1] / cfg.input_img_shape[0] * cfg.output_hm_shape[1]
                smpl_coord_img[:,2] = (smpl_coord_img[:,2] / (cfg.bbox_3d_size * 1000  / 2) + 1)/2. * cfg.output_hm_shape[0] # change cfg.bbox_3d_size from meter to milimeter

                # check truncation
                smpl_trunc = ((smpl_coord_img[:,0] >= 0) * (smpl_coord_img[:,0] < cfg.output_hm_shape[2]) * \
                            (smpl_coord_img[:,1] >= 0) * (smpl_coord_img[:,1] < cfg.output_hm_shape[1]) * \
                            (smpl_coord_img[:,2] >= 0) * (smpl_coord_img[:,2] < cfg.output_hm_shape[0])).reshape(-1,1).astype(np.float32)
                
                # split mesh and joint coordinates
                smpl_mesh_img = smpl_coord_img[:self.vertex_num]; smpl_joint_img = smpl_coord_img[self.vertex_num:];
                smpl_mesh_trunc = smpl_trunc[:self.vertex_num]; smpl_joint_trunc = smpl_trunc[self.vertex_num:];

                # if fitted mesh is too far from h36m gt, discard it
                is_valid_fit = True
                error = self.get_fitting_error(data['joint_cam'], smpl_mesh_cam, do_flip)
                if error > self.fitting_thr:
                    is_valid_fit = False

            else:
                smpl_joint_img = np.zeros((self.joint_num,3), dtype=np.float32) # dummy
                smpl_joint_cam = np.zeros((self.joint_num,3), dtype=np.float32) # dummy
                smpl_mesh_img = np.zeros((self.vertex_num,3), dtype=np.float32) # dummy
                smpl_pose = np.zeros((72), dtype=np.float32) # dummy
                smpl_shape = np.zeros((10), dtype=np.float32) # dummy
                smpl_joint_trunc = np.zeros((self.joint_num,1), dtype=np.float32) # dummy
                smpl_mesh_trunc = np.zeros((self.vertex_num,1), dtype=np.float32) # dummy
                is_valid_fit = False
            
            # 3D data rotation augmentation
            rot_aug_mat = np.array([[np.cos(np.deg2rad(-rot)), -np.sin(np.deg2rad(-rot)), 0], 
            [np.sin(np.deg2rad(-rot)), np.cos(np.deg2rad(-rot)), 0],
            [0, 0, 1]], dtype=np.float32)
            # h36m coordinate
            h36m_joint_cam = np.dot(rot_aug_mat, h36m_joint_cam.transpose(1,0)).transpose(1,0) / 1000 # milimeter to meter
            # parameter
            smpl_pose = smpl_pose.reshape(-1,3)
            root_pose = smpl_pose[self.root_joint_idx,:]
            root_pose, _ = cv2.Rodrigues(root_pose)
            root_pose, _ = cv2.Rodrigues(np.dot(rot_aug_mat,root_pose))
            smpl_pose[self.root_joint_idx] = root_pose.reshape(3)
            smpl_pose = smpl_pose.reshape(-1)
            # smpl coordinate
            smpl_joint_cam = smpl_joint_cam - smpl_joint_cam[self.root_joint_idx,None] # root-relative
            smpl_joint_cam = np.dot(rot_aug_mat, smpl_joint_cam.transpose(1,0)).transpose(1,0) / 1000 # milimeter to meter

            inputs = {'img': img}
            targets = {'orig_joint_img': h36m_joint_img, 'fit_joint_img': smpl_joint_img, 'fit_mesh_img': smpl_mesh_img, 'orig_joint_cam': h36m_joint_cam, 'fit_joint_cam': smpl_joint_cam, 'pose_param': smpl_pose, 'shape_param': smpl_shape}
            meta_info = {'orig_joint_valid': h36m_joint_valid, 'orig_joint_trunc': h36m_joint_trunc, 'fit_joint_trunc': smpl_joint_trunc, 'fit_mesh_trunc': smpl_mesh_trunc, 'is_valid_fit': float(is_valid_fit), 'is_3D': float(True)}
            return inputs, targets, meta_info
        else:
            inputs = {'img': img}
            targets = {}
            meta_info = {'bb2img_trans': bb2img_trans}
            return inputs, targets, meta_info

    def evaluate(self, outs, cur_sample_idx):

        annots = self.datalist
        sample_num = len(outs)
        eval_result = {'mpjpe_lixel': [], 'pa_mpjpe_lixel': [], 'mpjpe_param': [], 'pa_mpjpe_param': []}
        for n in range(sample_num):
            annot = annots[cur_sample_idx + n]
            out = outs[n]
            
            # mesh from lixel
            # x,y: resize to input image space and perform bbox to image affine transform
            mesh_out_img = out['mesh_coord_img']
            mesh_out_img[:,0] = mesh_out_img[:,0] / cfg.output_hm_shape[2] * cfg.input_img_shape[1]
            mesh_out_img[:,1] = mesh_out_img[:,1] / cfg.output_hm_shape[1] * cfg.input_img_shape[0]
            mesh_out_img_xy1 = np.concatenate((mesh_out_img[:,:2], np.ones_like(mesh_out_img[:,:1])),1)
            mesh_out_img[:,:2] = np.dot(out['bb2img_trans'], mesh_out_img_xy1.transpose(1,0)).transpose(1,0)[:,:2]
            # z: devoxelize and translate to absolute depth
            root_joint_depth = annot['root_joint_depth']
            mesh_out_img[:,2] = (mesh_out_img[:,2] / cfg.output_hm_shape[0] * 2. - 1) * (cfg.bbox_3d_size * 1000 / 2)
            mesh_out_img[:,2] = mesh_out_img[:,2] + root_joint_depth
            # camera back-projection
            cam_param = annot['cam_param']
            focal, princpt = cam_param['focal'], cam_param['princpt']
            mesh_out_cam = pixel2cam(mesh_out_img, focal, princpt)

            # h36m joint from gt mesh
            pose_coord_gt_h36m = annot['joint_cam'] 
            pose_coord_gt_h36m = pose_coord_gt_h36m - pose_coord_gt_h36m[self.h36m_root_joint_idx,None] # root-relative 
            pose_coord_gt_h36m = pose_coord_gt_h36m[self.h36m_eval_joint,:] 
            
            # h36m joint from lixel mesh
            pose_coord_out_h36m = np.dot(self.h36m_joint_regressor, mesh_out_cam)
            pose_coord_out_h36m = pose_coord_out_h36m - pose_coord_out_h36m[self.h36m_root_joint_idx,None] # root-relative
            pose_coord_out_h36m = pose_coord_out_h36m[self.h36m_eval_joint,:]
            pose_coord_out_h36m_aligned = rigid_align(pose_coord_out_h36m, pose_coord_gt_h36m)
            eval_result['mpjpe_lixel'].append(np.sqrt(np.sum((pose_coord_out_h36m - pose_coord_gt_h36m)**2,1)).mean())
            eval_result['pa_mpjpe_lixel'].append(np.sqrt(np.sum((pose_coord_out_h36m_aligned - pose_coord_gt_h36m)**2,1)).mean())
            
            if cfg.stage == 'param':
                # h36m joint from param mesh
                mesh_out_cam = out['mesh_coord_cam'] * 1000 # meter to milimeter
                pose_coord_out_h36m = np.dot(self.h36m_joint_regressor, mesh_out_cam) # meter to milimeter
                pose_coord_out_h36m = pose_coord_out_h36m - pose_coord_out_h36m[self.h36m_root_joint_idx,None] # root-relative
                pose_coord_out_h36m = pose_coord_out_h36m[self.h36m_eval_joint,:]
                pose_coord_out_h36m_aligned = rigid_align(pose_coord_out_h36m, pose_coord_gt_h36m)
                eval_result['mpjpe_param'].append(np.sqrt(np.sum((pose_coord_out_h36m - pose_coord_gt_h36m)**2,1)).mean())
                eval_result['pa_mpjpe_param'].append(np.sqrt(np.sum((pose_coord_out_h36m_aligned - pose_coord_gt_h36m)**2,1)).mean())

            vis = False
            if vis:
                filename = annot['img_path'].split('/')[-1][:-4]

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
