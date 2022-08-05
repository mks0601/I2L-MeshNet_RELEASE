import sys
import os
import os.path as osp
import argparse
import numpy as np
import cv2
import torch
import torchvision.transforms as transforms
from torch.nn.parallel.data_parallel import DataParallel
import torch.backends.cudnn as cudnn

sys.path.insert(0, osp.join('..', 'main'))
sys.path.insert(0, osp.join('..', 'data'))
sys.path.insert(0, osp.join('..', 'common'))
from config import cfg
from model import get_model
from utils.preprocessing import load_img, process_bbox, generate_patch_image
from utils.transforms import pixel2cam, cam2pixel
from utils.vis import vis_mesh, save_obj, render_mesh

sys.path.insert(0, cfg.smpl_path)
from smplpytorch.pytorch.smpl_layer import SMPL_Layer

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--gpu', type=str, dest='gpu_ids')
    parser.add_argument('--stage', type=str, dest='stage')
    parser.add_argument('--test_epoch', type=str, dest='test_epoch')
    args = parser.parse_args()

    # test gpus
    if not args.gpu_ids:
        assert 0, print("Please set proper gpu ids")

    if '-' in args.gpu_ids:
        gpus = args.gpu_ids.split('-')
        gpus[0] = int(gpus[0])
        gpus[1] = int(gpus[1]) + 1
        args.gpu_ids = ','.join(map(lambda x: str(x), list(range(*gpus))))
    
    if not args.stage:
        assert 0, "Please set training stage among [lixel, param]"

    assert args.test_epoch, 'Test epoch is required.'
    return args

# argument parsing
args = parse_args()
cfg.set_args(args.gpu_ids, args.stage)
cudnn.benchmark = True

# SMPL joint set
joint_num = 29 # original: 24. manually add nose, L/R eye, L/R ear
joints_name = ('Pelvis', 'L_Hip', 'R_Hip', 'Torso', 'L_Knee', 'R_Knee', 'Spine', 'L_Ankle', 'R_Ankle', 'Chest', 'L_Toe', 'R_Toe', 'Neck', 'L_Thorax', 'R_Thorax', 'Head', 'L_Shoulder', 'R_Shoulder', 'L_Elbow', 'R_Elbow', 'L_Wrist', 'R_Wrist', 'L_Hand', 'R_Hand', 'Nose', 'L_Eye', 'R_Eye', 'L_Ear', 'R_Ear')
flip_pairs = ( (1,2), (4,5), (7,8), (10,11), (13,14), (16,17), (18,19), (20,21), (22,23) , (25,26), (27,28) )
skeleton = ( (0,1), (1,4), (4,7), (7,10), (0,2), (2,5), (5,8), (8,11), (0,3), (3,6), (6,9), (9,14), (14,17), (17,19), (19, 21), (21,23), (9,13), (13,16), (16,18), (18,20), (20,22), (9,12), (12,24), (24,15), (24,25), (24,26), (25,27), (26,28) )

# SMPl mesh
vertex_num = 6890
smpl_layer = SMPL_Layer(gender='neutral', model_root=cfg.smpl_path + '/smplpytorch/native/models')
face = smpl_layer.th_faces.numpy()
joint_regressor = smpl_layer.th_J_regressor.numpy()
root_joint_idx = 0

# snapshot load
model_path = './snapshot_%d.pth.tar' % int(args.test_epoch)
assert osp.exists(model_path), 'Cannot find model at ' + model_path
print('Load checkpoint from {}'.format(model_path))
model = get_model(vertex_num, joint_num, 'test')

model = DataParallel(model).cuda()
ckpt = torch.load(model_path)
model.load_state_dict(ckpt['network'], strict=False)
model.eval()

# prepare input image
transform = transforms.ToTensor()
img_path = 'input.jpg'
original_img = load_img(img_path)
original_img_height, original_img_width = original_img.shape[:2]

# prepare bbox
bbox = [139.41, 102.25, 222.39, 241.57] # xmin, ymin, width, height
bbox = process_bbox(bbox, original_img_width, original_img_height)
img, img2bb_trans, bb2img_trans = generate_patch_image(original_img, bbox, 1.0, 0.0, False, cfg.input_img_shape) 
img = transform(img.astype(np.float32))/255
img = img.cuda()[None,:,:,:]

# forward
inputs = {'img': img}
targets = {}
meta_info = {'bb2img_trans': bb2img_trans}
with torch.no_grad():
    out = model(inputs, targets, meta_info, 'test')
img = img[0].cpu().numpy().transpose(1,2,0) # cfg.input_img_shape[1], cfg.input_img_shape[0], 3
mesh_lixel_img = out['mesh_coord_img'][0].cpu().numpy()
mesh_param_cam = out['mesh_coord_cam'][0].cpu().numpy()

# restore mesh_lixel_img to original image space and continuous depth space
mesh_lixel_img[:,0] = mesh_lixel_img[:,0] / cfg.output_hm_shape[2] * cfg.input_img_shape[1]
mesh_lixel_img[:,1] = mesh_lixel_img[:,1] / cfg.output_hm_shape[1] * cfg.input_img_shape[0]
mesh_lixel_img[:,:2] = np.dot(bb2img_trans, np.concatenate((mesh_lixel_img[:,:2], np.ones_like(mesh_lixel_img[:,:1])),1).transpose(1,0)).transpose(1,0)
mesh_lixel_img[:,2] = (mesh_lixel_img[:,2] / cfg.output_hm_shape[0] * 2. - 1) * (cfg.bbox_3d_size / 2)

# root-relative 3D coordinates -> absolute 3D coordinates
focal = (1500,1500);  princpt = (original_img_width/2, original_img_height/2);
root_xy = np.dot(joint_regressor, mesh_lixel_img)[root_joint_idx,:2]
root_depth = 11250.5732421875 # obtain this from RootNet (https://github.com/mks0601/3DMPPE_ROOTNET_RELEASE/tree/master/demo)
root_depth /= 1000 # output of RootNet is milimeter. change it to meter
root_img = np.array([root_xy[0], root_xy[1], root_depth])
root_cam = pixel2cam(root_img[None,:], focal, princpt)
mesh_lixel_img[:,2] += root_depth
mesh_lixel_cam = pixel2cam(mesh_lixel_img, focal, princpt)
mesh_param_cam += root_cam.reshape(1,3)

# visualize lixel mesh in 2D space
vis_img = original_img.copy()
vis_img = vis_mesh(vis_img, mesh_lixel_img)
cv2.imwrite('output_mesh_lixel.jpg', vis_img)

# visualize lixel mesh in 2D space
vis_img = original_img.copy()
mesh_param_img = cam2pixel(mesh_param_cam, focal, princpt)
vis_img = vis_mesh(vis_img, mesh_param_img)
cv2.imwrite('output_mesh_param.jpg', vis_img)

# save mesh (obj)
save_obj(mesh_lixel_cam, face, 'output_mesh_lixel.obj')
save_obj(mesh_param_cam, face, 'output_mesh_param.obj')

# render mesh from lixel
vis_img = original_img.copy()
rendered_img = render_mesh(vis_img, mesh_lixel_cam, face, {'focal': focal, 'princpt': princpt})
cv2.imwrite('rendered_mesh_lixel.jpg', rendered_img)

# render mesh from param
vis_img = original_img.copy()
rendered_img = render_mesh(vis_img, mesh_param_cam, face, {'focal': focal, 'princpt': princpt})
cv2.imwrite('rendered_mesh_param.jpg', rendered_img)

