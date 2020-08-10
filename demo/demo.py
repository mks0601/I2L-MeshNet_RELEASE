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
from utils.preprocessing import process_bbox, generate_patch_image
from utils.vis import vis_mesh, save_obj

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
        gpus[0] = 0 if not gpus[0].isdigit() else int(gpus[0])
        gpus[1] = len(mem_info()) if not gpus[1].isdigit() else int(gpus[1]) + 1
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

# snapshot load
model_path = './snapshot_%d.pth.tar' % int(args.test_epoch)
assert osp.exists(model_path), 'Cannot find model at ' + model_path
print('Load checkpoint from {}'.format(model_path))
model = get_model(vertex_num, joint_num, 'test')

model = DataParallel(model).cuda()
ckpt = torch.load(model_path)
model.load_state_dict(ckpt['network'])
model.eval()

# prepare input image
transform = transforms.ToTensor()
img_path = 'input.jpg'
img = cv2.imread(img_path)

# prepare bbox
bbox = [164, 93, 222, 252] # xmin, ymin, width, height
bbox = process_bbox(bbox, img.shape[1], img.shape[0])
assert len(bbox) == 4, 'Please set bbox'
img, img2bb_trans, bb2img_trans = generate_patch_image(img, bbox, 1.0, 0.0, False, cfg.input_img_shape) 
img = transform(img.astype(np.float32))/255
img = img.cuda()[None,:,:,:]

# forward
inputs = {'img': img}
targets = {}
meta_info = {'bb2img_trans': bb2img_trans}
with torch.no_grad():
    out = model(inputs, targets, meta_info, 'test')

img = img[0].cpu().numpy()
mesh_img = out['mesh_coord_img'][0].cpu().numpy()
mesh_cam = out['mesh_coord_cam'][0].cpu().numpy()

# visualize mesh in 2D space
vis_img = img.copy() * 255
vis_img = vis_img.astype(np.uint8)
vis_img = np.transpose(vis_img,(1,2,0)).copy()
mesh_img[:,0] = mesh_img[:,0] / cfg.output_hm_shape[2] * cfg.input_img_shape[1]
mesh_img[:,1] = mesh_img[:,1] / cfg.output_hm_shape[1] * cfg.input_img_shape[0]
vis_img = vis_mesh(vis_img, mesh_img)
cv2.imwrite('output_mesh.jpg', vis_img)

# save mesh (obj)
save_obj(mesh_cam, face, 'output_mesh.obj')
