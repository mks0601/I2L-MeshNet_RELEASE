import torch
import torch.nn as nn
from torch.nn import functional as F
from nets.resnet import ResNetBackbone
from nets.module import PoseNet, Pose2Feat, MeshNet, ParamRegressor
from nets.loss import CoordLoss, ParamLoss, NormalVectorLoss, EdgeLengthLoss
from utils.smpl import SMPL
from utils.mano import MANO
from config import cfg
from contextlib import nullcontext
import math

class Model(nn.Module):
    def __init__(self, pose_backbone, pose_net, pose2feat, mesh_backbone, mesh_net, param_regressor):
        super(Model, self).__init__()
        self.pose_backbone = pose_backbone
        self.pose_net = pose_net
        self.pose2feat = pose2feat
        self.mesh_backbone = mesh_backbone
        self.mesh_net = mesh_net
        self.param_regressor = param_regressor
        
        if 'FreiHAND' in cfg.trainset_3d + cfg.trainset_2d + [cfg.testset]:
            self.human_model = MANO()
            self.human_model_layer = self.human_model.layer.cuda()
        else:
            self.human_model = SMPL()
            self.human_model_layer = self.human_model.layer['neutral'].cuda()
        self.root_joint_idx = self.human_model.root_joint_idx
        self.mesh_face = self.human_model.face
        self.joint_regressor = self.human_model.joint_regressor
        if cfg.stage == 'lixel':
            self.trainable_modules = [self.pose_backbone, self.pose_net, self.pose2feat, self.mesh_backbone, self.mesh_net]
        else:
            self.trainable_modules = [self.param_regressor]

        self.coord_loss = CoordLoss()
        self.param_loss = ParamLoss()
        self.normal_loss = NormalVectorLoss(self.mesh_face)
        self.edge_loss = EdgeLengthLoss(self.mesh_face)
   
    def make_gaussian_heatmap(self, joint_coord_img):
        x = torch.arange(cfg.output_hm_shape[2])
        y = torch.arange(cfg.output_hm_shape[1])
        z = torch.arange(cfg.output_hm_shape[0])
        zz,yy,xx = torch.meshgrid(z,y,x)
        xx = xx[None,None,:,:,:].cuda().float(); yy = yy[None,None,:,:,:].cuda().float(); zz = zz[None,None,:,:,:].cuda().float();
        
        x = joint_coord_img[:,:,0,None,None,None]; y = joint_coord_img[:,:,1,None,None,None]; z = joint_coord_img[:,:,2,None,None,None];
        heatmap = torch.exp(-(((xx-x)/cfg.sigma)**2)/2 -(((yy-y)/cfg.sigma)**2)/2 - (((zz-z)/cfg.sigma)**2)/2)
        return heatmap
    
    def forward(self, inputs, targets, meta_info, mode):

        if cfg.stage == 'lixel':
            cm = nullcontext()
        else:
            cm = torch.no_grad()
        
        with cm:
            # posenet forward
            shared_img_feat, pose_img_feat = self.pose_backbone(inputs['img'])
            joint_coord_img = self.pose_net(pose_img_feat)
            
            # make 3D heatmap from posenet output and convert to image feature
            with torch.no_grad():
                joint_heatmap = self.make_gaussian_heatmap(joint_coord_img.detach())
            shared_img_feat = self.pose2feat(shared_img_feat, joint_heatmap)

            # meshnet forward
            _, mesh_img_feat = self.mesh_backbone(shared_img_feat, skip_early=True)
            mesh_coord_img = self.mesh_net(mesh_img_feat)
            
            # joint coordinate outputs from mesh coordinates
            joint_img_from_mesh = torch.bmm(torch.from_numpy(self.joint_regressor).cuda()[None,:,:].repeat(mesh_coord_img.shape[0],1,1), mesh_coord_img)
            mesh_coord_cam = None

        if cfg.stage == 'param':
            # parameter regression
            pose_param, shape_param = self.param_regressor(joint_img_from_mesh.detach())

            # get mesh and joint coordinates
            mesh_coord_cam, _ = self.human_model_layer(pose_param, shape_param)
            joint_coord_cam = torch.bmm(torch.from_numpy(self.joint_regressor).cuda()[None,:,:].repeat(mesh_coord_cam.shape[0],1,1), mesh_coord_cam)

            # root-relative 3D coordinates
            root_joint_cam = joint_coord_cam[:,self.root_joint_idx,None,:]
            mesh_coord_cam = mesh_coord_cam - root_joint_cam
            joint_coord_cam = joint_coord_cam - root_joint_cam
       
        if mode == 'train':
            # loss functions
            loss = {}
            if cfg.stage == 'lixel':
                loss['joint_fit'] = self.coord_loss(joint_coord_img, targets['fit_joint_img'], meta_info['fit_joint_trunc'] * meta_info['is_valid_fit'][:,None,None])
                loss['joint_orig'] = self.coord_loss(joint_coord_img, targets['orig_joint_img'], meta_info['orig_joint_trunc'], meta_info['is_3D'])
                loss['mesh_fit'] = self.coord_loss(mesh_coord_img, targets['fit_mesh_img'], meta_info['fit_mesh_trunc'] * meta_info['is_valid_fit'][:,None,None])
                loss['mesh_joint_orig'] = self.coord_loss(joint_img_from_mesh, targets['orig_joint_img'], meta_info['orig_joint_trunc'], meta_info['is_3D'])
                loss['mesh_joint_fit'] = self.coord_loss(joint_img_from_mesh, targets['fit_joint_img'], meta_info['fit_joint_trunc'] * meta_info['is_valid_fit'][:,None,None])
                loss['mesh_normal'] = self.normal_loss(mesh_coord_img, targets['fit_mesh_img'], meta_info['fit_mesh_trunc'] * meta_info['is_valid_fit'][:,None,None]) * cfg.normal_loss_weight
                loss['mesh_edge'] = self.edge_loss(mesh_coord_img, targets['fit_mesh_img'], meta_info['fit_mesh_trunc'] * meta_info['is_valid_fit'][:,None,None])
            else:
                loss['pose_param'] = self.param_loss(pose_param, targets['pose_param'], meta_info['is_valid_fit'][:,None])
                loss['shape_param'] = self.param_loss(shape_param, targets['shape_param'], meta_info['is_valid_fit'][:,None])
                loss['joint_orig_cam'] = self.coord_loss(joint_coord_cam, targets['orig_joint_cam'], meta_info['orig_joint_valid'] * meta_info['is_3D'][:,None,None])
                loss['joint_fit_cam'] = self.coord_loss(joint_coord_cam, targets['fit_joint_cam'], meta_info['is_valid_fit'][:,None,None])
            return loss
        else:
            # test output
            out = {}
            out['joint_coord_img'] = joint_coord_img
            out['mesh_coord_img'] = mesh_coord_img
            out['bb2img_trans'] = meta_info['bb2img_trans']

            if cfg.stage == 'param':
                out['mesh_coord_cam'] = mesh_coord_cam
            if 'fit_mesh_coord_cam' in targets:
                out['mesh_coord_cam_target'] = targets['fit_mesh_coord_cam']
            return out

def init_weights(m):
    if type(m) == nn.ConvTranspose2d:
        nn.init.normal_(m.weight,std=0.001)
    elif type(m) == nn.Conv2d:
        nn.init.normal_(m.weight,std=0.001)
        nn.init.constant_(m.bias, 0)
    elif type(m) == nn.BatchNorm2d:
        nn.init.constant_(m.weight,1)
        nn.init.constant_(m.bias,0)
    elif type(m) == nn.Linear:
        nn.init.normal_(m.weight,std=0.01)
        nn.init.constant_(m.bias,0)

def get_model(vertex_num, joint_num, mode):
    pose_backbone = ResNetBackbone(cfg.resnet_type)
    pose_net = PoseNet(joint_num)
    pose2feat = Pose2Feat(joint_num)
    mesh_backbone = ResNetBackbone(cfg.resnet_type)
    mesh_net = MeshNet(vertex_num)
    param_regressor = ParamRegressor(joint_num)

    if mode == 'train':
        pose_backbone.init_weights()
        pose_net.apply(init_weights)
        pose2feat.apply(init_weights)
        mesh_backbone.init_weights()
        mesh_net.apply(init_weights)
        param_regressor.apply(init_weights)
   
    model = Model(pose_backbone, pose_net, pose2feat, mesh_backbone, mesh_net, param_regressor)
    return model

