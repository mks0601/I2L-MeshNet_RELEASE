import argparse

from matplotlib import pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import torch
from tqdm import tqdm

from manopth import argutils
from manopth.manolayer import ManoLayer
from manopth.demo import display_hand

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--batch_size', default=1, type=int)
    parser.add_argument('--cuda', action='store_true')
    parser.add_argument(
        '--no_display',
        action='store_true',
        help="Disable display output of ManoLayer given random inputs")
    parser.add_argument('--side', default='left', choices=['left', 'right'])
    parser.add_argument('--random_shape', action='store_true', help="Random hand shape")
    parser.add_argument('--rand_mag', type=float, default=1, help="Controls pose variability")
    parser.add_argument(
        '--flat_hand_mean',
        action='store_true',
        help="Use flat hand as mean instead of average hand pose")
    parser.add_argument(
        '--iters',
        type=int,
        default=1,
        help=
        "Use for quick profiling of forward and backward pass accross ManoLayer"
    )
    parser.add_argument('--mano_root', default='mano/models')
    parser.add_argument('--root_rot_mode', default='axisang', choices=['rot6d', 'axisang'])
    parser.add_argument('--no_pca', action='store_true', help="Give axis-angle or rotation matrix as inputs instead of PCA coefficients")
    parser.add_argument('--joint_rot_mode', default='axisang', choices=['rotmat', 'axisang'], help="Joint rotation inputs")
    parser.add_argument(
        '--mano_ncomps', default=6, type=int, help="Number of PCA components")
    args = parser.parse_args()

    argutils.print_args(args)

    layer = ManoLayer(
        flat_hand_mean=args.flat_hand_mean,
        side=args.side,
        mano_root=args.mano_root,
        ncomps=args.mano_ncomps,
        use_pca=not args.no_pca,
        root_rot_mode=args.root_rot_mode,
        joint_rot_mode=args.joint_rot_mode)
    if args.root_rot_mode == 'axisang':
        rot = 3
    else:
        rot = 6
    print(rot)
    if args.no_pca:
        args.mano_ncomps = 45

    # Generate random pose coefficients
    pose_params = args.rand_mag * torch.rand(args.batch_size, args.mano_ncomps + rot)
    pose_params.requires_grad = True
    if args.random_shape:
        shape = torch.rand(args.batch_size, 10)
    else:
        shape = torch.zeros(1)  # Hack to act like None for PyTorch JIT
    if args.cuda:
        pose_params = pose_params.cuda()
        shape = shape.cuda()
        layer.cuda()

    # Loop for forward/backward quick profiling
    for idx in tqdm(range(args.iters)):
        # Forward pass
        verts, Jtr = layer(pose_params, th_betas=shape)

        # Backward pass
        loss = torch.norm(verts)
        loss.backward()

    if not args.no_display:
        verts, Jtr = layer(pose_params, th_betas=shape)
        joints = Jtr.cpu().detach()
        verts = verts.cpu().detach()

        # Draw obtained vertices and joints
        display_hand({
            'verts': verts,
            'joints': joints
        },
                     mano_faces=layer.th_faces)
