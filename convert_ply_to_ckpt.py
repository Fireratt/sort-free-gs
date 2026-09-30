import torch
from scene.gaussian_model import GaussianModel
from argparse import ArgumentParser


parser = ArgumentParser()
parser.add_argument("--ply", required=True, help="3DGS/Grendel 输出的 point_cloud.ply 路径")
parser.add_argument("--output", required=True, help="输出的 .pth checkpoint 路径")
parser.add_argument("--sh_degree", type=int, default=3)
parser.add_argument("--depth_correct", action="store_true")
parser.add_argument(
    "--spatial_lr_scale",
    type=float,
    default=1.0,
    help="续训时位置和 sigma 学习率的缩放因子；若不确定可先用 1.0。",
)
args = parser.parse_args()

# 1. 初始化 Sort-Free 高斯模型。
gaussians = GaussianModel(args.sh_degree, args.depth_correct)
gaussians.spatial_lr_scale = args.spatial_lr_scale

# 2. 从原生 3DGS/Grendel PLY 读取基础高斯，并为 Sort-Free 额外参数做默认初始化。
gaussians.load_3dgs_ply(args.ply)

# 3. 构造“初始化型 checkpoint”。
#    这里不保存优化器状态，让后续 train.py 在 restore() 里使用本次训练命令的优化参数重新建优化器。
num_points = gaussians.get_xyz.shape[0]
xyz_gradient_accum = torch.zeros((num_points, 1), device="cuda")
denom = torch.zeros((num_points, 1), device="cuda")

model_params = (
    gaussians.active_sh_degree,
    gaussians._xyz,
    gaussians._features_dc,
    gaussians._features_rest,
    gaussians._scaling,
    gaussians._rotation,
    gaussians._opacity,
    gaussians._features_do,
    gaussians._features_ro,
    gaussians.max_radii2D,
    xyz_gradient_accum,
    denom,
    None,
    gaussians.spatial_lr_scale,
)
checkpoint_data = (model_params, 0)

torch.save(checkpoint_data, args.output)
print(f"转换完成: {args.output}")
print(f"高斯数量: {gaussians.get_xyz.shape[0]}")
print(f"SH degree: {args.sh_degree}, depth_correct: {args.depth_correct}")
print(f"spatial_lr_scale: {args.spatial_lr_scale}")
