import torch
from plyfile import PlyData, PlyElement
from scene import Scene, GaussianModel
from argparse import ArgumentParser, Namespace
from arguments import ModelParams, PipelineParams, OptimizationParams
import sys

import ipdb


def main():
    # Set up command line argument parser
    parser = ArgumentParser(description="Training script parameters")
    lp = ModelParams(parser)
    op = OptimizationParams(parser)
    pp = PipelineParams(parser)
    parser.add_argument("--config", type=str, default="config.yaml",
                        help="Path to YAML configuration file")
    args = parser.parse_args(sys.argv[1:])
    lp.from_yaml(args.config)
    op.from_yaml(args.config)
    for thr_ratio in [1e-4, 1e-3, 1e-2]:
        path = "./remove_gs/point_cloud.ply"
        path = "/data/3dgs/test2_15_out/iteration_1000000/point_cloud.ply"
        dataset = lp.extract(args)
        gaussians = GaussianModel(dataset.sh_degree)
        gaussians.load_ply(path)
        ratio = torch.max(gaussians.get_scaling, axis=1)[
            0] / torch.min(gaussians.get_scaling, axis=1)[0]
        length = gaussians._xyz.shape[0]
        thr = ratio.sort()[0][int(length*(1-thr_ratio))]
        mask = ratio < thr
        print("Mask size:", mask.sum())
        print("Original count:", length)
        # ipdb.set_trace()

        gaussians._xyz = gaussians._xyz[mask]
        gaussians._features_dc = gaussians._features_dc[mask]
        gaussians._features_rest = gaussians._features_rest[mask]
        gaussians._opacity = gaussians._opacity[mask]
        gaussians._scaling = gaussians._scaling[mask]
        gaussians._rotation = gaussians._rotation[mask]

        gaussians.save_ply(f"/data/3dgs/test2_15_out/point_cloud_{thr_ratio}.ply")


if __name__ == "__main__":
    main()