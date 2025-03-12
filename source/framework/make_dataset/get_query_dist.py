import argparse
import sys

import trimesh
import numpy as np

import os.path
from source.base import fs
from source.base import sdf


def _get_and_save_query_pts(
        file_in_mesh: str, file_out_query_pts: str, file_out_query_dist: str, file_out_query_vis: str,
        num_query_pts: int, patch_radius: float,
        far_query_pts_ratio=0.1, signed_distance_batch_size=1000, debug=False):

    import trimesh

    # random state for file name
    rng = np.random.RandomState(fs.filename_to_hash(file_in_mesh))

    in_mesh = trimesh.load(file_in_mesh)

    # get query pts
    query_pts_ms = sdf.get_query_pts_for_mesh(
        in_mesh, num_query_pts, patch_radius, far_query_pts_ratio, rng)
    np.save(file_out_query_pts, query_pts_ms.astype(np.float32))

    # get signed distance
    query_dist_ms = sdf.get_signed_distance(
        in_mesh, query_pts_ms, signed_distance_batch_size)
    # fix NaNs, Infs and truncate
    nan_ids = np.isnan(query_dist_ms)
    inf_ids = np.isinf(query_dist_ms)
    query_dist_ms[nan_ids] = 0.0
    query_dist_ms[inf_ids] = 1.0
    query_dist_ms[query_dist_ms < -1.0] = -1.0
    query_dist_ms[query_dist_ms > 1.0] = 1.0
    np.save(file_out_query_dist, query_dist_ms.astype(np.float32))

    if debug and file_out_query_vis is not None:
        # save visualization
        sdf.visualize_query_points(query_pts_ms, query_dist_ms, file_out_query_vis)

def get_patch_radius(grid_res, epsilon):
    return (1.0 + epsilon) / grid_res

def run(args):
    if args.input_mesh[-4:] != ".ply":
        exit(2)

    debug = True
    patch_radius = get_patch_radius(args.grid_resolution, args.epsilon)
    output_query_dist = os.path.join(args.output_query_dist, os.path.basename(args.input_mesh) + ".npy")
    output_query_pts = os.path.join(args.output_query_pts, os.path.basename(args.input_mesh) + ".npy")

    # random state for file name
    rng = np.random.RandomState(fs.filename_to_hash(args.input_mesh))

    in_mesh = trimesh.load(args.input_mesh)

    # get query pts
    query_pts_ms = sdf.get_query_pts_for_mesh(
        in_mesh, args.num_query_pts, patch_radius, args.far_query_pts_ratio, rng)
    np.save(output_query_pts, query_pts_ms.astype(np.float32))

    # get signed distance
    query_dist_ms = sdf.get_signed_distance(
        in_mesh, query_pts_ms, args.signed_distance_batch_size)
    # fix NaNs, Infs and truncate
    nan_ids = np.isnan(query_dist_ms)
    inf_ids = np.isinf(query_dist_ms)
    query_dist_ms[nan_ids] = 0.0
    query_dist_ms[inf_ids] = 1.0
    query_dist_ms[query_dist_ms < -1.0] = -1.0
    query_dist_ms[query_dist_ms > 1.0] = 1.0
    np.save(output_query_dist, query_dist_ms.astype(np.float32))

    if debug and args.output_query_vis is not None:
        # save visualization
        output_query_vis = os.path.join(args.output_query_vis, os.path.basename(args.input_mesh)[:-4] + ".ply")
        sdf.visualize_query_points(query_pts_ms, query_dist_ms, output_query_vis)


def main(args):
    parser = argparse.ArgumentParser(prog="get_query_dist")
    parser.add_argument("--signed-distance-batch-size", type=int, help="signed_distance_batch_size", default=5000)
    parser.add_argument("--num_query_pts", type=int, help="num_patches_per_shape", default=2000)
    parser.add_argument("--skip-distances", action="store_true", help="skip_distances", default=False)
    parser.add_argument("--grid_resolution", type=str, help="output query vis", default=256)
    parser.add_argument("--epsilon", type=str, help="output query vis", default=5)
    parser.add_argument("--far_query_pts_ratio", type=str, help="output query vis", default=0.5)

    parser.add_argument("input_mesh", type=str, help="input mesh")
    parser.add_argument("output_query_dist", type=str, help="output query dist")
    parser.add_argument("output_query_pts", type=str, help="output query pts")
    parser.add_argument("output_query_vis", type=str, help="output query vis")
    args = parser.parse_args(args)

    run(args)


if __name__ == '__main__':
    main(sys.argv[1:])
