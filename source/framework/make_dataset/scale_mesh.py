import os
import argparse
import sys

import trimesh
import numpy as np


def _normalize_mesh(file_in: str, file_out: str):

    mesh: trimesh.Trimesh = trimesh.load(file_in)

    # TODO: Task 1 Start
    # TODO: normalize the mesh to [-0.5...+0.5]
    # See documentation of Trimesh: https://trimsh.org/trimesh.html#trimesh.Trimesh
    mesh.apply_transform(trimesh.transformations.identity_matrix())  # just a placeholder
    raise NotImplementedError('Task 1 is not implemented')  # delete this line when done
    # TODO: Task 1 End

    if np.min(mesh.vertices) < -0.5 or np.max(mesh.vertices) > 0.5:
        raise ValueError('Given mesh exceeds the boundaries!')

    mesh.export(file_out)


def run(args):
    if args.input and args.output:
        file_base_name = os.path.basename(args.input)
        _normalize_mesh(args.input, os.path.join(args.output, file_base_name))


def main(args):
    parser = argparse.ArgumentParser(prog="scale_mesh")
    parser.add_argument("input", type=str, help="input file")
    parser.add_argument("output", type=str, help="output location")
    args = parser.parse_args(args)

    run(args)


if __name__ == '__main__':
    main(sys.argv[1:])
