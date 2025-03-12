import argparse
import sys

import trimesh
import os


def _clean_mesh(file_in, file_out, num_max_faces=None, enforce_solid=True):
    mesh = trimesh.load(file_in)

    mesh.process()
    mesh.remove_unreferenced_vertices()
    mesh.remove_degenerate_faces()
    mesh.remove_duplicate_faces()

    if not mesh.is_watertight:
        mesh.fill_holes()
        trimesh.repair.fill_holes(mesh)

    if enforce_solid and not mesh.is_watertight:
        return

    if not mesh.is_winding_consistent:
        trimesh.repair.fix_inversion(mesh, multibody=True)
        trimesh.repair.fix_normals(mesh, multibody=True)
        trimesh.repair.fix_winding(mesh)

    if enforce_solid and not mesh.is_winding_consistent:
        return

    if enforce_solid and not mesh.is_volume:  # watertight, consistent winding, outward facing normals
        return

    # large meshes might cause out-of-memory errors in signed distance calculation
    if num_max_faces is None:
        mesh.export(file_out)
    elif len(mesh.faces) < num_max_faces:
        mesh.export(file_out)


def run(args):
    if args.input and args.output:
        file_base_name = os.path.basename(args.input)
        _clean_mesh(args.input, os.path.join(args.output, file_base_name), args.num_max_faces, args.enforce_solid)


def main(args):
    parser = argparse.ArgumentParser(prog="convert_mesh")
    parser.add_argument("-n", "--num-max-faces", type=int, help="number max faces", default=None)
    parser.add_argument("-e", "--enforce-solid", action="store_true", help="enforce solid", default=True)
    parser.add_argument("input", type=str, help="input file")
    parser.add_argument("output", type=str, help="output location")
    args = parser.parse_args(args)

    run(args)


if __name__ == '__main__':
    main(sys.argv[1:])
