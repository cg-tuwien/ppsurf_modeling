import os
import argparse
import sys

import trimesh


def _convert_mesh(in_mesh, out_mesh):
    mesh = None
    try:
        mesh = trimesh.load(in_mesh)
    except AttributeError as e:
        print(e)
    except IndexError as e:
        print(e)
    except ValueError as e:
        print(e)
    except NameError as e:
        print(e)

    if mesh is not None:
        try:
            mesh.export(out_mesh)
        except ValueError as e:
            print(e)


def run(args):
    allowed_mesh_types = ['.off', '.ply', '.obj', '.stl']

    if args.input and args.output and args.input[-4:] in allowed_mesh_types:
        file_base_name = os.path.basename(args.input)
        _convert_mesh(args.input, os.path.join(args.output, file_base_name[:-4] + args.target_file_type))


def main(args):
    parser = argparse.ArgumentParser(prog="convert_mesh")
    parser.add_argument("-t", "--target-file-type", type=str, help="target file type", required=True)
    parser.add_argument("input", type=str, help="input file")
    parser.add_argument("output", type=str, help="output location")
    args = parser.parse_args(args)

    run(args)


if __name__ == '__main__':
    main(sys.argv[1:])
