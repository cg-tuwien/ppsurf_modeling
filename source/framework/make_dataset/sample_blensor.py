import argparse
import os
import subprocess
import sys

import numpy as np
import trimesh.transformations as trafo
from source.base import fs, point_cloud, utils


def _pcd_files_to_pts(pcd_files, pts_file_npy, pts_file, obj_locations, obj_rotations, min_pts_size=0):
    """
    Convert pcd blensor results to xyz or directly to npy files. Merge front and back scans.
    Moving the object instead of the camera because the point cloud is in some very weird space that behaves
    crazy when the camera moves. A full day wasted on this shit!
    :param pcd_files:
    :param pts_file_npy:
    :param pts_file:
    :param trafos_inv:
    :return:
    """

    debug = False

    import gzip

    # https://www.blensor.org/numpy_import.html
    def extract_xyz_from_blensor_numpy(arr_raw):
        # timestamp
        # yaw, pitch
        # distance,distance_noise
        # x,y,z
        # x_noise,y_noise,z_noise
        # object_id
        # 255*color[0]
        # 255*color[1]
        # 255*color[2]
        # idx
        hits = arr_raw[arr_raw[:, 3] != 0.0]  # distance != 0.0 --> hit
        noisy_xyz = hits[:, [8, 9, 10]]
        return noisy_xyz

    pts_data_to_cat = []
    for fi, f in enumerate(pcd_files):
        try:
            if f.endswith('.numpy.gz'):
                pts_data_vs = extract_xyz_from_blensor_numpy(np.loadtxt(gzip.GzipFile(f, "r")))
            elif f.endswith('.numpy'):
                pts_data_vs = extract_xyz_from_blensor_numpy(np.loadtxt(f))
            elif f.endswith('.pcd'):
                pts_data_vs, header_info = point_cloud.load_pcd(file_in=f)
            else:
                raise ValueError('Input file {} has an unknown format!'.format(f))
        except EOFError as er:
            print('Error processing {}: {}'.format(f, er))
            continue

        # undo coordinate system changes
        pts_data_vs = utils.right_handed_to_left_handed(pts_data_vs)

        # move back from camera distance, always along x axis
        obj_location = np.array(obj_locations[fi])
        revert_offset(pts_data_vs, -obj_location)

        # get and apply inverse rotation matrix of camera
        scanner_rotation_inv = trafo.quaternion_matrix(trafo.quaternion_conjugate(obj_rotations[fi]))
        pts_data_ws_test_inv = trafo.transform_points(pts_data_vs, scanner_rotation_inv, translate=False)
        pts_data_ws = pts_data_ws_test_inv

        if pts_data_ws.shape[0] > 0:
            pts_data_to_cat += [pts_data_ws.astype(np.float32)]

        # debug outputs to check the rotations... the point cloud MUST align exactly with the mesh
        if debug:
            point_cloud.write_xyz(file_path=os.path.join('debug', 'test_{}.xyz'.format(str(fi))), points=pts_data_ws)

    if len(pts_data_to_cat) > 0:
        pts_data = np.concatenate(tuple(pts_data_to_cat), axis=0)

        if pts_data.shape[0] > min_pts_size:
            point_cloud.write_xyz(file_path=pts_file, points=pts_data)
            np.save(pts_file_npy, pts_data)


def revert_offset(pts_data: np.ndarray, inv_offset: np.ndarray):
    pts_reverted = pts_data
    if pts_reverted.shape[0] > 0:  # don't just check the header because missing rays may be added with NaNs
        pts_offset_correction = np.broadcast_to(inv_offset, pts_reverted.shape)
        pts_reverted += pts_offset_correction

    return pts_reverted


def run(args):
    input_file = args.input

    if input_file[-4:] != ".ply":
        exit(2)

    input_file_blensor_script = os.path.join(args.blensor_scripts, os.path.basename(input_file)[:-4] + ".py")

    with open('blensor_script_template.py', 'r') as file:
        blensor_script_template = file.read()

    rnd = np.random.RandomState(fs.filename_to_hash(input_file))
    num_scans = rnd.randint(args.num_scans_per_mesh_min, args.num_scans_per_mesh_max + 1)
    noise_sigma = rnd.rand() * (args.scanner_noise_sigma_max - args.scanner_noise_sigma_min) + \
                  args.scanner_noise_sigma_min

    new_pcd_base_files = []
    new_pcd_noisy_files = []
    new_obj_locations = []
    new_obj_rotations = []

    for num_scan in range(num_scans):
        pcd_base_file = os.path.join(
            args.output_pcd, os.path.basename(input_file)[:-4] + '_{num}.numpy.gz'.format(num=str(num_scan).zfill(5)))
        pcd_noisy_file = pcd_base_file[:-9] + '00000.numpy.gz'

        obj_location = (rnd.rand(3) * 2.0 - 1.0)
        obj_location_rand_factors = np.array([0.1, 1.0, 0.1])
        obj_location *= obj_location_rand_factors
        obj_location[1] += 4.0  # offset in cam view dir
        obj_rotation = trafo.random_quaternion(rnd.rand(3))

        # extend lists of pcd output files
        new_pcd_base_files.append(pcd_base_file)
        new_pcd_noisy_files.append(pcd_noisy_file)
        new_obj_locations.append(obj_location.tolist())
        new_obj_rotations.append(obj_rotation.tolist())

    new_scan_sigmas = [noise_sigma] * num_scans

    blensor_script = blensor_script_template.format(
        file_loc=input_file,
        obj_locations=str(new_obj_locations),
        obj_rotations=str(new_obj_rotations),
        evd_files=str(new_pcd_base_files),
        scan_sigmas=str(new_scan_sigmas),
    )
    blensor_script = blensor_script.replace('\\', '/')  # '\' would require escape sequence

    with open(input_file_blensor_script, "w") as text_file:
        text_file.write(blensor_script)

    # start blender with python script (-P) and close without prompt (-b)
    blender_blensor_call = '"{}" -P "{}" -b'.format(args.blensor_path, input_file_blensor_script)
    process = subprocess.run(blender_blensor_call, shell=True,
                             stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    print(process.stdout, process.stderr)

    def get_pcd_origin_file(pcd_file):
        origin_file = os.path.basename(pcd_file)[:-9] + '.xyz'
        origin_file = origin_file.replace('00000.xyz', '.xyz')
        origin_file = origin_file.replace('_noisy.xyz', '.xyz')
        origin_file = origin_file.replace('_00000.xyz', '.xyz')
        return origin_file

    print('### convert pcd to pts')
    call_params = None
    pcd_files_abs = [os.path.join(args.output_pcd, os.path.basename(f)) for f in new_pcd_noisy_files]
    pcd_origin = get_pcd_origin_file(new_pcd_noisy_files[0])
    xyz_file = os.path.join(args.output_vis, pcd_origin)
    xyz_npy_file = os.path.join(args.output, pcd_origin + '.npy')

    if fs.call_necessary(pcd_files_abs, [xyz_npy_file, xyz_file]):
        call_params = (pcd_files_abs, xyz_npy_file, xyz_file, new_obj_locations, new_obj_rotations, args.min_pts_size)

    _pcd_files_to_pts(*call_params)


def main(args):
    parser = argparse.ArgumentParser(prog="sample_blensor")
    parser.add_argument("--num_scans_per_mesh_min", type=int, help="num_scans_per_mesh_min", default=5)
    parser.add_argument("--num_scans_per_mesh_max", type=int, help="num_scans_per_mesh_max", default=30)
    parser.add_argument("--scanner_noise_sigma_min", type=float, help="scanner_noise_sigma_min", default=0.0)
    parser.add_argument("--scanner_noise_sigma_max", type=float, help="scanner_noise_sigma_max", default=0.05)
    parser.add_argument("--min_pts_size", type=int, help="min_pts_size", default=10)
    parser.add_argument("blensor_path", type=str, help="path to blensor")
    parser.add_argument("input", type=str, help="input file")
    parser.add_argument("blensor_scripts", type=str, help="blensor script location")
    parser.add_argument("output", type=str, help="output location")
    parser.add_argument("output_vis", type=str, help="output vis location")
    parser.add_argument("output_pcd", type=str, help="output pcd location")
    args = parser.parse_args(args)
    run(args)


if __name__ == '__main__':
    main(sys.argv[1:])
