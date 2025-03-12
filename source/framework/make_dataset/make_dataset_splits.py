import argparse
import os
import random
import sys

from source.base import fs


def run(args):
    rnd = random.Random(args.seed)

    # write files for train / test / eval set
    final_out_dir_abs = args.final_output
    final_subdirs = []
    for folder in os.listdir(final_out_dir_abs):
        folder_abs = os.path.join(final_out_dir_abs, folder)
        if os.path.isdir(folder_abs):
            for file in os.listdir(folder_abs):
                if os.path.isfile(os.path.join(folder_abs, file)):
                    final_subdirs.append(file)
        elif os.path.isfile(folder_abs):
            final_subdirs.append(folder)

    final_output_files = [f for f in final_subdirs
                          if f[-4:] == '.npy']
    files_dataset = [f[:-4] for f in final_output_files]

    if len(files_dataset) == 0:
        raise ValueError('Dataset is empty!')

    if args.only_test_set:
        files_test = files_dataset
    else:
        files_test = rnd.sample(files_dataset, max(3, min(int(0.1 * len(files_dataset)), 100)))  # 3..100, ~10%
    files_train = list(set(files_dataset).difference(set(files_test)))

    files_test.sort()
    files_train.sort()

    file_test_set = os.path.join(args.set_folder, 'testset.txt')
    file_train_set = os.path.join(args.set_folder, 'trainset.txt')
    file_val_set = os.path.join(args.set_folder, 'valset.txt')

    fs.make_dir_for_file(file_test_set)
    nl = '\n'
    file_test_set_str = nl.join(files_test)
    file_train_set_str = nl.join(files_train)
    with open(file_test_set, "w") as text_file:
        text_file.write(file_test_set_str)
    with open(file_train_set, "w") as text_file:
        text_file.write(file_train_set_str)
    with open(file_val_set, "w") as text_file:
        text_file.write(file_test_set_str)  # validate the test set by default


def main(args):
    parser = argparse.ArgumentParser(prog="make_dataset_splits")
    parser.add_argument("-s", "--seed", type=int, help="seed", default=42)
    parser.add_argument("--only-test-set", action="store_true", help="only_test_set", default=False)

    parser.add_argument("set_folder", type=str, help="location where to save final sets")
    parser.add_argument("final_output", type=str, help="final output")
    args = parser.parse_args(args)

    run(args)


if __name__ == '__main__':
    main(sys.argv[1:])
