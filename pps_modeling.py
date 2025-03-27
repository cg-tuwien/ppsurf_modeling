import sys
import time
from pps import cli_main

# TODO: Task 3: Run this

# You can find the configuration in configs/pps_modeling.yaml
# Change the name each run if you don't want to overwrite the previous results. It also keeps
# the results in the tensorboard separate.
name = 'pps_modeling' 
run_stages = {
    'train': True,
    'test': True,
    'predict': True
}

def print_stage_header(title):
    header_width = 62
    equal_signs = '=' * header_width
    title_line = f'== {title.ljust(header_width - 5)} ='
    print(f'''
{equal_signs}
{title_line}
{equal_signs}''')


def format_time(seconds):
    minutes = int(seconds // 60)
    seconds = int(seconds % 60)
    return f"{minutes}m {seconds}s"


def main():
    # Training stage
    if run_stages['train']:
        print_stage_header('TRAINING')
        start_time = time.time()
        sys.argv = ['pps.py',
                    'fit',
                    '-c', 'configs/pps_modeling.yaml',
                    '--model.init_args.name', name,
                    '--trainer.default_root_dir', 'models/{}'.format(name),
                    # '--debug', 'True',
                    # '--print_config'
                    ]
        cli_main()
        print(f"== Training completed in {format_time(time.time() - start_time)}\n")

    # Testing stage
    if run_stages['test']:
        print_stage_header('TESTING')
        start_time = time.time()
        sys.argv = ['pps.py',
                    'test',
                    '-c', 'configs/pps_modeling.yaml',
                    '--model.init_args.name', name,
                    '--trainer.default_root_dir', 'models/{}'.format(name),
                    '--ckpt_path', 'models/{}/alpha/checkpoints/last.ckpt'.format(name),
                    # '--print_config'
                    ]
        cli_main()
        print(f"== Testing completed in {format_time(time.time() - start_time)}\n")

    # Prediction stage
    if run_stages['predict']:
        print_stage_header('PREDICTION')
        start_time = time.time()
        sys.argv = ['pps.py',
                    'predict',
                    '-c', 'configs/pps_modeling.yaml',
                    '--model.init_args.name', name,
                    '--trainer.default_root_dir', 'models/{}'.format(name),
                    '--ckpt_path', 'models/{}/alpha/checkpoints/last.ckpt'.format(name),
                    # '--print_config'
                    ]
        cli_main()
        print(f"== Prediction completed in {format_time(time.time() - start_time)}\n")


if __name__ == '__main__':
    main()