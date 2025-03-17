import sys
from pps import cli_main


def main():
    # TODO: Task 3
    # TODO: Run this
    name = 'pps_modeling'

    # train
    sys.argv = ['pps.py',
                'fit',
                '-c', 'configs/pps_modeling.yaml',
                '--model.init_args.name', name,
                '--trainer.default_root_dir', 'models/{}'.format(name),
                # '--debug', 'True',
                # '--print_config'
                ]
    cli_main()

    # test
    sys.argv = ['pps.py',
                'test',
                '-c', 'configs/pps_modeling.yaml',
                '--model.init_args.name', name,
                '--trainer.default_root_dir', 'models/{}'.format(name),
                '--ckpt_path', 'models/{}/alpha/checkpoints/last.ckpt'.format(name),
                # '--print_config'
                ]
    cli_main()

    # predict
    sys.argv = ['pps.py',
                'predict',
                '-c', 'configs/pps_modeling.yaml',
                '--model.init_args.name', name,
                '--trainer.default_root_dir', 'models/{}'.format(name),
                '--ckpt_path', 'models/{}/alpha/checkpoints/last.ckpt'.format(name),
                # '--print_config'
                ]
    cli_main()


if __name__ == '__main__':
    main()
