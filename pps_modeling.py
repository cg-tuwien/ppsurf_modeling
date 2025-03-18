import sys
from pps import cli_main


def main():
    # TODO: Task 3
    # TODO: Run this
    name = 'pps_modeling'

    # Change these if you run out of memory
    # workers = 4  # 5 GB RAM
    workers = 8  # 8 GB RAM
    # batch_size = 2  # 5 GB VRAM
    batch_size = 4  # 8 GB VRAM
    max_epochs = 200

    # train
    sys.argv = ['pps.py',
                'fit',
                '-c', 'configs/pps_modeling.yaml',
                '--model.init_args.name', name,
                '--data.workers', str(workers),
                '--data.batch_size', str(batch_size),
                '--trainer.max_epochs', str(max_epochs),
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
                '--data.workers', str(workers),
                '--data.batch_size', str(batch_size),
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
                '--data.workers', str(workers),
                '--data.batch_size', str(batch_size),
                '--trainer.default_root_dir', 'models/{}'.format(name),
                '--ckpt_path', 'models/{}/alpha/checkpoints/last.ckpt'.format(name),
                # '--print_config'
                ]
    cli_main()


if __name__ == '__main__':
    main()
