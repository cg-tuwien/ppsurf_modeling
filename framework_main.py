import os
from sys import stdout, stderr
import argparse
from pathlib import Path

from source.framework import global_config
from source.framework.framework_manager import FrameworkManager
from source.framework.framework_shell import FrameworkShell
from source.framework.logger import LoggingManager


def parse_arguments(args=None):
    argument_parser = argparse.ArgumentParser(prog="main",
                                              description="Executes the dataset generation framework")
    argument_parser.add_argument("-i", "--interactive", action="store_true",
                                 help="Run interactively (with shell)")
    argument_parser.add_argument("-s", "--schema", type=str, default='experiments/configSchema.xsd',
                                 help="Path to XML schema file (.xsd)")
    argument_parser.add_argument("-x", "--xml", type=str, default='experiments/p2s_abc_train.xml',
                                 help="Path to XML pipeline description (.xml)")
    argument_parser.add_argument("-l", "--log_dir", type=str, default='logs_framework',
                                 help="Directory of logs")
    argument_parser.add_argument("-w", "--worker_processes", type=str, default='4',
                                 help="number of worker processes")
    return argument_parser.parse_args(args=args)


def framework_main(args):

    print('PWD: {}'.format(os.getcwd()))

    config = args.xml
    schema = args.schema
    num_workers = int(args.worker_processes)
    global_config.set_max_processors(num_workers)

    try:
        log_folder = os.path.join(args.log_dir, Path(config).stem)
        if not os.path.exists(log_folder):
            os.makedirs(log_folder, exist_ok=True)
        if os.path.isdir(log_folder):
            global_config.set_log_folder(log_folder, init=True)
        else:
            print("Log folder is no valid folder!", file=stderr)
            exit(1)
        LoggingManager.init()

        if not os.path.isfile(config):
            print("Config file {} does not exist!".format(os.path.abspath(config)), file=stderr)
            exit(1)
        if not os.path.isfile(schema):
            print("XML file {} does not exist!".format(os.path.abspath(config)), file=stderr)
            exit(1)

        if args.interactive:
            shell = FrameworkShell(config, schema)
            shell.cmdloop()
        else:
            framework_manager = FrameworkManager(config, schema, stdout)
            framework_manager.parse_config()
            print("Running framework now!", flush=True)
            framework_manager.run()
            framework_manager.save_config()

    except KeyboardInterrupt:
        pass
    except Exception:
        print("An exception occurred! Please see log file for more information!", file=stderr)
        LoggingManager.get_error_logger().exception("Exception in framework main():")


if __name__ == '__main__':
    args = parse_arguments()
    framework_main(args)
