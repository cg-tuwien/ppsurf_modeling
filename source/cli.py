import os
import sys
import typing
from typing import Any, Dict, List, Optional, Type, TYPE_CHECKING, Union, Set
from pathlib import Path
import abc

from tqdm import tqdm

import pytorch_lightning.profilers
from pytorch_lightning.cli import LightningCLI, Namespace, LightningArgumentParser
from pytorch_lightning.callbacks import TQDMProgressBar, ModelCheckpoint
from lightning_fabric.utilities.types import _PATH
from typing import Optional, Literal, Dict
from datetime import timedelta

from source.base.profiling import get_now_str


class PPSProgressBar(TQDMProgressBar):  # disable validation prog bar
    def init_validation_tqdm(self):
        bar_disabled = tqdm(disable=True)
        return bar_disabled

class TorchScriptModelCheckpoint(ModelCheckpoint):

    def __init__(
        self,
        dirpath: Optional[_PATH] = None,
        filename: Optional[str] = None,
        monitor: Optional[str] = None,
        verbose: bool = False,
        save_last: Optional[Literal[True, False, "link"]] = None,
        save_top_k: int = 1,
        save_weights_only: bool = False,
        mode: str = "min",
        auto_insert_metric_name: bool = True,
        every_n_train_steps: Optional[int] = None,
        train_time_interval: Optional[timedelta] = None,
        every_n_epochs: Optional[int] = None,
        save_on_train_epoch_end: Optional[bool] = None,
        enable_version_counter: bool = True,
    ):
        super().__init__(
            dirpath=dirpath,
            filename=filename,
            monitor=monitor,
            verbose=verbose,
            save_last=save_last,
            save_top_k=save_top_k,
            save_weights_only=save_weights_only,
            mode=mode,
            auto_insert_metric_name=auto_insert_metric_name,
            every_n_train_steps=every_n_train_steps,
            train_time_interval=train_time_interval,
            every_n_epochs=every_n_epochs,
            save_on_train_epoch_end=save_on_train_epoch_end,
            enable_version_counter=enable_version_counter,
        )

class PPSProfiler(pytorch_lightning.profilers.PyTorchProfiler):
    def __init__(
        self,
        dirpath: Optional[Union[str, Path]] = None,
        filename: Optional[str] = None,
        group_by_input_shapes: bool = False,
        emit_nvtx: bool = False,
        export_to_chrome: bool = True,
        row_limit: int = 20,
        sort_by_key: Optional[str] = None,
        record_module_names: bool = True,
        with_stack: bool = False,
        **profiler_kwargs: Any,  # can't by used with YAML
    ) -> None:
        prof_kwargs = {
            'profile_memory': True,
            'record_shapes': True,
        }
        super().__init__(dirpath=dirpath, filename=filename, group_by_input_shapes=group_by_input_shapes,
                         emit_nvtx=emit_nvtx, export_to_chrome=export_to_chrome, row_limit=row_limit,
                         sort_by_key=sort_by_key, record_module_names=record_module_names, with_stack=with_stack,
                         **prof_kwargs)


class CLI(LightningCLI):
    def __init__(self, model_class, subclass_mode_model, datamodule_class, subclass_mode_data):
        print('{}: Starting {}'.format(get_now_str(), ' '.join(sys.argv)))
        sys.argv = self.handle_rec_subcommand(sys.argv)  # only call this with args from system command line
        super().__init__(
            model_class=model_class, subclass_mode_model=subclass_mode_model,
            datamodule_class=datamodule_class, subclass_mode_data=subclass_mode_data,
            save_config_kwargs={'overwrite': True})
        print('{}: Finished {}'.format(get_now_str(), ' '.join(sys.argv)))

    def cur_config(self) -> Namespace:
        return self.config[self.config.subcommand]

    def add_arguments_to_parser(self, parser: LightningArgumentParser) -> None:
        # fundamentals
        parser.add_argument('--debug', type=bool, default=False,
                            help='set to True if you want debug outputs to validate the model')

    @abc.abstractmethod
    def handle_rec_subcommand(self, args: typing.List[str]) -> typing.List[str]:
        """
        Replace rec subcommand with predict and its default parameters before any argparse.
        Args:
            args: typing.List[str]

        Returns:
            new_args: typing.List[str]
        """
        pass

    # def before_fit(self):
    #     pass
    #
    # def after_fit(self):
    #     pass
    #
    # def before_predict(self):
    #     pass
    #
    # def after_predict(self):
    #     pass

    def before_instantiate_classes(self):
        import torch
        # torch.set_float32_matmul_precision('medium')  # PPSurf 50NN: 5.123h, ABC CD 0.012920511
        torch.set_float32_matmul_precision('high')  # PPSurf 50NN: xh, ABC CD y
        # torch.set_float32_matmul_precision('highest')  # PPSurf 50NN: xh, ABC CD y

        if bool(self.cur_config().debug):
            os.environ['CUDA_LAUNCH_BLOCKING'] = '1'
            os.environ['TORCH_DISTRIBUTED_DEBUG '] = '1'

            self.cur_config().trainer.detect_anomaly = True

    # def instantiate_classes(self):
    #     pass

    # def instantiate_trainer(self):
    #     pass

    # def parse_arguments(self, parser, args):
    #     pass

    # def setup_parser(self, add_subcommands, main_kwargs, subparser_kwargs):
    #     pass

    @staticmethod
    def subcommands() -> Dict[str, Set[str]]:
        """Defines the list of available subcommands and the arguments to skip."""
        return {
            'fit': {'model', 'train_dataloaders', 'val_dataloaders', 'datamodule'},
            # 'validate': {'model', 'dataloaders', 'datamodule'}, # no val for this
            'test': {'model', 'dataloaders', 'datamodule'},
            'predict': {'model', 'dataloaders', 'datamodule'},
            # 'tune': {'model', 'train_dataloaders', 'val_dataloaders', 'datamodule'},
        }
