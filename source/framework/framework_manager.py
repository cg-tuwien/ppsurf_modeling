import sys
import os
from os import listdir, remove, path, rmdir
from threading import Thread, Event
from networkx.drawing.nx_agraph import to_agraph
from source.framework import global_config
from source.framework.config_manager import ConfigManager
from source.framework.config_parser import ConfigParser
from source.framework.logger import LoggingManager
from source.framework.utility import GraphIterator, IOType
from source.framework.execution_model import ExecutionModelEnum, Sequential, SimpleProcessPool, GlobalProcessPool


class FrameworkManager(Thread):
    def __init__(self, config, config_schema, user_feedback=sys.stdout, error_log=sys.stderr):
        Thread.__init__(self)
        self._config_parser = ConfigParser(config, config_schema, user_feedback, error_log)
        self._config_manager = ConfigManager(os.path.join(global_config.get_log_folder(), 'config.ini'), config)
        self._config_manager.load()
        self._graph = None
        self._set_execution_model()
        self._user_feedback = user_feedback
        self._error_log = error_log
        self._save_config = True

    def _set_execution_model(self):
        if global_config.get_execution_model() == ExecutionModelEnum.SEQUENTIAL:
            self._execution_model = Sequential()
        elif global_config.get_execution_model() == ExecutionModelEnum.SIMPLE_PROCESS_POOL:
            self._execution_model = SimpleProcessPool()
        elif global_config.get_execution_model() == ExecutionModelEnum.GLOBAL_PROCESS_POOL:
            self._execution_model = GlobalProcessPool()
        else:
            print("Unknown execution model!", file=self._user_feedback)

    def run(self):
        if self._graph is not None:
            if global_config.is_dirty_config():
                # if input("The config seems to have changed! Do you want to clean the output folders and logs? [y/N] ") == "y":
                #     self.dirty_config_cleanup()
                self._save_config = True
                self._execution_model.execute(self._graph)
                # else:
                #     self._save_config = False
            else:
                self._execution_model.execute(self._graph)
        else:
            print("No graph available!")

    def dirty_config_cleanup(self):
        def clean_files_from_folder(folder_name):
            if path.isdir(folder_name):
                for f in listdir(folder_name):
                    f_full_path = path.join(folder_name, f)
                    if path.isfile(f_full_path):
                        remove(f_full_path)
                    elif path.isdir(f_full_path):
                        clean_files_from_folder(f_full_path)
                        rmdir(f_full_path)

        for call in GraphIterator(self._graph):
            folders_to_clean = [folder for (type, folder) in call.get_format_order() if type == IOType.OUTPUT]

            for folder in folders_to_clean:
                clean_files_from_folder(folder)

        LoggingManager.empty_db()
        with open(path.join(global_config.get_log_folder(), "error.log"), "w") as error_log:
            error_log.write("")

    def save_config(self):
        if self._save_config:
            self._config_manager.save()

    def set_config(self, new_config):
        self._config_parser.set_config(new_config)

    def set_schema(self, new_schema):
        self._config_parser.set_schema(new_schema)

    def update_execution_model(self):
        self._set_execution_model()

    def parse_config(self):
        if self._config_parser.validate_config():
            result = self._config_parser.parse_config()

            if result is not None:
                self._graph = result

    def traverse_graph(self):
        if self._graph is not None:
            for call in GraphIterator(self._graph):
                print(call)

    def draw_graph(self, file_name):
        if self._graph is not None:
            a = to_agraph(self._graph)
            a.draw(file_name + ".png", prog="dot")
