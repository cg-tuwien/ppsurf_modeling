from cmd import Cmd
from source.framework import global_config
from source.framework.framework_manager import FrameworkManager
from source.framework.execution_model import ExecutionModelEnum
from source.framework.logger import LoggingManager


class FrameworkShell(Cmd):
    prompt = "=> "
    intro = "Welcome to the dataset generation framework!"

    def __init__(self, config, schema):
        super().__init__()
        self.config = config
        self.schema = schema

    def get_manager(self):
        return self.manager

    def preloop(self):
        self.manager = FrameworkManager(self.config,
                                         self.schema,
                                         self.stdout)
        self.manager.parse_config()

    def postloop(self):
        self.manager.save_config()

    def do_EOF(self, line):
        return True

    def do_exit(self, line):
        """exit
           Exits the shell"""
        return True

    def do_quit(self, line):
        """quit
           Exits the shell"""
        return True

    def do_set_model(self, line):
        """set_model [execution model]
           Sets the execution model"""
        model = ExecutionModelEnum.from_str(line)

        if model is not None:
            global_config.set_execution_model(model)
            self.manager.update_execution_model()
        else:
            print("Unknown execution model!", file=self.stdout)

    def do_set_config(self, line):
        """set_config [path]
           Takes new config from given path"""
        print("New config: {}".format(line), file=self.stdout)
        self.manager.set_config(line)

    def do_set_schema(self, line):
        """set_schema [path]
           Takes new config schema from given path"""
        print("New config schema: {}".format(line), file=self.stdout)
        self.manager.set_schema(line)

    def do_parse_config(self, line):
        """parse_config
           Parses configuration again"""
        self.manager.parse_config()

    def do_graph(self, line):
        """graph
           Returns edges in dfs order"""
        self.manager.traverse_graph()

    def do_draw(self, line):
        """draw [file]
           Saves the graph in given file"""
        if line is not None:
            self.manager.draw_graph(line)

    def do_run(self, line):
        """run
           Runs pipeline"""
        self.manager.run()
        self.manager.save_config()

    def do_log_file(self, line):
        """log_file [file]
           Prints output of given file to console"""
        if line is not None and line != "":
            LoggingManager.get_log_file(line)

    def do_log_all(self, line):
        """log_all [file]
           Writes all outputs of last run into given log file"""
        if line is not None and line != "":
            LoggingManager.get_log_all_files(line)

    def do_log_edges(self, line):
        """log_edges
           Prints overview of edges (how many calls/how many successful calls) to console"""
        LoggingManager.get_log_all_edges()

    def do_log_which_calls(self, line):
        """log_which_calls [file]
           Prints calls which are responsible for creating given file"""
        if line is not None and line != "":
            LoggingManager.get_log_which_calls(line)
