import sys
from multiprocessing import Pool, cpu_count
from os import path, listdir, makedirs
from importlib.util import spec_from_file_location, module_from_spec
from io import StringIO
import subprocess

from networkx import dfs_tree
import numpy as np
from enum import Enum
from Levenshtein import distance as levenshtein_distance

from source.framework import global_config
from source.framework.logger import LoggingManager


class GraphIterator:
    def __init__(self, graph):
        self._all_edges = {}

        for e in graph.edges:
            edge_info = graph.edges[e[0], e[1]]["info"]

            if edge_info not in self._all_edges:
                self._all_edges[edge_info] = [e]
            else:
                self._all_edges[edge_info].append(e)

        self._finished_nodes = [list(dfs_tree(graph))[0]]

    def __iter__(self):
        return self

    def __next__(self):
        call = self._get_executable_edge()

        if call is None:
            raise StopIteration

        for e in self._all_edges[call]:
            self._finished_nodes.append(e[1])
        del self._all_edges[call]

        return call

    def _get_executable_edge(self):
        for call in self._all_edges:
            valid = True

            for e in self._all_edges[call]:
                if e[0] not in self._finished_nodes:
                    valid = False

            if valid:
                return call

        return None


class ExecutionModelEnum(Enum):
    SEQUENTIAL = 0,
    SIMPLE_PROCESS_POOL = 1
    GLOBAL_PROCESS_POOL = 2

    @staticmethod
    def from_str(s):
        if s in ("SEQUENTIAL", "sequential", "seq", "ExecutionModelEnum.SEQUENTIAL"):
            return ExecutionModelEnum.SEQUENTIAL
        elif s in ("SIMPLE", "simple", "ExecutionModelEnum.SIMPLE_PROCESS_POOL"):
            return ExecutionModelEnum.SIMPLE_PROCESS_POOL
        elif s in ("GLOBAL", "global", "POOL", "pool", "ExecutionModelEnum.GLOBAL_PROCESS_POOL"):
            return ExecutionModelEnum.GLOBAL_PROCESS_POOL
        else:
            return None


class Edge:
    def __init__(self, name, description, call, format_order, type):
        self._name = name
        self._description = description
        self._call = call
        self._format_order = format_order
        self._type = type
        self._require_complete = None

    def set_require_complete(self, value):
        if type(value) == bool:
            self._require_complete = value

    def get_require_complete(self):
        return self._require_complete

    def get_name(self):
        return self._name

    def get_call(self):
        return self._call

    def get_format_order(self):
        return self._format_order

    def get_type(self):
        return self._type

    def __str__(self):
        return "Edge {}".format(self._name)

    def make_program_call(self, file, inputs, input_files, outputs, inputs_available=True, logging_queue=None):
        # Fetch the basic call command
        program_call = self.get_call()

        i = 0
        o = 0
        output_folders = []
        input_files_dict = {}
        params = []
        for (format_type, _) in self.get_format_order():
            if inputs_available and format_type == IOType.INPUT:
                input_file = ""
                input_directory = ""
                if i == 0:
                    input_file = file
                    input_directory = inputs[0]
                else:
                    input_file = ExecutionModelHelper.find_best_match(file, input_files[i])
                    input_directory = inputs[i]
                params.append(path.join(input_directory, input_file))
                if input_directory in input_files_dict.keys():
                    input_files_dict[input_directory].append(input_file)
                else:
                    input_files_dict[input_directory] = [input_file]
                i += 1
            elif format_type == IOType.OUTPUT:
                params.append(outputs[o])
                output_folders.append(outputs[o])
                o += 1

        for o in output_folders:
            if not path.exists(o):
                makedirs(o, exist_ok=True)

        # Insert the parameters in the placeholders of the call
        program_call = program_call.format(*params)
        # Execute command
        return ExecutionModelHelper.run_command(program_call, self.get_name(),
                                                None if not input_files_dict else input_files_dict, output_folders, logging_queue=logging_queue)

    def process(self, execution_model, process_pool_data=None, logging_queue=None):
        def check_call_necessary(input_files, output_folders):
            # In case of no input files a call is always necessary
            if len(input_files) <= 0:
                return True

            # Return early if output folders don't exist or are empty
            if all(map(lambda folder: len(listdir(folder)) == 0 if path.exists(folder) and path.isdir(folder) else True, output_folders)):
                return True

            # Determine available output files
            output_files = []
            for folder in output_folders:
                if path.exists(folder) and path.isdir(folder):
                    for file in listdir(folder):
                        f = path.join(folder, file)
                        if path.isfile(f):
                            output_files.append(f)

            return ExecutionModelHelper.call_necessary(input_files, output_files)

        if execution_model == ExecutionModelEnum.GLOBAL_PROCESS_POOL and (process_pool_data is None or len(process_pool_data) != 3):
            return False, None

        # Get list of input and output folders
        inputs = [f[1] for f in self.get_format_order() if f[0] is IOType.INPUT]
        if execution_model == ExecutionModelEnum.GLOBAL_PROCESS_POOL and process_pool_data[0] > 0:
            inputs = [path.join(f, process_pool_data[1]) for f in inputs]

        if execution_model == ExecutionModelEnum.GLOBAL_PROCESS_POOL:
            outputs = [path.join(f[1], process_pool_data[1]) for f in self.get_format_order() if f[0] is IOType.OUTPUT]
        else:
            outputs = [f[1] for f in self.get_format_order() if f[0] is IOType.OUTPUT]

        # Generate list of input files which are inside the input directories
        input_files = []
        for i in inputs:
            tmp = []
            if execution_model == ExecutionModelEnum.GLOBAL_PROCESS_POOL:
                if i in process_pool_data[2].keys():
                    for f in process_pool_data[2][i]:
                        if path.isfile(path.join(i, f)):
                            tmp.append(f)

                    input_files.append(tmp)
            else:
                for f in listdir(i):
                    if path.isfile(path.join(i, f)):
                        tmp.append(f)

                input_files.append(tmp)

        tmp = []
        for (index, filelist) in enumerate(input_files):
            for f in filelist:
                tmp.append(path.join(inputs[index], f))

        created_files = None
        if check_call_necessary(tmp, outputs):
            # Check if input files are specified
            if len(inputs) == 0:
                created_files = self.make_program_call(None, inputs, input_files, outputs, inputs_available=False, logging_queue=logging_queue)
            else:
                # Check if all input folders have the same amount of files inside them
                if len(set([len(l) for l in input_files])) != 1:
                    return False, None

                if execution_model == ExecutionModelEnum.SEQUENTIAL or execution_model == ExecutionModelEnum.GLOBAL_PROCESS_POOL:
                    # Process each input file in this loop
                    for file in input_files[0]:
                        created_files = self.make_program_call(file, inputs, input_files, outputs, logging_queue=logging_queue)
                elif execution_model == ExecutionModelEnum.SIMPLE_PROCESS_POOL:
                    file_processing_pool = Pool(cpu_count() if global_config.get_max_processors() is None
                                                else global_config.get_max_processors())
                    # Process each input file in this process pool
                    file_processing_pool.starmap(self.make_program_call,
                                                 [(file, inputs, input_files, outputs) for file in input_files[0]])
        else:
            created_files = {}
            for folder in outputs:
                created_files[folder] = []
                if path.exists(folder) and path.isdir(folder):
                    for file in listdir(folder):
                        tmp = path.join(folder, file)
                        if path.exists(tmp) and path.isfile(tmp):
                            created_files[folder].append(file)

        return True, created_files


class IOType(Enum):
    INPUT = 0,
    OUTPUT = 1


class ExecutionModelHelper:
    @staticmethod
    def find_best_match(input_file, relevant_files):
        distances = [(levenshtein_distance(input_file, f), f) for f in relevant_files]
        return str(min(distances)[1])

    @staticmethod
    def run_command(command, edge, input_files, output_folders, logging_queue=None):
        args = command.split(" ")

        if args[0] == "python":
            output = StringIO()
            sys.stdout = output
            name = path.splitext(path.basename(args[1]))[0]
            location = args[1]
            if not path.isfile(location):
                error_str = 'Failed to load Python module "{}" at "{}". File does not exist!'.format(name, location)
                print(error_str)
                raise Exception(error_str)
            called_module_spec = spec_from_file_location(name, location)
            if called_module_spec is not None:
                called_module = module_from_spec(called_module_spec)
                called_module_spec.loader.exec_module(called_module)
            else:
                raise Exception('Failed to load Python module "{}" at "{}"'.format(name, location))
            # try:
            called_module.main(args[2:])
            # except AttributeError as ex:
            #     print("Missing main method!")
            #     return_code = 1
            # except FileNotFoundError as ex:
            #     print(ex)
            #     return_code = 1
            # except OSError as ex:
            #     print(ex)
            #     return_code = 1
            # except Exception as ex:
            #     print(ex)
            #     return_code = 1
            # else:
            #     return_code = 0
            # finally:
            return_code = 0
            sys.stout = sys.__stdout__
            output = output.getvalue()
        else:
            process = subprocess.run(args, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                                     universal_newlines=True)
            output = process.stdout
            return_code = process.returncode

        created_files = ExecutionModelHelper.extract_created_files(output_folders)
        logging_information = (edge, input_files, created_files, return_code, output)
        if logging_queue is None:
            LoggingManager.log_process_execution(*logging_information)
        else:
            logging_queue.put(logging_information)

        return created_files

    @staticmethod
    def extract_created_files(folders):
        created_files = {}

        for folder in folders:
            files = [path.join(folder, file) for file in listdir(folder) if path.isfile(path.join(folder, file))]
            for file in files:
                file_path, file_name = path.split(file)
                if file_path in created_files.keys():
                    created_files[file_path].append(file_name)
                else:
                    created_files[file_path] = [file_name]

        return created_files

    @staticmethod
    def call_necessary(file_in, file_out, min_file_size=0):
        """
        Check if all input files exist and at least one output file does not exist or is invalid.
        :param file_in: list of str or str
        :param file_out: list of str or str
        :param min_file_size: int
        :return:
        """

        if isinstance(file_in, str):
            file_in = [file_in]
        elif isinstance(file_in, list):
            pass
        else:
            raise ValueError('Wrong input type')

        if isinstance(file_out, str):
            file_out = [file_out]
        elif isinstance(file_out, list):
            pass
        else:
            raise ValueError('Wrong output type')

        inputs_exist = all([path.isfile(f) for f in file_in])
        if not inputs_exist:
            print('WARNING: Input file {} does not exist'.format(file_in))
            return False

        outputs_exist = all([path.isfile(f) for f in file_out])
        if not outputs_exist:
            return True

        min_output_file_size = min([path.getsize(f) for f in file_out])
        if min_output_file_size < min_file_size:
            return True

        oldest_input_file_mtime = max([path.getmtime(f) for f in file_in])
        youngest_output_file_mtime = min([path.getmtime(f) for f in file_out])

        if oldest_input_file_mtime >= youngest_output_file_mtime:
            # debug
            import time
            input_file_mtime_arg_max = np.argmax(np.array([path.getmtime(f) for f in file_in]))
            output_file_mtime_arg_min = np.argmin(np.array([path.getmtime(f) for f in file_out]))
            input_file_mtime_max = time.strftime('%Y-%m-%d %H:%M:%S', time.gmtime(oldest_input_file_mtime))
            output_file_mtime_min = time.strftime('%Y-%m-%d %H:%M:%S', time.gmtime(youngest_output_file_mtime))
            print('Input file {} \nis newer than output file {}: \n{} >= {}'.format(
                file_in[input_file_mtime_arg_max], file_out[output_file_mtime_arg_min],
                input_file_mtime_max, output_file_mtime_min))
            return True

        return False
