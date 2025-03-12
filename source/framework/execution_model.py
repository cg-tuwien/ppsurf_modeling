import os
from abc import ABC, abstractmethod
from source.framework.process_pool import ProcessPool
from source.framework import global_config
from source.framework.logger import LoggingManager
from source.framework.utility import IOType, GraphIterator, ExecutionModelHelper, ExecutionModelEnum
from enum import Enum
from multiprocessing import Pool, cpu_count
from tqdm import tqdm
from itertools import tee


class ExecutionModel(ABC):
    @abstractmethod
    def execute(self, graph):
        pass


class Sequential(ExecutionModel):
    def execute(self, graph):
        if graph is not None:
            LoggingManager.start_new_run()
            graph_iter = GraphIterator(graph)
            graph_iter, graph_iter2 = tee(graph_iter)
            for call in tqdm(graph_iter, desc="Edges executed", total=len(list(graph_iter2))):
                call.process(ExecutionModelEnum.SEQUENTIAL)


class SimpleProcessPool(ExecutionModel):
    def execute(self, graph):
        if graph is not None:
            LoggingManager.start_new_run()
            graph_iter = GraphIterator(graph)
            graph_iter, graph_iter2 = tee(graph_iter)
            for call in tqdm(graph_iter, desc="Edges executed", total=len(list(graph_iter2))):
                call.process(ExecutionModelEnum.SIMPLE_PROCESS_POOL)


class GlobalProcessPool(ExecutionModel):
    def execute(self, graph):
        if graph is not None:
            LoggingManager.start_new_run()
            graph_iter = GraphIterator(graph)
            graph_iter, graph_iter2 = tee(graph_iter)
            try:
                pool = ProcessPool(self._get_starting_files(next(graph_iter)), list(graph_iter2),
                                   num_processes=global_config.get_max_processors())
                pool.start()
                pool.join()
            except NotImplementedError as ex:
                print(ex)

    def _get_starting_files(self, starting_edge):
        inputs = [f[1] for f in starting_edge.get_format_order() if f[0] is IOType.INPUT]

        input_files = []
        for i in inputs:
            tmp = []
            for f in os.listdir(i):
                if os.path.isfile(os.path.join(i, f)):
                    tmp.append(os.path.join(i, f))

            input_files.append(tmp)

        result = []
        if len(input_files) > 1:
            raise NotImplementedError("Path must be changed here!")
            tmp = []
            for f in input_files[0]:
                tmp.clear()
                tmp.append(f)
                for i in input_files[1:]:
                    tmp.append(ExecutionModelHelper.find_best_match(f, i))

                result.append(tuple(tmp))
        else:
            result = input_files[0]

        return result
