import multiprocessing
import threading
from multiprocessing import Process, Value, Event, Pool
from os import cpu_count, path, makedirs, listdir
from queue import Queue
from threading import Thread
from time import sleep

from tqdm import tqdm

from source.framework import global_config
from source.framework.logger import LoggingManager
from source.framework.utility import GraphIterator, IOType, ExecutionModelHelper, ExecutionModelEnum


class SharedAccessLimiter:
    def __init__(self, maximum_active, init_val=0):
        self.maximum_active = maximum_active
        self.active = Value("i", init_val)
        self.free = Event()
        self.free.set()

    def wait(self):
        self.free.wait()

    def inc(self):
        with self.active.get_lock():
            self.active.value += 1

            if self.active.value >= self.maximum_active:
                self.free.clear()

    def dec(self):
        with self.active.get_lock():
            self.active.value -= 1

            if self.active.value < self.maximum_active:
                self.free.set()


class WorkerProcess(Process):
    def __init__(self, calls, shared_counter, logging_queue):
        Process.__init__(self)
        self.calls = calls
        self.created_files = {}
        self.folder_name = None
        self.shared_counter = shared_counter
        self.logging_queue = logging_queue

    def set_file(self, file):
        if len(self.created_files) == 0:
            file_path, file_name = path.split(file)
            self.created_files[file_path] = [file_name]
            self.folder_name = "".join(file_name.split(".")[:-1])

    def run(self):
        if (len(self.created_files) > 0 and self.calls) is not None:
            for (i, call) in enumerate(self.calls):
                if call.get_require_complete() is not None and call.get_require_complete:
                    break

                result = call.process(ExecutionModelEnum.GLOBAL_PROCESS_POOL, (i, self.folder_name, self.created_files), logging_queue=self.logging_queue)

                if len(result) == 2 and result[0]:
                    for k in result[1].keys():
                        if k in self.created_files.keys():
                            self.created_files[k].append(result[1][k])
                        else:
                            self.created_files[k] = result[1][k]

                self.logging_queue.put(("Update",))

        self.shared_counter.dec()


class ProcessPool:
    def __init__(self, files, calls, num_processes=1):
        self.num_processes = num_processes if num_processes is not None and 0 < num_processes <= cpu_count() \
                                           else cpu_count()
        self.counter = SharedAccessLimiter(self.num_processes, 0)
        self.workers = Queue()
        self.all_workers = Queue()
        self.logging_queue = multiprocessing.Queue()
        for _ in range(len(files)):
            w = WorkerProcess(calls, self.counter, self.logging_queue)
            self.workers.put(w)
            self.all_workers.put(w)

        self.files = Queue()
        for f in files:
            self.files.put(f)

        self.calls = calls
        self.has_require_complete = False
        if len([c for c in self.calls if c.get_require_complete() is not None and c.get_require_complete()]) > 0:
            self.has_require_complete = True

        bar_size = self.files.qsize() * len([c for c in self.calls if c.get_require_complete is not None and not c.get_require_complete()])
        self.logging_thread = Thread(target=self.logging_thread_method, args=(bar_size,))

    def logging_thread_method(self, bar_size):
        bar = tqdm(desc="Calls made", total=bar_size)
        t = threading.currentThread()
        while getattr(t, "do_run", True):
            log_obj = self.logging_queue.get(block=True, timeout=None)
            if log_obj is not None and len(log_obj) == 5:
                LoggingManager.log_process_execution(log_obj[0], log_obj[1], log_obj[2], log_obj[3], log_obj[4])
            elif log_obj is not None and len(log_obj) == 1 and log_obj[0] == "Update":
                bar.update(1)
        bar.close()

    def start_logging(self):
        self.logging_thread.start()

    def stop_logging(self):
        if self.logging_thread.is_alive():
            self.logging_thread.do_run = False
            self.logging_queue.put(None)

    def start(self):
        def get_remaining_steps():
            # try:
            first_require_complete_element = next((c for c in self.calls if c.get_require_complete is not None and c.get_require_complete()), None)
            index = self.calls.index(first_require_complete_element) if first_require_complete_element is not None else None
            unique_edges = []
            till_index = 0
            for remaining_edge in self.calls[index:]:
                if remaining_edge.get_type() == "unique":
                    unique_edges.append(remaining_edge)
                    till_index += 1
                else:
                    break

            return (unique_edges, self.calls[index + till_index:]) if index is not None else None
            # except ValueError:
            #     return None

        def get_starting_files(starting_edge):
            inputs = [f[1] for f in starting_edge.get_format_order() if f[0] is IOType.INPUT]

            input_files = []
            for i in inputs:
                tmp = []
                for f in listdir(i):
                    if path.isfile(path.join(i, f)):
                        tmp.append(path.join(i, f))

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

        self.start_logging()
        while not self.files.empty():
            self.counter.wait()
            self.counter.inc()
            process = self.workers.get()
            file = self.files.get()
            process.set_file(file)
            process.start()

        if self.has_require_complete:
            self.join()
            remaining_steps = get_remaining_steps()
            if remaining_steps is not None:
                for remaining_edge in remaining_steps[0]:
                    remaining_edge.process(ExecutionModelEnum.SIMPLE_PROCESS_POOL)
                self.stop_logging()
                if len(remaining_steps[1]) > 0:
                    follow_pool = ProcessPool(get_starting_files(remaining_steps[1][0]), remaining_steps[1])
                    follow_pool.start()
                    follow_pool.join()
        else:
            self.join()

        self.stop_logging()

    def join(self):
        while not self.all_workers.empty():
            self.all_workers.get().join()
