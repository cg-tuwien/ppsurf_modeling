from functools import wraps
from sys import stdout
from time import perf_counter
from source.framework import global_config
from source.framework.framework_manager import FrameworkManager
from source.framework.execution_model import ExecutionModelEnum


class Benchmark:
    measurements = []

    @classmethod
    def add_measurement(cls, measurement):
        cls.measurements.append(measurement)

    @classmethod
    def get_measurement(cls):
        if len(cls.measurements) > 0:
            return cls.measurements.pop(0)

    @classmethod
    def get_measurements(cls):
        return cls.measurements


def measure_time(func):
    @wraps(func)
    def wrapper(*args, **kwargs):
        start = perf_counter()
        result = func(*args, **kwargs)
        time = perf_counter() - start
        Benchmark.add_measurement((func.__name__, time))
        return result

    return wrapper


@measure_time
def measured_run(manager):
    manager.simple_run()


if __name__ == '__main__':
    global_config.set_execution_model(ExecutionModelEnum.SEQUENTIAL)
    framework_manager = FrameworkManager("/home/max/Desktop/BA-Repo/framework/testConfig.xml",
                                         "/home/max/Desktop/BA-Repo/framework/configSchema.xsd",
                                         stdout)
    framework_manager.parse_config()

    print("Start Benchmark now!")
    measured_run(framework_manager)
    global_config.set_execution_model(ExecutionModelEnum.SIMPLE_PROCESS_POOL)
    framework_manager.update_execution_model()
    measured_run(framework_manager)
    print("Benchmark finished!")
    print(Benchmark.get_measurements())
    seq = Benchmark.get_measurements()[0][1]
    par = Benchmark.get_measurements()[1][1]
    print("Speed gain: {:05.2f}%".format((par / seq) * 100))
