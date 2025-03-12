from source.framework.execution_model import ExecutionModelEnum

__version = None
__log_folder = r"call set_log_folder with init=True"
__execution_model = ExecutionModelEnum.GLOBAL_PROCESS_POOL
__max_processors = None

__dirty_config = False


def set_dirty_config():
    global __dirty_config
    __dirty_config = True


def unset_dirty_config():
    global __dirty_config
    __dirty_config = False


def is_dirty_config():
    return __dirty_config


def get_max_processors():
    return __max_processors


def set_max_processors(p):
    global __max_processors, __dirty_config
    if p != __max_processors:
        __max_processors = p
        set_dirty_config()


def get_execution_model():
    return __execution_model


def set_execution_model(m):
    global __execution_model, __dirty_config
    if m != __execution_model:
        __execution_model = m
        set_dirty_config()


def get_log_folder():
    return __log_folder


def set_log_folder(l, init=False):
    global __log_folder, __dirty_config
    if l != __log_folder:
        __log_folder = l
        if not init:
            set_dirty_config()


def get_version():
    return __version


def set_version(v):
    global __version, __dirty_config
    if v != __version:
        __version = v
        set_dirty_config()
