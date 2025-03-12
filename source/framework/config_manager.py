from configparser import ConfigParser
from os import path, utime
from time import ctime, mktime
import datetime
from source.framework import global_config
from source.framework.execution_model import ExecutionModelEnum


class ConfigManager:
    def __init__(self, config_file, xml_config):
        self.parser = ConfigParser()
        self.config_file = config_file
        self.xml_config = xml_config

    def load(self):
        if self.read_config_from_file():
            self.update_global_config()
            if self.check_if_file_changed(self.config_file, "INI"):
                global_config.set_dirty_config()
            if self.check_if_file_changed(self.xml_config, "XML"):
                global_config.set_dirty_config()
        else:
            global_config.set_dirty_config()

    def save(self):
        self.read_from_global_config()
        if self.write_config_to_file():
            global_config.unset_dirty_config()

    def write_config_to_file(self):
        if not path.exists(self.config_file):
            with open(self.config_file, "w") as file:
                file.write("")

        if path.exists(self.config_file) and path.isfile(self.config_file):
            timestamp = datetime.datetime.now()
            timestamp = timestamp.replace(microsecond=0)
            self.add_entry("MOD-DATES", {"INI": str(timestamp)})
            self.write_mod_date_to_config(self.xml_config, "XML")
            with open(self.config_file, "w") as file:
                self.parser.write(file)

            mod_time = mktime(timestamp.timetuple())
            utime(self.config_file, (mod_time, mod_time))
            return True

        return False

    def write_mod_date_to_config(self, file_name, ini_entry):
        if path.exists(file_name) and path.isfile(file_name):
            file_mod_date = datetime.datetime.strptime(ctime(path.getmtime(file_name)), "%a %b %d %H:%M:%S %Y")
            file_mod_date = file_mod_date.replace(microsecond=0)

            self.add_entry("MOD-DATES", {ini_entry: str(file_mod_date)})

    def read_config_from_file(self):
        if path.exists(self.config_file) and path.isfile(self.config_file):
            self.parser.read(self.config_file)
            return True

        return False

    def read_from_global_config(self):
        self.add_entry("CONFIG", {
            "version": global_config.get_version(),
            "log_folder": global_config.get_log_folder(),
            "execution_model": global_config.get_execution_model(),
            "max_processors": global_config.get_max_processors()
        })

    def update_global_config(self):
        if "CONFIG" in self.parser:
            tmp = True if global_config.is_dirty_config() else False
            for key in self.parser["CONFIG"]:
                value = None
                func = None

                try:
                    if key == "version":
                        func = global_config.set_version
                        value = float(self.parser["CONFIG"][key])
                    elif key == "log_folder":
                        func = global_config.set_log_folder
                        value = self.parser["CONFIG"][key]
                    elif key == "execution_model":
                        func = global_config.set_execution_model
                        value = ExecutionModelEnum.from_str(self.parser["CONFIG"][key])
                    elif key == "max_processors":
                        func = global_config.set_max_processors
                        value = int(self.parser["CONFIG"][key])
                except ValueError:
                    value = None
                finally:
                    if func is not None:
                        func(value)

            if tmp:
                global_config.set_dirty_config()
            else:
                global_config.unset_dirty_config()

    def check_if_file_changed(self, file_name, ini_entry):
        if "MOD-DATES" in self.parser and ini_entry in self.parser["MOD-DATES"] and path.exists(file_name) and path.isfile(file_name):
            saved_mod_date = datetime.datetime.strptime(self.parser["MOD-DATES"][ini_entry], "%Y-%m-%d %H:%M:%S")
            saved_mod_date = saved_mod_date.replace(microsecond=0)
            file_mod_date = datetime.datetime.strptime(ctime(path.getmtime(file_name)), "%a %b %d %H:%M:%S %Y")
            file_mod_date = file_mod_date.replace(microsecond=0)
            return saved_mod_date != file_mod_date

    def add_entry(self, section, key_value):
        if type(section) == str and type(key_value) == dict:
            if section not in self.parser:
                self.parser[section] = {}
            for key in key_value.keys():
                self.parser[section][key] = str(key_value[key])


if __name__ == '__main__':
    manager = ConfigManager("test.ini", "/home/max/Desktop/BA-Repo/framework/testConfig.xml")
    manager.load()
    print(global_config.is_dirty_config())
    manager.save()
