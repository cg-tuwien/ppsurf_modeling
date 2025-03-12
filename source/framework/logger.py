from logging import getLogger, FileHandler, Formatter
from sys import stderr

from source.framework import global_config
import sqlite3
from os import path
from multiprocessing import Lock


class LoggingManager:
    initialised = False
    default_formatter = None
    db_conn = None
    lock = Lock()
    current_run_id = 0

    @classmethod
    def init(cls):
        if not cls.initialised:
            cls.default_formatter = Formatter(fmt="[%(asctime)s] %(levelname)s: %(message)s",
                                              datefmt="%d.%m.%Y %H:%M:%S")

            error = getLogger("error")
            fh = FileHandler(path.join(global_config.get_log_folder(), "error.log"))
            fh.setFormatter(cls.default_formatter)
            error.addHandler(fh)
            error.setLevel("WARNING")

            cls.db_conn = sqlite3.connect(path.join(global_config.get_log_folder(), "log.db"), check_same_thread=False)

            cursor = cls.db_conn.cursor()

            cursor.execute('''CREATE TABLE IF NOT EXISTS "Edge" (
                              "ID" INTEGER PRIMARY KEY AUTOINCREMENT,
                              "Run-ID" INTEGER NOT NULL,
                              "Edge" TEXT NOT NULL
                              )''')

            cursor.execute('''CREATE TABLE IF NOT EXISTS "Call" (
                              "ID" INTEGER PRIMARY KEY AUTOINCREMENT,
                              "Edge-ID" INTEGER NOT NULL,
                              "Timestamp" DATETIME DEFAULT CURRENT_TIMESTAMP,
                              "Status-Code"	TEXT NOT NULL,
                              "Message"	TEXT NOT NULL,
                              FOREIGN KEY("Edge-ID") REFERENCES "Edge"("ID")
                              )''')

            cursor.execute('''CREATE TABLE IF NOT EXISTS "File" (
                              "ID" INTEGER PRIMARY KEY AUTOINCREMENT,
                              "Call-ID" INTEGER NOT NULL,
                              "Type" TEXT NOT NULL,
                              "File" TEXT NOT NULL,
                              FOREIGN KEY("Call-ID") REFERENCES "Call"("ID")
                              )''')

            cls.initialised = True

    @classmethod
    def get_error_logger(cls):
        return getLogger("error")

    @classmethod
    def start_new_run(cls):
        cursor = cls.db_conn.cursor()
        cursor.execute('''SELECT max("Run-ID") from "Edge"''')
        max_id = cursor.fetchone()
        cls.current_run_id = 0 if max_id is None or max_id[0] is None else int(max_id[0]) + 1

    @classmethod
    def log_process_execution(cls, edge, input_files, output_files, status_code, message):
        def log_files(call_id, files, type):
            for directory in files.keys():
                for file in files[directory]:
                    cursor.execute('''INSERT INTO "File" ("Call-ID", "Type", "File")
                                  VALUES (?, ?, ?)''', (call_id, type, path.join(directory, file)))

        cls.lock.acquire()
        try:
            cursor = cls.db_conn.cursor()
            cursor.execute('''SELECT ID FROM "Edge" WHERE "Run-ID"=? AND "Edge"=?''', (cls.current_run_id, edge))
            res = cursor.fetchall()

            if len(res) == 0:
                cursor.execute('''INSERT INTO "Edge" ("Run-ID", "Edge")
                                  VALUES (?, ?)''', (cls.current_run_id, edge))
                cursor.execute('''SELECT last_insert_rowid()''')
                edge_id = int(cursor.fetchone()[0])
            elif len(res) == 1:
                edge_id = res[0][0]
            else:
                return

            cursor.execute('''INSERT INTO "Call" ("Edge-ID", "Status-Code", "Message") VALUES (?, ?, ?)''',
                           (edge_id, status_code, message))

            if output_files is not None and len(output_files) > 0 or input_files is not None and len(input_files) > 0:
                cursor.execute('''SELECT last_insert_rowid()''')
                call_id = int(cursor.fetchone()[0])
                if output_files is not None and len(output_files) > 0:
                    log_files(call_id, output_files, "OUTPUT")
                if input_files is not None and len(input_files) > 0:
                    log_files(call_id, input_files, "INPUT")

        except sqlite3.Error:
            print("An exception occurred! Please see log file for more information!", file=stderr)
            LoggingManager.get_error_logger().exception("SQLite Error in log_process_execution():")
        else:
            cls.db_conn.commit()
        finally:
            cls.lock.release()

    @classmethod
    def get_log_file(cls, file):
        if path.exists(file) and path.isfile(file):
            cls.lock.acquire()
            try:
                cursor = cls.db_conn.cursor()
                cursor.execute('''SELECT "Status-Code", "Message", "File" FROM "File"
                                      LEFT JOIN "Call"
                                      ON "Call"."ID" = "File"."Call-ID"
                                      WHERE "File" LIKE "%{}%"
                                      ORDER BY "Call"."Timestamp" DESC'''.format(file))
                result = cursor.fetchone()
            except sqlite3.Error:
                print("An exception occurred! Please see log file for more information!", file=stderr)
                LoggingManager.get_error_logger().exception("Exception in get_log_for_file():")
            else:
                if result is not None and len(result) == 3:
                    print("========== Output of file {0} ==========".format(result[2]))
                    print("Statuscode: {0}".format(result[0]))
                    print(result[1])
                    print("========== End of Output ==========")
            finally:
                cls.lock.release()
        else:
            print("The given file doesn't exist!")

    @classmethod
    def get_log_all_files(cls, file_name):
        """
        Writes the outputs of every created file of the last run into a log file
        :param file_name: File name of the log file which gets created
        """
        cls.lock.acquire()
        try:
            cursor = cls.db_conn.cursor()
            cursor.execute('''SELECT "Call"."ID", "Edge", "Timestamp", "Status-Code", "Message" FROM "Call" 
                              LEFT JOIN "Edge"
                              ON "Call"."Edge-ID" = "Edge"."ID"
                              WHERE "Edge-ID" in (
                              SELECT "ID" FROM "Edge" WHERE "Run-ID" = (
                              SELECT MAX("Run-ID") FROM "Edge"))''')
            result = cursor.fetchall()

            final_results = []

            for call in result:
                if len(call) == 5:
                    cursor.execute('''SELECT "Type", "File" FROM "File" WHERE "Call-ID"=?''', (int(call[0]),))
                    files = cursor.fetchall()
                    real_files = [f for f in files if path.exists(f[1]) and path.isfile(f[1])]
                    if len(real_files) > 0:
                        final_results.append((call, real_files))
        except sqlite3.Error:
            print("An exception occurred! Please see log file for more information!", file=stderr)
            LoggingManager.get_error_logger().exception("SQLite Error in get_log_all_files():")
        else:
            with open(file_name, "w") as log_file:
                for r in final_results:
                    input_files = [f[1] for f in r[1] if f[0] == "INPUT"]
                    output_files = [f[1] for f in r[1] if f[0] == "OUTPUT"]
                    log_file.write("Call for Edge \"{}\" at {}:\n\n".format(r[0][1], r[0][2]))
                    log_file.write("Status Code: {}\n".format(r[0][3]))
                    for (i, file) in enumerate(input_files):
                        log_file.write("Input files:\n{}".format(file) if i == 0 else "\n{}".format(file))
                    log_file.write("\n")
                    for (i, file) in enumerate(output_files):
                        log_file.write("Output files:\n{}".format(file) if i == 0 else "\n{}".format(file))
                    log_file.write("\n\n")
                    log_file.write("Output:\n{}".format(r[0][4]) if r[0][4] is not None and r[0][4] != "" else "")
                    log_file.write("="*200 + "\n")
                    log_file.write("="*200 + "\n")
        finally:
            cls.lock.release()

    @classmethod
    def get_log_all_edges(cls):
        """Prints an overview of the edges of the last run to the console"""
        cls.lock.acquire()
        result = {}
        try:
            cursor = cls.db_conn.cursor()
            cursor.execute('''SELECT * FROM "Edge" 
                              WHERE "Run-ID" = (
                              SELECT MAX("Run-ID") FROM "Edge")''')
            edges = cursor.fetchall()
            for edge in edges:
                cursor.execute('''SELECT * FROM "Call"
                                  WHERE "Edge-ID" = ?''', (edge[0],))
                calls = cursor.fetchall()
                result[edge[0]] = (edge[2], calls)
        except sqlite3.Error:
            print("An exception occurred! Please see log file for more information!", file=stderr)
            LoggingManager.get_error_logger().exception("Exception in get_log_all_edges():")
        else:
            print("Edge overview:")
            for edge in result.keys():
                sum_calls = len([c for c in result[edge][1]])
                sum_successful_calls = len([c for c in result[edge][1] if int(c[3]) == 0])
                print("{} - {} calls executed / {} calls successful".format(result[edge][0], sum_calls, sum_successful_calls))
        finally:
            cls.lock.release()

    @classmethod
    def get_log_which_calls(cls, file_name):
        """
        Prints the calls which lead to the given file to the console
        :param file_name: File for which the necessary calls are determined
        """
        def get_callstack(file):
            cursor.execute('''SELECT "Call"."ID", "Edge", "Timestamp" FROM "File"
                              LEFT JOIN "Call"
                              ON "File"."Call-ID" = "Call"."ID"
                              LEFT JOIN "Edge"
                              ON "Call"."Edge-ID" = "Edge"."ID"
                              WHERE "File" LIKE "%{0}%"
                              AND "Type" = "OUTPUT"
                              ORDER BY "Call"."Timestamp" DESC'''.format(file))
            call = cursor.fetchone()
            if call is None or len(call) == 0:
                return []
            else:
                cursor.execute('''SELECT "File" FROM "File" WHERE "Call-ID" = ? AND "Type"="INPUT"''', (call[0],))
                input_files = cursor.fetchall()
                calls = []

                for f in input_files:
                    calls += [c for c in get_callstack(f[0]) if c not in calls]

                return calls + [call]

        if path.exists(file_name) and path.isfile(file_name):
            cls.lock.acquire()
            try:
                cursor = cls.db_conn.cursor()
                res = get_callstack(file_name)
            except sqlite3.Error:
                print("An exception occurred! Please see log file for more information!", file=stderr)
                LoggingManager.get_error_logger().exception("Exception in which_calls():")
            else:
                print("Which calls lead to file {}:".format(file_name))
                for r in res:
                    print("Call with ID {} of edge {} at {}".format(r[0], r[1], r[2]))
            finally:
                cls.lock.release()
        else:
            print("The given file doesn't exist!")

    @classmethod
    def empty_db(cls):
        """Empties database"""
        cls.lock.acquire()
        result = {}
        try:
            cursor = cls.db_conn.cursor()
            cursor.execute('''DELETE FROM "Edge"''')
            cursor.execute('''DELETE FROM "Call"''')
            cursor.execute('''DELETE FROM "File"''')
            cursor.execute('''DELETE FROM "sqlite_sequence"''')
        except sqlite3.Error:
            print("An exception occurred! Please see log file for more information!", file=stderr)
            LoggingManager.get_error_logger().exception("Exception in get_log_all_edges():")
        else:
            cls.db_conn.commit()
        finally:
            cls.lock.release()
