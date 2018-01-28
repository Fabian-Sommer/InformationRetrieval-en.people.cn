#!/usr/bin/env python3

import time
from functools import wraps
from contextlib import contextmanager


class Report():
    def __init__(self, quiet_mode=False, log_file_path=''):
        self.quiet_mode = quiet_mode
        self.log_file_path = log_file_path
        self.time_measures = {}

    def report(self, *args, **kwargs):
        if not self.quiet_mode:
            print(*args, **kwargs)

    def all_time_measures(self):
        if len(self.time_measures.keys()) == 0:
            return

        max_length = max((len(task) for task in self.time_measures.keys()))
        total_time = sum(self.time_measures.values())
        total_row = \
            f'{"total_time":{max_length + 5}} | {total_time:>8.2f} sec'
        '   |   100.00%\n'
        self.report(f'\n{"task":{max_length + 5}} |    duration    |'
                    '   % duration')
        self.report((len(total_row) + 5) * '-')
        for task, duration in self.time_measures.items():
            self.report(f'{task:{max_length + 5}} | {duration:>8.2f} sec   |'
                        f'   {duration/total_time:7.2%}')
        self.report((len(total_row) + 5) * '-')
        self.report(total_row)

        if self.log_file_path != '':
            with open(self.log_file_path, 'a') as log_file:
                table_header = ''
                table_row = ''
                for task, duration in self.time_measures.items():
                    table_header += f'{task},'
                    table_row += f'{duration:.2f},'
                table_header += 'total'
                table_row += f'{total_time:.2f}'

                self.report(table_header, file=log_file)
                self.report(table_row, file=log_file)

    def progress(self, progress, message, interval=10000):
        if progress % interval == 0:
            self.report(f'{progress}{message}')

    def measure_deco(self, measured_function, *args, **kwargs):
        @wraps(measured_function)
        def measure_wrapper(*args, **kwargs):
            print(f'{measured_function.__name__}...')
            t_begin = time.time()
            result = measured_function(*args, **kwargs)
            duration = time.time() - t_begin
            self.time_measures[task_name] = duration
            self.report(f'finished {measured_function.__name__} in '
                        '{duration:.2f} sec')
            return result
        return measure_wrapper

    @contextmanager
    def measure(self, task_name):
        self.report(f'{task_name}...')
        t_begin = time.time()
        yield
        duration = time.time() - t_begin
        self.time_measures[task_name] = duration
        self.report(f'finished {task_name} in {duration:.2f} sec')


if __name__ == '__main__':
    report = Report()
    with report.measure('heavy work'):
        t_begin = time.time()
        while time.time() - t_begin < 0.2:
            pass
    report.all_time_measures()
