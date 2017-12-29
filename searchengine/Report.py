import time

quiet_mode = False
def set_quiet_mode(boolean):
    global quiet_mode
    quiet_mode = boolean

def report(message = ''):
    if not quiet_mode:
        print(message)

time_measures = {}
def report_begin(task):
    time_measures[task] = time.process_time()
    report(f'\n{task}...')

def report_finish(task):
    if not task in time_measures:
        report(f'WARNING: report_begin("{task}") has to be called before report_finish("{task}")')
        return

    time_measures[task] = time.process_time() - time_measures[task]
    report(f'finished {task} after {time_measures[task]:.2f} seconds')

def report_all_time_measures():
    max_length = max([ len(task) for task in time_measures.keys() ])
    total_time = sum(time_measures.values())
    total_row = f'{"total_time":{max_length + 5}} | {total_time:>8.2f} sec   |   100.00%'
    report()
    report(f'{"task":{max_length + 5}} |    duration    |   % duration')
    report((len(total_row) + 5) * '-')
    for task, duration in time_measures.items():
        report(f'{task:{max_length + 5}} | {duration:>8.2f} sec   |   {duration/total_time:7.2%}')
    report((len(total_row) + 5) * '-')
    report(total_row)

def report_progress(progress, message, report_interval = 10000):
    if progress % report_interval == 0:
        report(f'{progress}{message}')
