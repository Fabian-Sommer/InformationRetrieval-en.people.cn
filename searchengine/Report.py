import time

quiet_mode = False
def set_quiet_mode(boolean):
    global quiet_mode
    quiet_mode = boolean

def report(message = ''):
    if not quiet_mode:
        print(message)

time_measures = {}
def report_time(message):
    if not message in time_measures:
        time_measures[message] = time.process_time()
        report(f'\n{message}...')
    elif isinstance(time_measures[message], str):
        report(f'WARNING: report_time was called more than 2 times with argument "{message}"')
        time_measures[message] = time.process_time()
        report(f'{message}...')
    else:
        duration = str(time.process_time() - time_measures[message])
        report(f'finished {message} after {duration} seconds')
        time_measures[message] = duration

def report_progress(progress, message, report_interval = 1000):
    if progress % report_interval == 0:
        report(f'{progress}{message}')
