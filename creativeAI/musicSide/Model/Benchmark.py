import time
from contextlib import contextmanager

class TimerError(Exception):
    """ handle exceptions """


class Benchmark:
    def __init__(self, _name):
        self.name = _name
        self.timer = time
        self.is_stopped = True
        self.start = 0
        self.end = 0
        self.elapsed = 0

    def __enter__(self):
        print('Enter call')


    def start_timer(self):
        if not self.is_stopped:
            self.reset_timer()

        self.is_stopped = False
        self.start = self.timer.perf_counter()
        print(f'[{self.name}] started.')

    def end_timer(self):
        if self.is_stopped:
            raise TimerError(f'[{self.name}] is stopped.  Use start_timer() to start it.')

        self.is_stopped = True
        self.end = self.timer.perf_counter()

        self.elapsed = self.end - self.start
        if self.elapsed > 60*60:
            self.elapsed = (self.elapsed / 60) / 60
            print(f'[{self.name}] stopped. Elapsed time: {self.elapsed:0.4f} hours')
        elif self.elapsed > 60:
            self.elapsed = self.elapsed / 60
            print(f'[{self.name}] stopped. Elapsed time: {self.elapsed:0.4f} minutes')
        else:
            print(f'[{self.name}] stopped. Elapsed time: {self.elapsed:0.4f} seconds')

    def reset_timer(self):
        self.timer = time
        self.is_stopped = True
        self.start = 0
        self.end = 0
        self.elapsed = 0
        print(f'[{self.name}] resetted.')