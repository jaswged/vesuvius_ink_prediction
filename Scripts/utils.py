import psutil
from time import time


class TimerError(Exception):
    pass


class Timer:
    """A utility class that, when used as a context manager,
    will report the time spent on code inside its block.
    Created by [https://www.kaggle.com/brettolsen]
    """

    def __init__(self, text=None):
        if text is not None:
            self.text = text + ": {:0.4f} seconds"
        else:
            self.text = "Elapsed time: {:0.4f} seconds"

        def log_func(x):
            print(x)

        self.logger = log_func
        self._start_time = None

    def start(self):
        if self._start_time is not None:
            raise TimerError("Timer is already running.  Use .stop() to stop it.")
        self._start_time = time()

    def stop(self):
        if self._start_time is None:
            raise TimerError("Timer is not running.  Use .start() to start it.")
        elapsed_time = time() - self._start_time
        self._start_time = None

        if self.logger is not None:
            self.logger(self.text.format(elapsed_time))

        return elapsed_time

    def __enter__(self):
        self.start()
        return self

    def __exit__(self, exc_type, exc_value, exc_traceback):
        self.stop()


def show_mem_use():
    process = psutil.Process()
    mb_mem = process.memory_info().rss / 1e6
    print(f"{mb_mem:6.2f} MB used")


if __name__ == "__main__":
    print("Main called")
