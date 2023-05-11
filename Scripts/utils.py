import psutil
from time import time
import matplotlib.pyplot as plt
from torch import Tensor


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


def save_predictions_image(ink_pred: Tensor, inklabels, file_name: str) -> None:
    fig, axs = plt.subplots(nrows=2, ncols=3, figsize=(30, 30))
    axs.flatten()
    if inklabels:
        axs[0][0].imshow(inklabels, cmap="gray")
        axs[0][0].set_title("Labels")
    else:
        axs[0][1].imshow(ink_pred >= 0.35, cmap="gray")
        axs[0][1]. set_title("@ .35")

    # Show the output images at different thresholds
    axs[0][1].imshow(ink_pred >= 0.4, cmap="gray")
    axs[0][1]. set_title("@ .4")

    axs[0][2].imshow(ink_pred >= 0.5, cmap="gray")
    axs[0][2].set_title("@ .5")

    axs[1][0].imshow(ink_pred >= 0.6, cmap="gray")
    axs[1][0].set_title("@ .6")

    axs[1][1].imshow(ink_pred >= 0.7, cmap="gray")
    axs[1][1].set_title("@ .7")

    axs[1][2].imshow(ink_pred >= 0.8, cmap="gray")
    axs[1][2].set_title("@ .8")

    [axi.set_axis_off() for axi in axs.ravel()]  # Turn off the axes on all the sub plots
    plt.savefig(file_name, transparent=False)
    print("Graph has saved")


def dice_coef_torch(preds: Tensor, targets: Tensor, beta=0.5, smooth=1e-5) -> float:
    """
    https://www.kaggle.com/competitions/vesuvius-challenge-ink-detection/discussion/397288
    """
    # comment out if your model contains a sigmoid or equivalent activation layer. Ie You have already sigmoid(ed)
    # preds = torch.sigmoid(preds)

    # .41 notebook didn't have these .views
    # flatten label and prediction tensors
    preds = preds.view(-1).float()
    targets = targets.view(-1).float()

    y_true_count = targets.sum()
    ctp = preds[targets == 1].sum()
    cfp = preds[targets == 0].sum()
    beta_squared = beta * beta

    c_precision = ctp / (ctp + cfp + smooth)
    c_recall = ctp / (y_true_count + smooth)
    dice = (1 + beta_squared) * (c_precision * c_recall) / (beta_squared * c_precision + c_recall + smooth)
    return dice


if __name__ == "__main__":
    print("Main called")
