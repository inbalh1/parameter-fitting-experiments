# File with utils for running the experiments files
import run
import multiprocessing



def setup(is_colab=False):
    # Try using more than 1 core
    run.use_cores(max(1, multiprocessing.cpu_count() - 2))

    # For google colab - to import pygirgs.

    #if is_colab:
    #    import os
    #    os.environ["PYTHONPATH"] = os.path.abspath("src/pygirgs")

