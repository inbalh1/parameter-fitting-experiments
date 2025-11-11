# File with utils for running the experiments files
import run
import multiprocessing



def setup(is_colab=False):
    # Try using more than 1 core
    run.use_cores(max(1, multiprocessing.cpu_count() - 2))

    # For google colab
    if is_colab:
        import importlib.util
        so_path = 'src/pygirgs/pygirgs.cpython-311-x86_64-linux-gnu.so'
        spec = importlib.util.spec_from_file_location("pygirgs", so_path)
        pygirgs = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(pygirgs)
        

