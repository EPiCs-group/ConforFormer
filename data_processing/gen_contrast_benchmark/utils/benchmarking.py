import time
from functools import wraps
import contextlib
import numpy as np
import hashlib

def timer(func):
    @wraps(func)
    def wrapper(*args, **kwargs):
        start = time.perf_counter()
        result = func(*args, **kwargs)
        end = time.perf_counter()
        duration = end - start

        print(f"Function '{func.__name__}' executed in {duration:.4f} seconds")
        return result

    return wrapper


def string_to_seed(s):
    return int(hashlib.sha256(s.encode("utf-8")).hexdigest(), 16) % int(1e6)


@contextlib.contextmanager
def numpy_seed(seed, *addl_seeds):
    """Context manager which seeds the NumPy PRNG with the specified seed and
    restores the state afterward"""
    if seed is None:
        yield
        return
    if len(addl_seeds) > 0:
        seed = int(hash((seed, *addl_seeds)) % 1e6)
    state = np.random.get_state()
    np.random.seed(seed)
    try:
        yield
    finally:
        np.random.set_state(state)
