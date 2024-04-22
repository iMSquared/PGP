import time

from debug.debug import debug_data


def debug_profile(debug_entry: str = None):
    """NOTE(ssh): Some custom decorator for profiling."""
    def decorator(function):

        if debug_entry == None:
            debug_entry = function.__name__

        def wrapper(*args, **kwargs):
            # Profiler start
            print(f"{function.__name__} start")
            check_time = time.time()
            # Some function...
            result = function(*args, **kwargs)
            # Profiler ends
            print(f"{function.__name__} end")
            check_time = time.time() - check_time
            debug_data[debug_entry].append(check_time)
            return result
        return wrapper
    return decorator