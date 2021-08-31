import uuid
import os
import dill

CACHE_DIR = "./cache/"


class _Cache(object):
    def __init__(self, function, cache_dir=CACHE_DIR):
        self.cache_dir = CACHE_DIR
        self.function = function

        if not os.path.exists(CACHE_DIR):
            os.makedirs(CACHE_DIR)

    def __call__(self, *args, **kwargs):
        key = str(args)+str(kwargs)
        key = str(uuid.uuid5(uuid.NAMESPACE_DNS, key))

        output_path = os.path.join(self.cache_dir, key)

        if os.path.exists(output_path):
            return dill.load(output_path)

        output = self.function(*args, **kwargs)
        dill.dump(output, open(output_path, "wb+"))

        return output


def Cache(function=None, cache_dir=None):
    if function:
        return _Cache(function)
    else:
        def wrapper(function):
            return _Cache(function, cache_dir)
        return wrapper
