import uuid
import os
import dill


class _Cache(object):
    def __init__(self, function, cache_dir="./cache/", compress=False, hash_self=False):
        self.cache_dir = cache_dir
        self.hash_self = hash_self
        self.compress = compress
        self.function = function

        if not os.path.exists(self.cache_dir):
            os.makedirs(self.cache_dir)

    def __call__(self, *args, **kwargs):
        args = list(args)
        if self.hash_self:
            args[0] = hash(args[0])

        key = str(args)+str(kwargs)
        key = str(uuid.uuid5(uuid.NAMESPACE_DNS, key))

        output_path = os.path.join(self.cache_dir, key)

        if os.path.exists(output_path):
            return dill.load(output_path)

        output = self.function(*args, **kwargs)
        dill.dump(output, open(output_path, "wb+"))

        return output


def Cache(function=None, cache_dir=None, compress=False, hash_self=False):
    if function:
        return _Cache(function)
    else:
        def wrapper(function):
            return _Cache(function, cache_dir, compress, hash_self)
        return wrapper
