"""
Lazy static utility classo

Allows for re-use of functions and objects


Examples
```

dataset_fn = lazy_static("dataset", expensive_dataloading_fn)

# Repeated calls don't cost anything extra
dataset = dataset_fn()
dataset = dataset_fn()
dataset = dataset_fn()
dataset = dataset_fn()
```
"""

from functools import partial


class LazyStatic:
    """
    Lazy static that is globally available when this module is imported
    """
    data: dict[str, object]

    def __init__(self):
        self.data = {}

    def __setitem__(self, key, value):
        self.data[key] = value


__lazy_static = LazyStatic()


def lazy_static(key, func, *args, **kwargs):
    """
    Lazily evaluate a function but keep it's returned data is cached so that future calls don't call the function.
    :param key: Key to save
    :type key:
    :param func: The funtion to call
    :type func:
    :param args: positonal Arguments for the function
    :type args:
    :param kwargs: Keyword arguments for the function
    :type kwargs:
    :return:
    :rtype:
    """

    def inner(key, func, *args, **kwargs):
        data = None

        if key not in __lazy_static.data:
            data = func(*args, **kwargs)
            __lazy_static[key] = data

        return data

    return partial(inner, key, func, *args, **kwargs)
