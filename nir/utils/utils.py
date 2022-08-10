import collections
import inspect

import loguru
import os
import sys
from collections.abc import MutableMapping


def create_output_file(config, config_fp, remove, output_file, output_directory, **kwargs):
    """
    Create output file based on config instructions

    :param config: The config object with output file options.
    :param config_fp: The config file path used in default naming options for the output file.
    :param remove: Overwrites the output file if it exists
    :param output_file: The output file path if it exists
    :param output_directory: The output directory used for default naming (specified in nir config)
    :param kwargs: Compatibility arguments
    :return:
    """
    if output_file is None:
        os.makedirs(name=f"{output_directory}/{config.index}", exist_ok=True)
        output_file = (
            f"{output_directory}/{config.index}/{config_fp.split('/')[-1].replace('.toml', '')}"
        )
        loguru.logger.info(f"Output file not specified, writing to: {output_file}")

    if os.path.exists(output_file) and not remove:
        loguru.logger.info(f"Output file exists: {output_file}. Exiting...")
        sys.exit(0)

    if remove:
        loguru.logger.info(f"Output file exists: {output_file}. Overwriting...")
        open(output_file, "w+").close()

    assert (
            config.query_type
    ), "At least config or argument must be provided for query type"

    return output_file


async def unpack_coroutine(f):
    """
    Recursively unwraps co-routines until a result is reached.

    :param f: Wrapped co-routine function.
    :return:
        Results from the (final) evaluated co-routine.
    """
    res = await f
    while inspect.isawaitable(res):
        res = await res

    return res


def flatten(d, parent_key="", sep="_"):
    """

    Flattens a multidimensional dictionary (dictionary of dictionaries) to a single layer with child keys seperated by
    "sep"

    :param d: Multi-level dictionary to flatten.
    :param parent_key: Prepend a parent_key to all layers.
    :param sep: Seperator token between child and parent layers.
    :return:
        A flattened 1-D dictionary with keys seperated by *sep*.
    """
    items = []
    for k, v in d.items():
        new_key = parent_key + sep + k if parent_key else k
        if isinstance(v, MutableMapping):
            items.extend(flatten(v, new_key, sep=sep).items())
        else:
            items.append((new_key, None))
    return dict(items)
