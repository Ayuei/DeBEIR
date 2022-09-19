"""
First and foremost. Check if we can import the library.
"""

import pytest
import importlib
from pathlib import Path


def _import_helper(sub_module):
    imports = list(Path(f"../nir/{sub_module}").rglob("*.py"))
    for im in imports:
        if "__init__.py" in im.name:
            continue

        module_path = ".".join(list(im.parts)[1:])[:-3]

        try:
            importlib.import_module(module_path)
        except ModuleNotFoundError as e:
            raise Exception(f"Unable to import {im.name}", str(e))


def test_datasets_imports():
    _import_helper("data_sets")


def test_engines_imports():
    _import_helper("engines")


def test_evaluation_imports():
    _import_helper("evaluation")


def test_interfaces_imports():
    _import_helper("interfaces")


def test_models_imports():
    _import_helper("models")


def test_rankers_imports():
    _import_helper("rankers")


def test_training_import():
    _import_helper("training")


def test_utils_import():
    _import_helper("utils")


def test_bootstrap_import():
    import sys, os

    sys.path.insert(0, os.path.dirname("../main.py"))

    import nir
    import nir.data_sets
    import nir.rankers
    import nir.evaluation
    import nir.interfaces
    import nir.models
    import nir.rankers
    import nir.training
    import nir.utils

