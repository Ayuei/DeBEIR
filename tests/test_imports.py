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

        importlib.import_module(module_path)


def test_datasets_imports():
    _import_helper("data_sets")


def test_engines_imports():
    _import_helper("engines")


def test_interfaces_imports():
    _import_helper("interfaces")


def test_models_imports():
    _import_helper("models")


def test_evaluation_imports():
    _import_helper("evaluation")


def test_rankers_imports():
    _import_helper("rankers")


def test_utils_import():
    _import_helper("utils")
