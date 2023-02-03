"""
First and foremost. Check if we can import the library.
"""

import importlib
import os
import sys
from pathlib import Path

# Fix the import path for platform-agnostic testing
sys.path.append(os.path.dirname(os.path.realpath(__file__)) + "/../src")


def _import_helper(sub_module):
    os.chdir(os.path.dirname(os.path.realpath(__file__)) + "/../src")
    imports = list(Path(f"debeir/{sub_module}/").rglob("*.py"))
    assert len(imports) > 0

    for im in imports:
        if "__init__.py" in im.name:
            continue

        module_path = ".".join(list(im.parts))[:-3]

        try:
            importlib.import_module(module_path)
        except ModuleNotFoundError as e:
            raise Exception(f"Unable to import {module_path}", str(e))


def test_datasets_imports():
    _import_helper("datasets")


def test_engines_imports():
    _import_helper("engines")


def test_evaluation_imports():
    _import_helper("evaluation")


def test_interfaces_imports():
    _import_helper("core")


def test_models_imports():
    _import_helper("models")


def test_rankers_imports():
    _import_helper("rankers")


def test_training_import():
    _import_helper("training")


def test_utils_import():
    _import_helper("utils")


def test_bootstrap_import():
    sys.path.insert(0, os.path.dirname("../main.py"))


def test_toplevel_imports():
    pass
