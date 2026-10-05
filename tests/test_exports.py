import ast
import importlib
import inspect
import pkgutil

import pytest

import rf_tools

# public-looking top-level names that are intentionally not exported
NOT_EXPORTED = {'logger'}

SUBMODULES = [
    importlib.import_module(f'rf_tools.{info.name}')
    for info in pkgutil.iter_modules(rf_tools.__path__)
    if not info.name.startswith('_')
]


def _public_top_level_names(module) -> set[str]:
    """Names of public functions, classes and variables defined at module level."""
    names: set[str] = set()
    for node in ast.parse(inspect.getsource(module)).body:
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            names.add(node.name)
        elif isinstance(node, ast.Assign):
            names.update(t.id for t in node.targets if isinstance(t, ast.Name))
        elif isinstance(node, ast.AnnAssign) and isinstance(node.target, ast.Name):
            names.add(node.target.id)
    return {n for n in names if not n.startswith('_')} - NOT_EXPORTED


def test_package_all_names_exist():
    missing = [name for name in rf_tools.__all__ if not hasattr(rf_tools, name)]
    assert not missing, f'Names in `rf_tools.__all__` that do not exist: {missing}'


def test_package_all_has_no_duplicates():
    duplicates = {name for name in rf_tools.__all__ if rf_tools.__all__.count(name) > 1}
    assert not duplicates, f'Duplicate names in `rf_tools.__all__`: {duplicates}'


@pytest.mark.parametrize('module', SUBMODULES, ids=lambda m: m.__name__)
def test_submodule_defines_all(module):
    assert hasattr(module, '__all__'), f'`{module.__name__}` has no `__all__`'


@pytest.mark.parametrize('module', SUBMODULES, ids=lambda m: m.__name__)
def test_submodule_all_matches_public_names(module):
    exported = set(module.__all__)
    defined = _public_top_level_names(module)
    assert not exported - defined, f'Listed in `{module.__name__}.__all__` but not defined there: {exported - defined}'
    assert not defined - exported, f'Public in `{module.__name__}` but missing from its `__all__`: {defined - exported}'


@pytest.mark.parametrize('module', SUBMODULES, ids=lambda m: m.__name__)
def test_submodule_exported_by_package(module):
    missing = set(module.__all__) - set(rf_tools.__all__)
    assert not missing, f'Names from `{module.__name__}` not re-exported by `rf_tools`: {missing}'
