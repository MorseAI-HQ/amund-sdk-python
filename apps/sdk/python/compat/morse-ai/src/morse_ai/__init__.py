"""Compatibility shim: ``morse_ai`` is now ``amund``.

Morse is now Amund. The SDK ships as the ``amund`` distribution and is
imported as ``import amund``. This package exists so code written against
``morse-ai`` / ``import morse_ai`` keeps working unchanged:

* ``import morse_ai`` returns the ``amund`` module itself, so
  ``morse_ai is amund`` and every function, class and singleton is shared.
* ``import morse_ai.<sub>`` / ``from morse_ai.<sub> import X`` resolve to the
  very same module objects as ``amund.<sub>`` (an import hook maps the names),
  so ``isinstance`` checks, ``mock.patch("morse_ai...")`` targets and
  module-level state behave exactly as before.
* ``MORSE_*`` environment variables are still read (``AMUND_*`` wins when both
  are set). That fallback lives in ``amund`` itself, not here.

Importing this package emits one ``DeprecationWarning``. To migrate::

    pip uninstall morse-ai && pip install amund
    # then: import morse_ai  ->  import amund
"""

from __future__ import annotations

import importlib
import importlib.abc
import importlib.machinery
import importlib.util
import sys
import warnings
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from types import ModuleType

_OLD = "morse_ai"
_NEW = "amund"

warnings.warn(
    "The 'morse-ai' package has been renamed to 'amund'. "
    "Install it with `pip install amund` and replace `import morse_ai` with "
    "`import amund`. The morse_ai name keeps working for now as an alias of amund.",
    DeprecationWarning,
    stacklevel=2,
)


class _AliasFinder(importlib.abc.MetaPathFinder, importlib.abc.Loader):
    """Resolve ``morse_ai.<sub>`` to the already-loaded ``amund.<sub>`` module.

    It must sit ahead of the normal path finder: once ``sys.modules['morse_ai']``
    is the ``amund`` package, the path finder would otherwise load a second,
    independent copy of ``amund/<sub>.py`` under the ``morse_ai.<sub>`` name.
    """

    def __init__(self) -> None:
        self._real_specs: dict[str, importlib.machinery.ModuleSpec | None] = {}

    def find_spec(
        self,
        fullname: str,
        path: object = None,
        target: ModuleType | None = None,
    ) -> importlib.machinery.ModuleSpec | None:
        if not fullname.startswith(_OLD + "."):
            return None
        real_name = _NEW + fullname[len(_OLD) :]
        try:
            real_spec = importlib.util.find_spec(real_name)
        except (ImportError, ValueError):
            return None
        if real_spec is None:
            return None
        return importlib.util.spec_from_loader(
            fullname,
            self,
            is_package=real_spec.submodule_search_locations is not None,
        )

    def create_module(self, spec: importlib.machinery.ModuleSpec) -> ModuleType:
        real = importlib.import_module(_NEW + spec.name[len(_OLD) :])
        # The import system stamps ``__spec__`` onto whatever create_module
        # returns; remember the real one so exec_module can put it back.
        self._real_specs[spec.name] = getattr(real, "__spec__", None)
        return real

    def exec_module(self, module: ModuleType) -> None:
        alias = module.__spec__.name if module.__spec__ is not None else None
        if alias is not None and alias in self._real_specs:
            module.__spec__ = self._real_specs.pop(alias)


def _install() -> ModuleType:
    if not any(isinstance(f, _AliasFinder) for f in sys.meta_path):
        sys.meta_path.insert(0, _AliasFinder())
    amund = importlib.import_module(_NEW)
    # Alias every amund submodule that is already loaded; the finder covers
    # the ones imported later (adapters are loaded lazily).
    prefix = _NEW + "."
    for name, mod in list(sys.modules.items()):
        if name.startswith(prefix) and mod is not None:
            sys.modules.setdefault(_OLD + name[len(_NEW) :], mod)
    return amund


# Replacing our own sys.modules entry makes ``import morse_ai`` bind to the
# amund package object itself (the import system re-reads sys.modules after
# executing a module).
sys.modules[__name__] = _install()
