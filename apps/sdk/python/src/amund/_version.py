"""Single source of truth for the SDK version.

The version is declared exactly once, in `pyproject.toml`, and read back here
from the installed package metadata. Before this module existed the string was
copy-pasted into four places (`pyproject.toml`, `__init__.__version__`,
`Record.sdk_version`, and the transport's User-Agent header) — bumping the
package without bumping the copies shipped a release whose own telemetry
reported the *previous* version, silently, on every ingested span.
"""

from importlib.metadata import PackageNotFoundError, version

try:
    __version__ = version("amund")
except PackageNotFoundError:  # pragma: no cover - source tree with no install
    # Only reachable when the package is imported straight off disk with no
    # metadata present. Editable installs (`uv pip install -e`, what dogfooding
    # uses) do register metadata, so this is a last-resort marker rather than a
    # value anyone should see in real telemetry.
    __version__ = "0.0.0.dev0"
