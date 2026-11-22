from importlib.metadata import PackageNotFoundError, version
from pathlib import Path

try:
    __version__ = version("radlab-data")
except PackageNotFoundError:  # running from a source checkout
    __version__ = (Path(__file__).resolve().parent.parent / ".version").read_text(
        encoding="utf-8"
    ).strip()
