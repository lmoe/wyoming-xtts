from importlib.metadata import PackageNotFoundError, version

try:
    __version__ = version("wyoming-xtts")
except PackageNotFoundError:
    __version__ = "0.0.0.dev0"

SERVICE_NAME = "wyoming-xtts"
