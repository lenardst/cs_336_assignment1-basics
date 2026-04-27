try:
    from importlib.metadata import version

    __version__ = version("cs336_basics")
except Exception:
    pass
