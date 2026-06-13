try:
    from ember._version import __version__
except ImportError:
    __version__ = "unknown"

__all__ = [
    "__version__",
    "EmberData",
    "EmberModel",
    "EmberRunner",
    "EmberTracker",
]


def __getattr__(name: str):
    if name == "EmberData":
        from ember.data import EmberData

        globals()[name] = EmberData
        return EmberData
    if name == "EmberModel":
        from ember.models import EmberModel

        globals()[name] = EmberModel
        return EmberModel
    if name == "EmberRunner":
        from ember.run import EmberRunner

        globals()[name] = EmberRunner
        return EmberRunner
    if name == "EmberTracker":
        from ember.tracking import EmberTracker

        globals()[name] = EmberTracker
        return EmberTracker
    raise AttributeError(f"module 'ember' has no attribute {name!r}")
