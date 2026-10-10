from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from cellseg_models_pytorch.utils import FileHandler, H5Handler

__all__ = ["FileHandler", "H5Handler"]


def __getattr__(name: str) -> Any:
    """Load the upstream file handlers only when explicitly requested."""
    if name in __all__:
        from cellseg_models_pytorch.utils import FileHandler, H5Handler

        return {"FileHandler": FileHandler, "H5Handler": H5Handler}[name]
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


def __dir__() -> list[str]:
    return sorted(set(globals()) | set(__all__))
