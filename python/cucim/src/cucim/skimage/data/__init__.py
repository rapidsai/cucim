from .._shared import lazy

__getattr__, __dir__, __all__ = lazy.attach(
    __name__,
    submodules={},
    submod_attrs={
        '_binary_blobs': ['binary_blobs'],
    }
)
