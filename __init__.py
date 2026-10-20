"""ComfyUI custom-node package entry point.

ComfyUI imports every directory under ``custom_nodes/`` as a package and
requires an ``__init__.py`` at that directory root.  The implementation
lives in the ``lmsl_smart_offload`` sub-package so that a pip/registry
install (which lands as ``custom_nodes/lmsl_smart_offload``) and a plain
``git clone`` (which lands as ``custom_nodes/lmsl-smart-offload``) both
load through the exact same code path.
"""

from .lmsl_smart_offload import (
    NODE_CLASS_MAPPINGS,
    NODE_DISPLAY_NAME_MAPPINGS,
    __version__,
)

__all__ = ["NODE_CLASS_MAPPINGS", "NODE_DISPLAY_NAME_MAPPINGS", "__version__"]
