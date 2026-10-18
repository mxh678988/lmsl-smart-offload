"""
lmsl-smart-offload -- Smart Resident Scheduler for ComfyUI low VRAM.

Optimizes model swap-in/swap-out scheduling under --lowvram /
--enable-dynamic-vram by pre-scanning the prompt graph and injecting a
Belady-like (future-aware) residency policy at the unload decision point.
"""

import logging
import threading

logger = logging.getLogger("lmsl-smart-offload")

from . import smart_offload

NODE_CLASS_MAPPINGS = {
    "SmartResidentScheduler": smart_offload.SmartResidentScheduler,
}
NODE_DISPLAY_NAME_MAPPINGS = {
    "SmartResidentScheduler": "Smart Resident Scheduler (LMSL)",
}

__all__ = ["NODE_CLASS_MAPPINGS", "NODE_DISPLAY_NAME_MAPPINGS"]


def _late_register(attempts: int = 20, delay: float = 0.5) -> None:
    """ComfyUI imports custom_nodes after PromptServer() is constructed, but
    for robustness retry registration on a timer if the instance is not ready
    yet.  Fails soft -- stock scheduling is preserved either way."""
    if smart_offload.register_prompt_handler():
        return
    if attempts <= 0:
        logger.warning("[lmsl-smart-offload] prompt-handler registration abandoned (server not ready)")
        return
    t = threading.Timer(delay, _late_register, args=(attempts - 1, delay))
    t.daemon = True
    t.start()


def _init() -> None:
    try:
        smart_offload.install()
    except Exception:
        # Do not break ComfyUI startup if patching failed.
        logger.error("[lmsl-smart-offload] initialization failed; stock scheduling remains active", exc_info=True)
        return
    _late_register()


_init()
