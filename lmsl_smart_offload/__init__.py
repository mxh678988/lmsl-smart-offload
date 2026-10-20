"""
lmsl-smart-offload -- Smart Resident Scheduler for ComfyUI low VRAM.

Optimizes model swap-in/swap-out scheduling under --lowvram /
--enable-dynamic-vram by pre-scanning the prompt graph and injecting a
Belady-like (future-aware) residency policy at the unload decision point.
"""

import logging
import threading

logger = logging.getLogger("lmsl-smart-offload")

from . import routes, smart_offload
from .version import __version__

NODE_CLASS_MAPPINGS = {
    "SmartResidentScheduler": smart_offload.SmartResidentScheduler,
    "SmartModelAdvisor": smart_offload.SmartModelAdvisor,
    "SmartModelRecommender": smart_offload.SmartModelRecommender,
}
NODE_DISPLAY_NAME_MAPPINGS = {
    "SmartResidentScheduler": "Smart Resident Scheduler (LMSL)",
    "SmartModelAdvisor": "Smart Model Advisor (LMSL / TDS SDK)",
    "SmartModelRecommender": "Smart Model Recommender (LMSL / TDS SDK)",
}

__all__ = ["NODE_CLASS_MAPPINGS", "NODE_DISPLAY_NAME_MAPPINGS", "__version__"]

_ROUTES_DONE = False


def _ensure_routes() -> None:
    """只读状态路由注册一次即可（重复注册会被 aiohttp 拒绝）。"""
    global _ROUTES_DONE
    if _ROUTES_DONE:
        return
    _ROUTES_DONE = routes.register_routes()


def _late_register(attempts: int = 20, delay: float = 0.5) -> None:
    """ComfyUI imports custom_nodes after PromptServer() is constructed, but
    for robustness retry registration on a timer if the instance is not ready
    yet.  Fails soft -- stock scheduling is preserved either way."""
    if smart_offload.register_prompt_handler():
        _ensure_routes()
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

    # TDS SDK 选型体检：后台线程预热（不阻塞启动、失败即降级为纯调度插件）
    try:
        smart_offload.sdk_bridge.warmup()
    except Exception:
        logger.debug("[lmsl-smart-offload] sdk bridge warmup skipped", exc_info=True)

    _late_register()
    _ensure_routes()


_init()
