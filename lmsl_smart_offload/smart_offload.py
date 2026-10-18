"""
Smart Resident Scheduler (lmsl-smart-offload) -- core implementation.

Purpose
-------
Mitigate the performance loss caused by model swap-in/swap-out thrashing in
ComfyUI low VRAM modes (--lowvram / --enable-dynamic-vram).

How it works (Belady-like approximate future-aware residency)
-------------------------------------------------------------
1.  PROMPT TIME (pre-scan):
    We register an `add_on_prompt_handler` on `server.PromptServer.instance`.
    When a prompt is submitted, before execution, we parse the execution graph
    (json_data["prompt"]), topologically sort it, and locate every "model
    source node" (a node whose RETURN_TYPES contains MODEL/CLIP/VAE and which
    has a file-combo input such as ckpt_name / unet_name / clip_name /
    vae_name / lora_name ...).  Each source node is assigned a topological
    rank and a stable model key ``f"{class_type}|{file_name}"``; we then know,
    for every model, the full sequence of positions at which it will be used
    in this prompt ("use positions").

2.  UNLOAD TIME (Belady-ish decision injection):
    We monkey-patch `comfy.model_management.free_memory` (the single unload
    decision point, called by load_models_gpu when VRAM is tight).  Before
    delegating to the original implementation, we merge into its `keep_loaded`
    list every currently loaded model whose *next use distance* (computed from
    the pre-scanned plan and the current execution cursor) is
    <= ``keep_steps``.  This approximates Belady's optimal algorithm: evict
    the model that is needed furthest in the future, keep the one that will be
    needed again soon.
    A VRAM pressure guard disables this protection when free VRAM is already
    below the minimum inference memory, so we never risk introducing OOM.

3.  PREFETCH (optional, default OFF):
    If enabled, after a load event we look ahead to the next planned model
    source; if its LoadedModel object is still resident (partial offload
    typical of lowvram) and there is enough spare VRAM
    (free > prefetch_min_free_mb + model offloaded size), we call
    ``LoadedModel.model_load()`` to warm it up during a GPU idle window.

Design constraints
------------------
- Conservative by default: keep_steps=1, pressure guard ON, prefetch OFF.
- Never touches ComfyUI source files (runtime wrapping only).
- Fails soft: any internal exception is logged and the original behavior is
  preserved.
"""

from __future__ import annotations

import heapq
import logging
import threading
from typing import Any, Dict, List, Optional

import nodes  # ComfyUI node registry (available at prompt-execution time)

# Imported lazily below because this module may be imported very early.
import comfy.model_management as mm

logger = logging.getLogger("lmsl-smart-offload")

# --------------------------------------------------------------------------
# Configuration (global defaults; can be overridden per prompt by placing a
# SmartResidentScheduler node in the workflow).
# --------------------------------------------------------------------------


class Config:
    """Runtime configuration. Conservative defaults."""

    def __init__(self) -> None:
        self.enabled: bool = True
        self.keep_steps: int = 1          # protect models whose next use is within this many topological steps
        self.prefetch_enabled: bool = False
        self.prefetch_min_free_mb: int = 2048
        self.verbose: bool = False
        self.pressure_guard: bool = True  # disable keep-protection when VRAM is tight

    def apply_widgets(self, widgets: Optional[list]) -> None:
        if not widgets or len(widgets) < 4:
            return
        try:
            self.enabled = bool(widgets[0])
            self.keep_steps = int(widgets[1])
            self.prefetch_enabled = bool(widgets[2])
            self.prefetch_min_free_mb = int(widgets[3])
            self.verbose = bool(widgets[4]) if len(widgets) > 4 else self.verbose
        except Exception:
            logger.warning("[smart-offload] failed to parse config widgets", exc_info=True)


config = Config()

# --------------------------------------------------------------------------
# Internal state
# --------------------------------------------------------------------------

_lock = threading.RLock()

_plan: Optional[Dict[str, Dict[str, Any]]] = None  # key -> {"positions": [...]}
_events: List[Dict[str, Any]] = []                  # topological events, each {"key","rank","node_id","class_type"}
_event_index: int = 0                               # next unconsumed event index
_current_rank: int = -1                             # rank of the most recent real load event
_key_of_loadedmodel: Dict[int, str] = {}            # id(LoadedModel) -> model key
_used_count: Dict[str, int] = {}                    # key -> number of use positions already consumed

_orig_free_memory = mm.free_memory
_orig_load_models_gpu = mm.load_models_gpu

_patched = False

MODEL_SOCKETS = ("MODEL", "CLIP", "VAE")
CONFIG_NODE_CLASS = "SmartResidentScheduler"


# --------------------------------------------------------------------------
# Graph helpers
# --------------------------------------------------------------------------

def _is_link(value: Any) -> bool:
    """ComfyUI prompt inputs: a link is [from_node_id, from_socket]."""
    return (
        isinstance(value, list)
        and len(value) == 2
        and isinstance(value[0], (str, int))
        and isinstance(value[1], (str, int))
    )


def _node_model_key(class_type: str, inputs: Dict[str, Any], class_def: Any) -> Optional[str]:
    """Return a stable model key for a node that provides a MODEL/CLIP/VAE.

    The key is ``f"{class_type}|{file_name}"`` so that the same underlying
    model file used by several nodes (or LoRA variants) collapses to one key
    and its full use sequence is tracked.
    """
    rt = getattr(class_def, "RETURN_TYPES", ()) or ()
    if not any(t in rt for t in MODEL_SOCKETS):
        return None
    try:
        it = class_def.INPUT_TYPES()
    except Exception:
        return None
    file_in: Optional[str] = None
    for category in ("required", "optional"):
        for iname, ispec in it.get(category, {}).items():
            if isinstance(ispec, (list, tuple)) and len(ispec) >= 1 and isinstance(ispec[0], list):
                low = iname.lower()
                if any(k in low for k in ("name", "file", "model", "clip", "vae", "lora", "ckpt", "path")):
                    file_in = iname
                    break
        if file_in:
            break
    if file_in is None:
        return f"{class_type}|?"
    val = inputs.get(file_in)
    if val is None or _is_link(val):
        return f"{class_type}|?"
    return f"{class_type}|{val}"


def _node_id_sort_key(nid: Any) -> tuple:
    """Total-order sort key for node ids.

    ComfyUI prompt keys are strings, but they may be numeric ("12") or opaque
    (UUIDs such as "62705a2c-b4b7-40da-94a0-f2bbd9d286d2"), and a single prompt
    can mix both.  Comparing ``int`` against ``str`` raises::

        TypeError: '<' not supported between instances of 'str' and 'int'

    so every id is normalised to one comparable tuple shape:
        numeric ids -> (0, int_value, "")   (ordered numerically)
        other ids   -> (1, 0, str(id))      (ordered lexicographically)
    """
    s = str(nid)
    if s.isdigit():
        return (0, int(s), "")
    return (1, 0, s)


def _build_plan(prompt: Dict[str, Any]) -> tuple:
    """Topologically sort the prompt graph and compute per-model use sequences.

    Returns (plan, events):
      plan:   {model_key: {"positions": [rank, ...]}}
      events: [{"key", "rank", "node_id", "class_type"}, ...] in execution order
    """
    node_ids = [
        nid for nid, node in prompt.items()
        if isinstance(node, dict) and isinstance(node.get("class_type"), str)
    ]
    node_ids = sorted(node_ids, key=_node_id_sort_key)
    present = set(node_ids)

    # dependencies: node -> set of upstream node ids it consumes via links
    deps: Dict[str, set] = {}
    for nid in node_ids:
        node = prompt[nid]
        up = set()
        for iname, val in node.get("inputs", {}).items():
            if _is_link(val):
                up.add(val[0])
        deps[nid] = {d for d in up if d in present}

    indeg = {nid: len(deps[nid]) for nid in node_ids}
    # The heap must never compare mixed types, so push (sort_key, node_id)
    # tuples: the normalised key does the ordering, the id only breaks exact
    # ties (identical keys are impossible for distinct ids).
    ready = [(_node_id_sort_key(nid), nid) for nid in node_ids if indeg[nid] == 0]
    heapq.heapify(ready)  # deterministic order
    order: List[str] = []
    while ready:
        _, nid = heapq.heappop(ready)
        order.append(nid)
        for nid2 in node_ids:
            if nid in deps[nid2]:
                indeg[nid2] -= 1
                if indeg[nid2] == 0:
                    heapq.heappush(ready, (_node_id_sort_key(nid2), nid2))

    plan: Dict[str, Dict[str, Any]] = {}
    events: List[Dict[str, Any]] = []
    for rank, nid in enumerate(order):
        node = prompt[nid]
        ct = node.get("class_type")
        class_def = nodes.NODE_CLASS_MAPPINGS.get(ct)
        if class_def is None:
            continue
        key = _node_model_key(ct, node.get("inputs", {}), class_def)
        if key is None:
            continue
        plan.setdefault(key, {"positions": []})["positions"].append(rank)
        events.append({"key": key, "rank": rank, "node_id": nid, "class_type": ct})
    return plan, events


def _next_use_distance(key: str) -> Optional[int]:
    """Distance (in topological steps) between the current position and the
    next unconsumed use of ``key``. ``None`` means 'no future use'."""
    if _plan is None:
        return None
    info = _plan.get(key)
    if not info:
        return None
    positions = info.get("positions", [])
    used = _used_count.get(key, 0)
    if used >= len(positions):
        return None
    if _current_rank < 0:
        return None
    return positions[used] - _current_rank


# --------------------------------------------------------------------------
# Prompt handler (pre-scan + per-prompt config)
# --------------------------------------------------------------------------

def _apply_node_config(prompt: Dict[str, Any]) -> None:
    """If the workflow contains a SmartResidentScheduler node, use its widget
    values for this prompt (falling back to global defaults otherwise)."""
    try:
        for node in prompt.values():
            if not isinstance(node, dict):
                continue
            if node.get("class_type") != CONFIG_NODE_CLASS:
                continue
            wv = node.get("widgets_values")
            if isinstance(wv, list):
                config.apply_widgets(wv)
                return
    except Exception:
        logger.warning("[smart-offload] failed to read config node", exc_info=True)


def on_prompt_handler(json_data: dict) -> dict:
    """Registered via PromptServer.instance.add_on_prompt_handler.

    Runs while the prompt is being submitted (before queueing), which is the
    earliest safe point to scan the execution graph.  We only read the graph
    and build the plan; we never modify json_data, so existing behavior is
    untouched.
    """
    try:
        prompt = json_data.get("prompt")
        if isinstance(prompt, dict) and config.enabled:
            _apply_node_config(prompt)
            plan, events = _build_plan(prompt)
            with _lock:
                global _plan, _events, _event_index, _current_rank, _used_count, _key_of_loadedmodel
                _plan = plan
                _events = events
                _event_index = 0
                _current_rank = -1
                _used_count = {}
                _key_of_loadedmodel.clear()
            if config.verbose:
                logger.info(
                    "[smart-offload] plan ready: %d model keys, %d events",
                    len(plan), len(events),
                )
    except Exception:
        logger.warning("[smart-offload] plan build failed (continuing with stock behavior)", exc_info=True)
    return json_data


# --------------------------------------------------------------------------
# Unload decision injection (Belady approximation)
# --------------------------------------------------------------------------

def _patched_free_memory(
    memory_required,
    device,
    keep_loaded: Optional[List] = None,
    for_dynamic: bool = False,
    pins_required: int = 0,
    ram_required: int = 0,
):
    """Wrap comfy.model_management.free_memory.

    Merge 'models that will be needed again soon' into keep_loaded, then call
    the original unloader so that ComfyUI's own ordering logic is preserved.
    """
    keep = list(keep_loaded or [])
    if config.enabled and config.keep_steps >= 0 and device is not None:
        pressure = False
        if config.pressure_guard:
            try:
                if mm.get_free_memory(device) < mm.minimum_inference_memory():
                    pressure = True
            except Exception:
                pressure = False
        if not pressure:
            with _lock:
                for lm in list(mm.current_loaded_models):
                    key = _key_of_loadedmodel.get(id(lm))
                    if key is None:
                        continue
                    dist = _next_use_distance(key)
                    if dist is not None and dist <= config.keep_steps:
                        keep.append(lm)
    unloaded: List = []
    try:
        unloaded = _orig_free_memory(
            memory_required,
            device,
            keep_loaded=keep,
            for_dynamic=for_dynamic,
            pins_required=pins_required,
            ram_required=ram_required,
        )
    finally:
        # keep the key table in sync: drop entries that have been unloaded
        with _lock:
            for lm in (unloaded or []):
                _key_of_loadedmodel.pop(id(lm), None)
    return unloaded


# --------------------------------------------------------------------------
# Load event tracking + optional prefetch
# --------------------------------------------------------------------------

def _consume_load(new_lms: List) -> None:
    """Associate newly loaded LoadedModel objects with the next planned event."""
    global _event_index, _current_rank
    with _lock:
        if _event_index < len(_events):
            ev = _events[_event_index]
            key = ev["key"]
            _event_index += 1
            _current_rank = ev["rank"]
            _used_count[key] = _used_count.get(key, 0) + 1
            for lm in new_lms:
                _key_of_loadedmodel[id(lm)] = key
            if config.verbose:
                logger.info(
                    "[smart-offload] load event rank=%d key=%s models=%d",
                    ev["rank"], key, len(new_lms),
                )
        else:
            for lm in new_lms:
                _key_of_loadedmodel[id(lm)] = "?"


def _maybe_prefetch() -> None:
    """Optional look-ahead: warm the next planned model during a GPU idle
    window.  Only acts when the LoadedModel is still resident (lowvram keeps
    it partially offloaded) and free VRAM comfortably exceeds the requirement.
    """
    with _lock:
        if _event_index >= len(_events):
            return
        key = _events[_event_index]["key"]
        target = None
        for lm in list(mm.current_loaded_models):
            if _key_of_loadedmodel.get(id(lm)) == key:
                target = lm
                break
        if target is None:
            return
        try:
            dev = target.device
            if dev is None or str(dev).startswith("cpu"):
                return
            offloaded = target.model_offloaded_memory()
            if offloaded <= 0:
                return
            free = mm.get_free_memory(dev)
            headroom = config.prefetch_min_free_mb * 1024 * 1024
            if free > headroom + offloaded:
                target.model_load()
                if config.verbose:
                    logger.info("[smart-offload] prefetch warmed %s", key)
        except Exception:
            logger.debug("[smart-offload] prefetch skipped", exc_info=True)


def _patched_load_models_gpu(
    models,
    memory_required: int = 0,
    force_patch_weights: bool = False,
    minimum_memory_required=None,
    force_full_load: bool = False,
):
    """Wrap comfy.model_management.load_models_gpu.

    Detects which LoadedModel objects were newly created by this call,
    advances the plan cursor accordingly, and optionally runs prefetch.
    """
    with _lock:
        before = {id(lm) for lm in list(mm.current_loaded_models)}
    try:
        return _orig_load_models_gpu(
            models,
            memory_required=memory_required,
            force_patch_weights=force_patch_weights,
            minimum_memory_required=minimum_memory_required,
            force_full_load=force_full_load,
        )
    finally:
        with _lock:
            after = list(mm.current_loaded_models)
            new_lms = [lm for lm in after if id(lm) not in before]
            if new_lms:
                _consume_load(new_lms)
        if config.enabled and config.prefetch_enabled:
            try:
                _maybe_prefetch()
            except Exception:
                logger.debug("[smart-offload] prefetch failed", exc_info=True)


# --------------------------------------------------------------------------
# ComfyUI node (workflow-level configuration)
# --------------------------------------------------------------------------

class SmartResidentScheduler:
    """Place this node in a workflow to configure the scheduler for that
    prompt.  If the node is absent, global defaults (conservative) apply."""

    CATEGORY = "lmsl-smart-offload"
    FUNCTION = "nop"
    RETURN_TYPES = ()

    @classmethod
    def INPUT_TYPES(cls) -> dict:
        return {
            "required": {
                "enabled": ("BOOLEAN", {"default": True, "label_on": "enabled", "label_off": "disabled"}),
                "keep_steps": ("INT", {"default": 1, "min": 0, "max": 100, "step": 1}),
                "prefetch_enabled": ("BOOLEAN", {"default": False}),
                "prefetch_min_free_mb": ("INT", {"default": 2048, "min": 0, "max": 32768, "step": 128}),
                "verbose": ("BOOLEAN", {"default": False}),
            }
        }

    def nop(self, **kwargs) -> tuple:
        # The on_prompt_handler already read the widgets; this call keeps the
        # node a no-op in the execution graph.
        return ()


# --------------------------------------------------------------------------
# Install patches (idempotent)
# --------------------------------------------------------------------------

def install() -> None:
    global _patched
    if _patched:
        return
    try:
        mm.free_memory = _patched_free_memory
        mm.load_models_gpu = _patched_load_models_gpu
        _patched = True
        logger.info("[lmsl-smart-offload] patched free_memory / load_models_gpu")
    except Exception:
        logger.warning("[lmsl-smart-offload] failed to patch model_management", exc_info=True)
        raise


def register_prompt_handler() -> bool:
    """Register the on_prompt_handler with PromptServer.instance.

    Returns True when registration succeeded.  ComfyUI imports custom_nodes
    after PromptServer() is constructed, but we retry via a timer in
    __init__.py for robustness.
    """
    try:
        from server import PromptServer
    except Exception:
        return False
    instance = getattr(PromptServer, "instance", None)
    if instance is None:
        return False
    try:
        instance.add_on_prompt_handler(on_prompt_handler)
        logger.info("[lmsl-smart-offload] on_prompt_handler registered")
        return True
    except Exception:
        logger.warning("[lmsl-smart-offload] add_on_prompt_handler failed", exc_info=True)
        return False
