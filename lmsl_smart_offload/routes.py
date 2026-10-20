"""只读 HTTP 路由：把 TDS SDK 的选型结果暴露给前端 / 命令行。

设计原则与桥接层一致：

* **只读**：只返回查询结果，不修改显存状态、不改工作流、不入队；
* **fail-soft**：SDK 未就绪 / 未安装 / 探针失败时返回结构化状态与原因，
  而不是 500；
* **不阻塞**：全部读内存快照，不等待后台探测完成，事件循环零 IO 等待。
"""

from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional, Tuple

from . import sdk_bridge
from .version import __version__

logger = logging.getLogger("lmsl-smart-offload")

API_PREFIX = "/lmsl_sdk"

_TRUE = ("1", "true", "yes", "on", "y")


def _as_int(value: Optional[str], default: int) -> int:
    try:
        return int(str(value).strip())
    except (TypeError, ValueError):
        return default


def _as_bool(value: Optional[str], default: bool) -> bool:
    if value is None or str(value).strip() == "":
        return default
    return str(value).strip().lower() in _TRUE


def _tokens(request: Any, key: str) -> Tuple[str, ...]:
    """收集查询参数，支持 `?tags=a,b` 与 `?tags=a&tags=b` 两种写法。"""
    raw: List[str] = []
    try:
        raw.extend(request.query.getall(key, []))
    except (AttributeError, KeyError, TypeError):
        single = request.query.get(key)
        if single:
            raw.append(single)
    out: List[str] = []
    for chunk in raw:
        for part in str(chunk).split(","):
            token = part.strip()
            if token and token not in out:
                out.append(token)
    return tuple(out)


def _json(payload: Dict[str, Any], status: int = 200) -> Any:
    from aiohttp import web  # 延迟导入：单元测试环境可能没有 aiohttp

    return web.json_response(payload, status=status)


def register_routes() -> bool:
    """把三个只读端点挂到 ComfyUI 路由表上；失败返回 False（宿主照常启动）。"""
    try:
        try:
            from server import PromptServer
        except ImportError:  # 非 ComfyUI 环境（单测）
            return False

        instance = getattr(PromptServer, "instance", None)
        routes = getattr(instance, "routes", None)
        if routes is None:
            return False

        @routes.get(API_PREFIX + "/status")
        async def lmsl_sdk_status(request: Any) -> Any:
            payload = sdk_bridge.status_payload()
            payload["version"] = __version__
            payload["endpoints"] = [
                API_PREFIX + "/status",
                API_PREFIX + "/recommend",
                API_PREFIX + "/advise",
            ]
            return _json(payload)

        @routes.get(API_PREFIX + "/recommend")
        async def lmsl_sdk_recommend(request: Any) -> Any:
            query = request.query
            payload = sdk_bridge.recommend(
                domain=(query.get("domain") or "").strip(),
                tags=_tokens(request, "tags") or _tokens(request, "tag"),
                limit=max(0, _as_int(query.get("limit"), 3)),
                exclude=_tokens(request, "exclude"),
                include_offload=_as_bool(query.get("include_offload"), True),
            )
            payload["version"] = __version__
            return _json(payload)

        @routes.get(API_PREFIX + "/advise")
        async def lmsl_sdk_advise(request: Any) -> Any:
            model = (request.query.get("model") or "").strip()
            if not model:
                return _json({
                    "error": "缺少 model 参数",
                    "hint": API_PREFIX + "/advise?model=<模型文件名>",
                    "version": __version__,
                }, status=400)
            payload = sdk_bridge.advise(
                model, limit=max(0, _as_int(request.query.get("limit"), 3)),
            )
            payload["version"] = __version__
            return _json(payload)

        logger.debug("[lmsl-smart-offload] sdk routes registered")
        return True
    except Exception:  # noqa: BLE001 - 路由注册失败不影响插件本体
        logger.debug("[lmsl-smart-offload] sdk routes registration skipped", exc_info=True)
        return False
