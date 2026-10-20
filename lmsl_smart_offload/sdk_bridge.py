# -*- coding: utf-8 -*-
"""TDS SDK 选型桥接 —— 让 smart-offload 用推荐清单校验人工选模型。

定位
----
lmsl-smart-offload 原本只回答"显存不够时先换出谁"，**选哪个模型全靠人工**。
TDS 方向1 SDK（``tds.sdk.SetupEngine``）已能把「硬件画像 → 模型匹配 → 上下文
预算」一次算完。本模块把这份结果接进 smart-offload 的启动与出图路径：

1. **启动路径**：ComfyUI 启动后异步跑一次 ``SetupEngine.setup()``
   （显存探测 + 本地权重扫描 + 可行性匹配），结果缓存在进程内；
   探测在后台 daemon 线程执行，不阻塞启动，失败不影响调度器；
2. **出图路径**：``on_prompt_handler`` 解析工作流后，把里面**人工选定的模型
   文件名**逐一回查 SDK 目录，给出 ok / tight / oom / unknown / untracked 判定；
   tight/oom 时从同一 domain 的可行条目里挑评分最高的替补作为建议；
3. **只读**：只给判定与建议，绝不改写工作流、不下载权重、不替换模型。

判定口径（三态 + 两类补充）
--------------------------
    ok         估算可容纳，放行
    tight      余量偏小，可运行但建议关注峰值
    offload    超出单卡显存，但总规模在「显存 + 一半内存」内，由本插件
               分层卸载执行（能跑，速度下降），不计告警
    oom        超出可卸载总量上限，实际跑不动，给同域替补
    unknown    条目存在但体积未解析，保守放行、不臆测
    untracked  目录里没有该文件，放行（不自作主张）

开关
----
``LMSL_SDK_BRIDGE=0`` 关闭整个桥接（等价未接入）；默认开启。
"""
from __future__ import annotations

import logging
import os
import sys
import threading
import time
from typing import Any, Dict, FrozenSet, List, Optional, Tuple

logger = logging.getLogger("lmsl-smart-offload.sdk")

#: 视为"模型文件引用"的后缀（与 ComfyUI 权重类型对齐）
WEIGHT_EXTS: Tuple[str, ...] = (
    ".safetensors", ".ckpt", ".pt", ".pth", ".bin", ".gguf", ".onnx",
)

#: 单次 prompt 体检最多回查的模型引用条数（防止异常大图拖慢提交）
MAX_REFS = 64

_ENV_SWITCH = "LMSL_SDK_BRIDGE"
_OFF_VALUES = ("0", "off", "false", "no", "disabled", "none")

# SDK 模块缓存（进程级；失败也缓存，避免每次 prompt 反复尝试导入）
_SDK_MODULE: Any = None
_SDK_ERROR: str = ""
_SDK_LOCK = threading.Lock()


def enabled() -> bool:
    """桥接开关（环境变量优先，默认开启）。"""
    raw = os.environ.get(_ENV_SWITCH, "1")
    return str(raw).strip().lower() not in _OFF_VALUES


# -------------------------------------------------------------------------- #
# SDK 模块定位
# -------------------------------------------------------------------------- #

def _tds_root() -> Optional[str]:
    """定位 TDS 项目根（含 ``tds/config.py``）；找不到返回 None。

    依次尝试：环境变量 TDS_ROOT、custom_nodes 下的 TDS 目录、本插件相邻目录。
    """
    def has_pkg(base: Optional[str]) -> bool:
        return bool(base) and os.path.isfile(os.path.join(base, "tds", "config.py"))

    here = os.path.dirname(os.path.abspath(__file__))
    plugin_root = os.path.dirname(here)
    custom_nodes = os.path.dirname(plugin_root)
    candidates = [
        os.environ.get("TDS_ROOT"),
        os.path.join(custom_nodes, "TDS"),
        os.path.join(plugin_root, "..", "TDS"),
        os.path.join(here, "..", "..", "..", "TDS"),
    ]
    for base in candidates:
        if has_pkg(base):
            return os.path.abspath(base)
    return None


def load_sdk(refresh: bool = False) -> Any:
    """导入 TDS SDK 模块（``tds.sdk``），失败返回 None 并缓存原因。

    导入顺序：已加载模块 → ``tds.sdk``（补 TDS 根到 sys.path）→
    ``TDS.tds.sdk``（TDS 作为 custom_node 包被导入时的名字）。
    """
    global _SDK_MODULE, _SDK_ERROR
    with _SDK_LOCK:
        if _SDK_MODULE is not None and not refresh:
            return _SDK_MODULE
        if _SDK_ERROR and not refresh:
            return None
        for name in ("tds.sdk", "TDS.tds.sdk"):
            mod = sys.modules.get(name)
            if mod is not None:
                _SDK_MODULE = mod
                _SDK_ERROR = ""
                return mod
        root = _tds_root()
        if root and root not in sys.path:
            sys.path.insert(0, root)
        errors: List[str] = []
        for name in ("tds.sdk", "TDS.tds.sdk"):
            try:
                import importlib

                _SDK_MODULE = importlib.import_module(name)
                _SDK_ERROR = ""
                return _SDK_MODULE
            except Exception as exc:  # noqa: BLE001 - fail-soft，原因入日志
                errors.append("{}: {}: {}".format(name, type(exc).__name__, exc))
        _SDK_MODULE = None
        _SDK_ERROR = " | ".join(errors) or "未找到 TDS SDK（tds.sdk）"
        logger.info("[smart-offload] TDS SDK 不可用：%s", _SDK_ERROR)
        return None


# -------------------------------------------------------------------------- #
# 内部工具
# -------------------------------------------------------------------------- #

def _basename(value: str) -> str:
    return os.path.basename(str(value).replace("\\", "/"))


def _is_weight_ref(value: str) -> bool:
    """判断 inputs 里的字符串是否是模型文件引用（排除 URL / 超长文本）。"""
    if not value or len(value) > 512:
        return False
    low = value.strip().lower()
    if low.startswith(("http://", "https://")):
        return False
    return low.endswith(WEIGHT_EXTS)


def _display_name(key: str) -> str:
    """目录 key -> 可读名：``local:models/x/y.safetensors`` -> ``y.safetensors``。"""
    text = str(key or "")
    if ":" in text:
        text = text.split(":", 1)[1]
    return _basename(text) or text


def collect_model_refs(prompt: Any) -> List[Dict[str, str]]:
    """从 prompt 图里抽取模型文件引用，按 (class_type, 文件名) 去重。"""
    if not isinstance(prompt, dict):
        return []
    seen = set()
    refs: List[Dict[str, str]] = []
    for node_id, node in prompt.items():
        if not isinstance(node, dict):
            continue
        class_type = str(node.get("class_type") or "")
        inputs = node.get("inputs")
        if not isinstance(inputs, dict):
            continue
        for field, value in inputs.items():
            if not isinstance(value, str) or not _is_weight_ref(value):
                continue
            name = _basename(value)
            token = (class_type, name.lower())
            if token in seen:
                continue
            seen.add(token)
            refs.append({
                "node_id": str(node_id),
                "class_type": class_type,
                "field": str(field),
                "file": name,
            })
            if len(refs) >= MAX_REFS:
                return refs
    return refs


# -------------------------------------------------------------------------- #
# 桥接状态机
# -------------------------------------------------------------------------- #
#
#   disabled  开关关闭
#   idle      尚未探测（默认状态，首个 warmup 前）
#   probing   后台线程正在跑 setup()
#   ready     探测完成，索引可用
#   failed    探测失败（原因见 reason），后续一律放行
#
STATE_DISABLED = "disabled"
STATE_IDLE = "idle"
STATE_PROBING = "probing"
STATE_READY = "ready"
STATE_FAILED = "failed"

#: 判定等级 -> 是否告警
_WARN_STATUS = ("tight", "oom")
#: 替补同域筛选用的家族/用途标签（子串匹配，长标签在前避免误命中）
_FAMILY_TAGS = (
    "wan2.2", "wan2.1", "wan", "lightx2v", "umt5", "t5", "vae", "lora",
    "clip", "ltx", "flux", "qwen", "sd_xl", "sdxl", "sd3", "hunyuan", "svd",
)


class SdkBridge:
    """TDS SDK 选型桥接（线程安全，只读消费 SDK 结果）。"""

    def __init__(self) -> None:
        self._lock = threading.RLock()
        self._state = STATE_IDLE
        self._reason = ""
        self._thread: Optional[threading.Thread] = None
        self._generation = 0
        self._elapsed = 0.0
        self._matches: List[Dict[str, Any]] = []
        #: 全量评估（每条目录条目一条，不截断）
        self._evaluations: List[Dict[str, Any]] = []
        #: 全量评估中 status=ok 的子集（替补候选池）
        self._feasible: List[Dict[str, Any]] = []
        self._hardware: Dict[str, Any] = {}
        self._software: Dict[str, Any] = {}
        self._context: Optional[Dict[str, Any]] = None
        self._catalog: Dict[str, Any] = {}
        self._by_name: Dict[str, Any] = {}
        self._match_by_key: Dict[str, Dict[str, Any]] = {}
        self._last_audit: Optional[Dict[str, Any]] = None
        self._audit_count = 0
        self._warn_count = 0

    # ---------------- 状态 ----------------

    @property
    def state(self) -> str:
        with self._lock:
            return self._state

    def ready(self) -> bool:
        return self.state == STATE_READY

    # ---------------- 探测 ----------------

    def warmup(self, force: bool = False) -> bool:
        """启动后台探测；已在探测/已就绪时幂等返回。

        返回 True 表示本次调用真正拉起了线程。
        """
        if not enabled():
            with self._lock:
                self._state = STATE_DISABLED
                self._reason = "环境变量 {} 已关闭".format(_ENV_SWITCH)
            return False
        with self._lock:
            if not force and self._state in (STATE_PROBING, STATE_READY):
                return False
            if self._thread is not None and self._thread.is_alive():
                return False
            self._state = STATE_PROBING
            self._reason = ""
            self._thread = threading.Thread(
                target=self._probe_worker,
                name="lmsl-sdk-probe",
                daemon=True,
            )
            self._thread.start()
            return True

    def wait(self, timeout: float = 0.0) -> bool:
        """等待探测完成（测试与 CLI 用）；返回是否已就绪。"""
        thread = self._thread
        if thread is not None and thread.is_alive():
            thread.join(timeout if timeout and timeout > 0 else None)
        return self.ready()

    def _probe_worker(self) -> None:
        started = time.perf_counter()
        try:
            sdk = load_sdk()
            if sdk is None:
                with self._lock:
                    self._state = STATE_FAILED
                    self._reason = _SDK_ERROR or "TDS SDK 不可用"
                return
            engine = sdk.get_setup_engine()
            result = engine.setup()
            payload = result.to_dict() if hasattr(result, "to_dict") else dict(result)
            specs = engine.catalog_specs() if hasattr(engine, "catalog_specs") else {}
            matches = list(payload.get("matches") or [])
            # evaluations = 全量评估（不截断）；老版本 SDK 没有该字段时退回 matches
            evaluations = list(payload.get("evaluations") or matches)
            by_name, match_by_key = self._index(specs, evaluations)
            feasible = [
                m for m in evaluations
                if isinstance(m, dict) and m.get("status") == "ok"
            ]
            with self._lock:
                self._hardware = payload.get("hardware") or {}
                self._software = payload.get("software") or {}
                self._context = payload.get("context")
                self._catalog = payload.get("catalog") or {}
                self._matches = matches
                self._evaluations = evaluations
                self._feasible = feasible
                self._by_name = by_name
                self._match_by_key = match_by_key
                self._generation += 1
                self._state = STATE_READY
                self._reason = ""
        except Exception as exc:  # noqa: BLE001 - 宿主启动不能被拖垮
            with self._lock:
                self._state = STATE_FAILED
                self._reason = "{}: {}".format(type(exc).__name__, exc)
            logger.warning("[smart-offload] SDK 探测失败（选型建议停用）", exc_info=True)
        finally:
            with self._lock:
                self._elapsed = time.perf_counter() - started
            logger.info(
                "[smart-offload] SDK 探测结束 state=%s entries=%d %.2fs",
                self._state, len(self._matches), self._elapsed,
            )

    @staticmethod
    def _index(
        specs: Dict[str, Any],
        matches: List[Dict[str, Any]],
    ) -> Tuple[Dict[str, Any], Dict[str, Dict[str, Any]]]:
        """建立「文件名 -> 条目」与「key -> 匹配结果」两张索引。"""
        by_name: Dict[str, Any] = {}
        # 本地扫描条目带真实体积，优先入索引；内置条目往往 size 未解析，
        # 若让它先占坑，同一文件的判定会退化成 unknown。
        ordered = sorted(
            (specs or {}).items(),
            key=lambda kv: 0 if getattr(kv[1], "sizes", None) else 1,
        )
        for key, spec in ordered:
            names: List[str] = []
            files = getattr(spec, "files", None)
            if files:
                names.extend(str(f) for f in files)
            names.append(_display_name(key))
            for name in names:
                base = _basename(name).lower()
                if not base:
                    continue
                prev = by_name.get(base)
                if prev is None or (
                    not getattr(prev, "sizes", None) and getattr(spec, "sizes", None)
                ):
                    by_name[base] = spec
        match_by_key: Dict[str, Dict[str, Any]] = {}
        for item in matches or []:
            if isinstance(item, dict) and item.get("key"):
                match_by_key[str(item["key"])] = item
        return by_name, match_by_key

    # ---------------- 单条判定 ----------------

    def _substitutes(self, spec: Any, limit: int = 3) -> List[Dict[str, Any]]:
        """为不可行/余量紧张的条目挑选同 domain 的可行替补。"""
        domain = str(getattr(spec, "domain", "") or "")
        key = str(getattr(spec, "key", "") or "")
        return self._substitutes_for_domain(
            domain, limit, exclude_key=key, family=self._family_of(key),
        )

    def judge(self, filename: str) -> Dict[str, Any]:
        """判定单个模型文件名（只读）。"""
        base = _basename(filename).lower()
        spec = self._by_name.get(base)
        if spec is None:
            return {
                "file": filename,
                "status": "untracked",
                "verdict": "pass",
                "reason": "SDK 目录未收录该文件，不做可行性判断（放行）",
                "substitutes": [],
            }
        key = str(getattr(spec, "key", "") or "")
        match = self._match_by_key.get(key)
        if match is None:
            status = "unknown"
            reason = "条目存在但体积未解析，保守放行"
            footprint = 0.0
            headroom = 0.0
        else:
            status = str(match.get("status") or match.get("verdict_level") or "unknown")
            footprint = float(match.get("footprint_mb") or 0.0)
            headroom = float(match.get("headroom_mb") or 0.0)
            reason = str(match.get("reason") or "")
            if status == "oom":
                reason = reason or "预算内不可行"
                # 宿主就是分层卸载插件：单模型超出显存并不等于跑不动，
                # 只要总规模落在「显存 + 一半内存」内就降级为 offload（慢了但能跑）。
                if 0.0 < footprint <= self._offload_ceiling_mb():
                    status = "offload"
                    reason = (
                        "超出显存 {:.0f} MB，由 smart-offload 分层卸载执行（速度下降）"
                        .format(max(0.0, -headroom))
                    )
                elif footprint > self._offload_ceiling_mb():
                    reason = "超出显存且超出可卸载总量上限，实际不可运行"
        result: Dict[str, Any] = {
            "file": filename,
            "status": status,
            "verdict": "warn" if status in _WARN_STATUS else "pass",
            "key": key,
            "domain": str(getattr(spec, "domain", "") or ""),
            "footprint_mb": footprint,
            "headroom_mb": headroom,
            "reason": reason,
            "substitutes": [],
        }
        if status in _WARN_STATUS:
            result["substitutes"] = self._substitutes(spec)
        return result

    # ---------------- prompt 体检 ----------------

    def audit(self, prompt: Any) -> Dict[str, Any]:
        """对一张 prompt 图的人工选型做体检（只读，任何异常都不外抛）。"""
        try:
            refs = collect_model_refs(prompt)
        except Exception:  # noqa: BLE001 - 体检失败不影响提交
            logger.debug("[smart-offload] 模型引用抽取失败", exc_info=True)
            refs = []
        with self._lock:
            state = self._state
            generation = self._generation
        if state != STATE_READY:
            payload = {
                "state": state,
                "generation": generation,
                "verdict": "pending" if state in (STATE_IDLE, STATE_PROBING) else "pass",
                "models": [],
                "summary": {"total": len(refs), "warn": 0, "pending": len(refs)},
                "note": "SDK 未就绪，本次不做选型判定",
            }
            return payload

        models: List[Dict[str, Any]] = []
        counts: Dict[str, int] = {}
        for ref in refs:
            judged = self.judge(ref["file"])
            judged.update({
                "node_id": ref["node_id"],
                "class_type": ref["class_type"],
                "field": ref["field"],
            })
            models.append(judged)
            status = str(judged.get("status"))
            counts[status] = counts.get(status, 0) + 1
        warn = sum(v for k, v in counts.items() if k in _WARN_STATUS)
        payload = {
            "state": state,
            "generation": generation,
            "verdict": "warn" if warn else "ok",
            "models": models,
            "summary": {
                "total": len(models),
                "warn": warn,
                "counts": counts,
            },
        }
        with self._lock:
            self._last_audit = payload
            self._audit_count += 1
            if warn:
                self._warn_count += 1
        return payload

    def advise(self, model_name: str, limit: int = 3) -> Dict[str, Any]:
        """给单个模型出建议（节点入口）：判定 + 同域可行替补。"""
        if not model_name or not str(model_name).strip():
            return {"file": "", "status": "empty", "verdict": "pass",
                    "reason": "未提供模型文件名", "substitutes": []}
        if self.state != STATE_READY:
            return {"file": str(model_name), "status": self.state,
                    "verdict": "pending", "reason": "SDK 未就绪（后台探测中或已停用）",
                    "substitutes": []}
        judged = self.judge(str(model_name).strip())
        # untracked（目录未收录）、ok（本来就可行）、offload（已能跑，只是慢）
        # 都不需要替补，避免噪音
        if judged.get("status") in ("untracked", "empty", "error", "ok", "offload"):
            judged["substitutes"] = []
        elif not judged.get("substitutes"):
            judged["substitutes"] = self._substitutes_for_domain(
                judged.get("domain", ""), limit,
                exclude_key=str(judged.get("key") or ""),
                family=self._family_of(str(judged.get("file") or judged.get("key") or "")),
            )
        judged["substitutes"] = judged["substitutes"][:max(0, int(limit))]
        return judged

    def _effective_status(self, item: Dict[str, Any]) -> str:
        """评估条目的"落地上"状态：TDS 判 oom 但能在可卸载范围内 → offload。"""
        status = str(item.get("status") or "")
        if status == "oom":
            try:
                footprint = float(item.get("footprint_mb") or 0.0)
            except (TypeError, ValueError):
                footprint = 0.0
            if 0.0 < footprint <= self._offload_ceiling_mb():
                return "offload"
        return status

    def recommend(
        self,
        domain: str = "",
        tags: Tuple[str, ...] = (),
        limit: int = 3,
        exclude: Tuple[str, ...] = (),
        include_offload: bool = True,
    ) -> Dict[str, Any]:
        """按创作意图（域 + 家族/用途标签）给出首选模型，替代人工盲选。

        与 ``advise`` 方向相反：advise 是"你选了 A，A 行不行/换什么"；
        recommend 是"我要做文生视频，该选哪个"，直接消费 SDK 的 matches。

        tags 为"且"关系，例如 ("wan2.1", "t2v") 命中 wan2.1 文生视频主模型；
        include_offload=True 时把"超出显存但能分层卸载"的大模型也纳入推荐，
        并按 ok/tight 优先、offload 垫后排序。

        参数:
            domain: 目录域过滤（diffusion / text_encoder / vae / llm ...），空表示不限。
            tags: 家族或文件名子串标签元组。
            limit: 返回条数上限。
            exclude: 需要排除的文件名（小写比较）。
            include_offload: 是否把 offload 档纳入推荐。
        """
        payload: Dict[str, Any] = {
            "state": self.state, "domain": domain, "tags": list(tags),
            "ready": self.state == STATE_READY, "picks": [], "reason": "",
        }
        if self.state != STATE_READY:
            payload["reason"] = "SDK 未就绪（后台探测中或已停用）"
            return payload
        try:
            want = frozenset(t.lower() for t in tags if str(t).strip())
            blocked = {str(x).lower() for x in exclude if str(x).strip()}
            pool = []
            for item in self._evaluations:
                if not isinstance(item, dict):
                    continue
                status = self._effective_status(item)
                healthy = status in ("ok", "tight")
                if not healthy and not (include_offload and status == "offload"):
                    continue
                if domain and str(item.get("domain") or "") != domain:
                    continue
                name = _basename(str(item.get("file") or item.get("key") or ""))
                fam = set(self._family_of(name))
                low = name.lower()
                # 标签既支持家族词（wan2.1 / sdxl），也支持任意文件名子串
                # （t2v / i2v / talk 这类用途词不进家族表）；多个标签为"且"关系，
                # 用 ("wan2.1", "t2v") 就能精确缩到文生视频主模型
                if want and not all(
                    (t in fam or t in low) for t in want
                ):
                    continue
                if blocked and name.lower() in blocked:
                    continue
                pool.append((item, name))
            # 先按可行性分层（ok/tight 在前，offload 垫后），再按"体量"降序：
            # 同等可行下大模型效果更好，小模型只作兜底
            pool.sort(key=lambda x: (
                0 if self._effective_status(x[0]) in ("ok", "tight") else 1,
                -float(x[0].get("score") or 0.0),
                -float(x[0].get("footprint_mb") or 0.0),
                x[1].lower(),
            ))
            limit_n = max(0, int(limit))
            seen: set = set()
            for item, name in (pool if limit_n > 0 else ()):
                low = name.lower()
                if low in seen:  # 同名模型可能落在不同目录，推荐时按文件名去重
                    continue
                seen.add(low)
                payload["picks"].append({
                    "file": name,
                    "key": str(item.get("key") or ""),
                    "domain": str(item.get("domain") or ""),
                    "status": self._effective_status(item),
                    "footprint_mb": round(float(item.get("footprint_mb") or 0.0), 2),
                    "headroom_mb": round(float(item.get("headroom_mb") or 0.0), 2),
                    "reason": str(item.get("reason") or ""),
                })
                if len(payload["picks"]) >= limit_n:
                    break
            payload["reason"] = (
                "命中 {} 条可行候选".format(len(pool)) if pool
                else (
                    "该意图下没有匹配的候选（标签/域过滤后为空）" if want
                    else "该意图下没有可行候选（体积/显存不匹配）"
                )
            )
        except Exception:  # noqa: BLE001 - 推荐失败不影响出图
            logger.debug("[smart-offload] 选型推荐异常（已忽略）", exc_info=True)
            payload["reason"] = "推荐过程异常"
        return payload

    def _offload_ceiling_mb(self) -> float:
        """可分层卸载的规模上限（MB）：总显存 + 一半物理内存。

        超过该值的权重连 CPU 内存都放不下，桥接层才认为真的跑不动。
        """
        vram = 0.0
        for dev in (self._hardware.get("devices") or []):
            try:
                vram += float(dev.get("total_mb") or 0)
            except (TypeError, ValueError):
                continue
        try:
            ram = float(self._hardware.get("ram_mb") or 0)
        except (TypeError, ValueError):
            ram = 0.0
        return vram + ram * 0.5

    @staticmethod
    def _family_of(name: str) -> FrozenSet[str]:
        """按文件名提取模型家族/用途标签（wan2.1 / wan2.2 / lightx2v / lora ...）。

        domain 粒度偏粗（LoRA、放大模型都落在 other），只按 domain 选替补
        会把 4x 放大模型推给视频 LoRA，因此再收一层标签交集约束。
        """
        low = _basename(str(name)).lower()
        return frozenset(tag for tag in _FAMILY_TAGS if tag in low)

    def _substitutes_for_domain(
        self,
        domain: str,
        limit: int,
        exclude_key: str = "",
        family: FrozenSet[str] = frozenset(),
    ) -> List[Dict[str, Any]]:
        # 无 domain 时不跨域乱推（跨域替补对选型毫无意义）
        if not domain:
            return []
        pool = [
            m for m in self._feasible
            if str(m.get("domain")) == domain
            and str(m.get("key")) != exclude_key
        ]
        if family:
            # 有家族标签时只收有交集的条目，宁可给不出建议也不误导
            pool = [
                m for m in pool
                if self._family_of(_display_name(m.get("key"))) & family
            ]
        pool.sort(key=lambda m: (-float(m.get("score") or 0.0), _display_name(m.get("key"))))
        return [{
            "key": m.get("key"),
            "name": _display_name(m.get("key")),
            "domain": m.get("domain"),
            "score": m.get("score"),
            "footprint_mb": m.get("footprint_mb"),
            "headroom_mb": m.get("headroom_mb"),
        } for m in pool[:max(0, int(limit))]]

    # ---------------- 可观测 ----------------

    def status_payload(self) -> Dict[str, Any]:
        """给 HTTP 路由 / 日志用的状态快照。"""
        with self._lock:
            return {
                "enabled": enabled(),
                "state": self._state,
                "reason": self._reason,
                "generation": self._generation,
                "elapsed_s": round(self._elapsed, 3),
                "hardware": self._hardware,
                "software": self._software,
                "context": self._context,
                "catalog": self._catalog,
                "matches_total": len(self._matches),
                "evaluations_total": len(self._evaluations),
                "feasible_total": len(self._feasible),
                "top_matches": [
                    {
                        "name": _display_name(m.get("key")),
                        "domain": m.get("domain"),
                        "status": m.get("status"),
                        "score": m.get("score"),
                        "footprint_mb": m.get("footprint_mb"),
                        "headroom_mb": m.get("headroom_mb"),
                    }
                    for m in self._matches[:5]
                ],
                "audits": self._audit_count,
                "audits_with_warning": self._warn_count,
                "last_audit": self._last_audit,
            }

    def log_line(self, audit_result: Dict[str, Any]) -> str:
        """把一次体检压成单行日志（默认输出，避免刷屏）。"""
        summary = audit_result.get("summary") or {}
        if audit_result.get("state") != STATE_READY:
            return "[sdk] 选型体检跳过（{}）".format(audit_result.get("state"))
        parts = []
        for item in audit_result.get("models") or []:
            label = "{}={}".format(item.get("file"), item.get("status"))
            subs = item.get("substitutes") or []
            if subs:
                label += "(建议: {})".format(subs[0].get("name"))
            parts.append(label)
        head = "[sdk] 选型体检 {} 个模型 verdict={}".format(
            summary.get("total", 0), audit_result.get("verdict"),
        )
        return head + (" | " + "，".join(parts) if parts else "")

    def reset(self) -> None:
        """丢弃探测结果（测试 / 环境变更后重探）。"""
        with self._lock:
            self._state = STATE_IDLE
            self._reason = ""
            self._thread = None
            self._matches = []
            self._evaluations = []
            self._feasible = []
            self._hardware = {}
            self._software = {}
            self._context = None
            self._catalog = {}
            self._by_name = {}
            self._match_by_key = {}
            self._last_audit = None
            self._audit_count = 0
            self._warn_count = 0


# -------------------------------------------------------------------------- #
# 进程级单例
# -------------------------------------------------------------------------- #
_BRIDGE: Optional[SdkBridge] = None
_BRIDGE_LOCK = threading.Lock()


def get_bridge() -> SdkBridge:
    """返回进程级桥接单例。"""
    global _BRIDGE
    if _BRIDGE is None:
        with _BRIDGE_LOCK:
            if _BRIDGE is None:
                _BRIDGE = SdkBridge()
    return _BRIDGE


def reset_bridge() -> None:
    """丢弃单例（测试用）。"""
    global _BRIDGE
    with _BRIDGE_LOCK:
        _BRIDGE = None


# -------------------------------------------------------------------------- #
# 模块级便捷入口（供 smart_offload 与节点直接调用，全部 fail-soft）
# -------------------------------------------------------------------------- #

def warmup(force: bool = False) -> bool:
    """启动后台 SDK 探测（默认异步，不阻塞宿主）。"""
    return get_bridge().warmup(force=force)


def audit_prompt(prompt: Any) -> Dict[str, Any]:
    """体检一张 prompt 图的人工选型。"""
    try:
        return get_bridge().audit(prompt)
    except Exception:  # noqa: BLE001 - 体检永不阻塞出图
        logger.debug("[smart-offload] 选型体检异常（已忽略）", exc_info=True)
        return {"state": "error", "verdict": "pass", "models": [], "summary": {"total": 0}}


def advise(model_name: str, limit: int = 3) -> Dict[str, Any]:
    """单模型建议。"""
    return get_bridge().advise(model_name, limit=limit)


def recommend(
    domain: str = "",
    tags: Tuple[str, ...] = (),
    limit: int = 3,
    exclude: Tuple[str, ...] = (),
    include_offload: bool = True,
) -> Dict[str, Any]:
    """按创作意图推荐模型（替代人工盲选）。"""
    try:
        return get_bridge().recommend(
            domain=domain, tags=tags, limit=limit, exclude=exclude,
            include_offload=include_offload,
        )
    except Exception:  # noqa: BLE001 - 推荐失败不影响出图
        logger.debug("[smart-offload] 选型推荐异常（已忽略）", exc_info=True)
        return {"state": "error", "ready": False, "picks": [], "reason": "推荐异常"}


def status_payload() -> Dict[str, Any]:
    """状态快照。"""
    return get_bridge().status_payload()
