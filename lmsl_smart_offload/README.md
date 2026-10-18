---
AIGC:
    Label: "1"
    ContentProducer: 001191440300708461136T1XGW3
    ProduceID: 350ab7144af9618434cdce7bfa6d84e8_7093af94b0ef11f1ac01525400e6dd8f
    ReservedCode1: a9hQcGtbSex8ltxcj5jhVXtcBDXJ5kwSLqZpbS9ppUFgb5+grp2V1Odh+35Afc1Ohm/X9zcXoK56wblBkddkLanJo1+FUlHVjyP98D7MKPV1AFHJD4t9xj3u+cWkLUyeTcu+kvtbm9xOSvm9spNAY4mXEwNPhZGwT/n0fTlmShOXZ/TmYN5Bauo1LoQ=
    ContentPropagator: 001191440300708461136T1XGW3
    PropagateID: 350ab7144af9618434cdce7bfa6d84e8_7093af94b0ef11f1ac01525400e6dd8f
    ReservedCode2: a9hQcGtbSex8ltxcj5jhVXtcBDXJ5kwSLqZpbS9ppUFgb5+grp2V1Odh+35Afc1Ohm/X9zcXoK56wblBkddkLanJo1+FUlHVjyP98D7MKPV1AFHJD4t9xj3u+cWkLUyeTcu+kvtbm9xOSvm9spNAY4mXEwNPhZGwT/n0fTlmShOXZ/TmYN5Bauo1LoQ=
---

# Smart Resident Scheduler (lmsl-smart-offload)

面向 ComfyUI 低显存模式（`--lowvram` / `--enable-dynamic-vram`）的**未来感知驻留调度器**。
目标：减少模型换入换出抖动（swap thrashing）导致的性能损失，在 12GB 显存（如 RTX 3060）上让多模型工作流跑得更稳更快。

## 原理

ComfyUI 低显存模式下，模型在 GPU / CPU 之间按需换入换出。默认卸载策略对"接下来是否马上要用"不够敏感，
容易造成刚换出的模型立刻又要换回（抖动）。

本插件在 **prompt 排队时**（`PromptServer.instance.add_on_prompt_handler`，ComfyUI 0.35.x 官方钩子）预扫描执行图：

1. 对图做拓扑排序，得到近似的节点执行顺序；
2. 识别所有"模型来源节点"（`RETURN_TYPES` 含 `MODEL / CLIP / VAE`，且带文件选择型输入如 `ckpt_name / unet_name / clip_name / vae_name / lora_name` 等），
   生成稳定模型键 `f"{节点类}|{模型文件名}"`；
3. 统计每个模型（含同一模型被多节点共用、LoRA 叠加场景）在本次执行中的**使用序列**与**下次使用距离**。

执行阶段通过运行时包装（monkey-patch，不修改 ComfyUI 源码）两个关键 API：

- `comfy.model_management.free_memory` —— 卸载决策注入点：把"即将再次使用"（下次使用距离 ≤ `keep_steps`）的模型并入 `keep_loaded`，
  优先卸载"下次使用最远"的模型（近似 Belady 最优算法）；
- `comfy.model_management.load_models_gpu` —— 加载事件追踪：识别本次真正新建的 `LoadedModel`，推进计划游标；
  可选开启**预加载**：GPU 空闲且显存余量充足时，提前把下一个要用的模型唤醒（默认关闭）。

## 启用方式

1. 把本目录放入 `ComfyUI/custom_nodes/`（目录名 `lmsl-smart-offload`），重启 ComfyUI；
2. 无需任何手动配置即生效（采用保守默认值）；
3. 可选：在工作流中放入一个 `Smart Resident Scheduler (LMSL)` 节点，按本次工作流单独调整参数。

> 日志前缀：`[lmsl-smart-offload]`（`--verbose` 打开后输出更详细）。

## 参数说明（SmartResidentScheduler 节点 widgets）

| 参数 | 类型 | 默认 | 含义 |
| --- | --- | --- | --- |
| `enabled` | BOOLEAN | True | 总开关。关闭后恢复 ComfyUI 原调度行为 |
| `keep_steps` | INT | 1 | 驻留保护阈值：距下次使用 ≤ N 个拓扑步的模型在卸载时被保留。0 = 仅保护正在使用的模型（最保守） |
| `prefetch_enabled` | BOOLEAN | False | 预加载开关。打开后尝试在 GPU 空闲窗口提前唤醒下一个模型 |
| `prefetch_min_free_mb` | INT | 2048 | 预加载最低显存余量（MB）。余量低于该值 + 模型所需显存时不预加载 |
| `verbose` | BOOLEAN | False | 详细日志 |

未放置节点时使用全局默认（同上表，保守）。

## 安全与保守性设计

- **显存压力保护**：当空闲显存已低于 `minimum_inference_memory()` 时，驻留保护自动失效，完全交还原调度器，**避免引入 OOM**；
- **默认保守**：`keep_steps=1`、`prefetch_enabled=False`；
- **只读预扫描**：`on_prompt_handler` 仅读取 `json_data["prompt"]` 构建计划，不修改图、不拦截队列；
- **失败软降级**：任何内部异常只记日志，原调度行为不变；补丁安装失败不影响 ComfyUI 启动。

## 兼容性注意

- 已按 **ComfyUI 0.35.1**（本机魔改版，含 `--enable-dynamic-vram` / `comfy_aimdo`）核对 API：
  - `free_memory(memory_required, device, keep_loaded=[], for_dynamic=False, pins_required=0, ram_required=0)`（model_management.py L879）
  - `load_models_gpu(models, memory_required=0, force_patch_weights=False, minimum_memory_required=None, force_full_load=False)`（L1027）
  - `current_loaded_models`（L635）、`LoadedModel`（L757，`__eq__` 基于 ModelPatcher 引用）
  - `server.add_on_prompt_handler` / `trigger_on_prompt`（server.py L1456 / L1076）
  - 本版本 **没有** `PRE_EXECUTE_NODE` / `POST_EXECUTE_NODE` / `recursive_execute` 节点级钩子，故采用"prompt 预扫描 + 运行时包装"方案。
- 执行顺序为拓扑近似（缓存命中的节点不会触发加载事件，可能导致事件游标轻微偏移）；
  最坏影响仅为多保留一个模型（更保守），不会造成 OOM 或跳过节点；
- `ModelPatcher` 在本 fork 中无 `model_name` 字段，模型键基于"节点类 + 文件名输入"识别；
- 不修改任何 ComfyUI 源码；卸载逻辑仍在原 `free_memory` 内完成（排序规则、动态显存逻辑均保留）。

## 文件清单

- `__init__.py` —— 入口：安装补丁 + 注册 prompt 钩子 + 注册节点类
- `smart_offload.py` —— 核心实现（图预扫描 / 驻留策略 / 可选预加载）
- `README.md` —— 本说明
*（内容由AI生成，仅供参考）*
