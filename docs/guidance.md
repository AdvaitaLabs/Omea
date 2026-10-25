# AutoFragDiff 引导优化

## 方法与参数

引导模型为冻结的 GNINA CNN，联合计算坐标和连续元素特征的梯度。离散元素由连续特征生成；局部几何约束使用中间分子图的键长、键角，投影另外约束严重的分子内部原子重叠。约束不包含 Vina 或其他结合能评分。

采样保留 AutoFragDiff 的 500 步流程及配对噪声，每次尝试只生成一次。坐标更新上限为 0.02 Å，元素特征更新上限为 0.6，温度为 0.2；几何约束作用于最后 100 步。局部约束权重从 0.1 增至 0.3 是一组实验；另一组保留 0.1，并对坐标更新执行约束投影与回退。投影约束只适用于当前中间图上的坐标修正，不保证后续加噪或最终重建一定通过几何检查。

完整参数副本在 [configs/guidance_configurations.json](../configs/guidance_configurations.json)。实验执行仍读取 `guidance/scripts/cnn_clean_settings.py` 与 `cnn_geometry_followup_settings.py`，不会自动加载这个 JSON 副本。

## 主要代码

| 文件（均位于 `guidance/scripts/`） | 用途 |
|---|---|
| `gnina_soft_element_guidance.py` | 坐标与连续元素特征的联合 CNN 梯度 |
| `gnina_soft_element_client.py` / `gnina_soft_element_worker.py` | 生成环境和评分梯度环境之间的通信 |
| `cnn_local_geometry.py` | 局部几何惩罚及梯度 |
| `cnn_geometry_projection.py` | 键长、键角和内部重叠约束的投影 |
| `autofragdiff_cnn_geometry_hook.py` | 将引导和投影接入 AutoFragDiff 采样 |
| `run_cnn_geometry_followup.py` | 两张 GPU 的全量实验调度、续跑和汇总 |
| `score_cnn_clean.py` / `evaluate_cnn_clean_dual.py` | 方案冻结后的 CNN、Vina 和几何评估 |
| `summarize_cnn_geometry_followup.py` | 全尝试分母、配对结果核算及审计 |
| `check_cnn_geometry_followup.py` | 梯度、采样过程与投影约束的技术检查 |

其余脚本是这些入口依赖的此前筛参、采样、评估和审计模块。共 47 个文件：45 个与原运行冻结清单一致，双指标评估入口与较早的评估清单一致；`score_results.py` 是额外的旧有导入依赖。逐文件哈希在 [guidance/frozen_sources.json](../guidance/frozen_sources.json)。

原 benchmark 与历史实验的 `tasks.py`、`restore_scoring_serialization.py` 实现不同，因此引导脚本集中保存在独立目录。最终输出由 `NativeFinalEngine` 原生重建；旧依赖 `fixed_core_reconstruction.py` 在中间引导上下文等路径中使用，不代表加入了后续的最终核心保持实验。

## 环境与输入

历史脚本按原文保留，没有统一接入 `benchmark.py`。脚本以 `guidance/` 为运行根目录，并沿用部分原机器的绝对路径及前序实验目录；它们不读取 `RGROUP_BENCHMARK_ROOT`。因此仅解压源码不能直接重跑 3000 次实验，连入口的 `--help` 都可能先读取历史任务清单。

运行所需资源包括：

- AutoFragDiff 生成环境，以及独立的 CNN 梯度环境。主要版本分别见 `requirements/guidance_generators.versions.json` 与 `requirements/guidance.versions.json`，它们是已有环境的版本记录，不是经过空环境重装的锁文件。
- `vendor/AutoFragDiff`、`vendor/gnina-torch-guidance` 的固定源码及权重；提交号和资源哈希见 [configs/guidance_sources.json](../configs/guidance_sources.json)。上游源码沿用原发布目录的外部获取方式，未复制进仓库。
- `checkpoints/autofragdiff_pocket-gvp.ckpt`；`envs/generators/bin/python` 和 `envs/guidance/bin/python`。梯度环境须能导入 `gninatorch`，原环境使用 `.pth` 指向对应 vendor 目录。
- 原任务中的蛋白、保留核心与参考分子，以及前序实验的记录、参考评分、配对轨迹和技术检查。`cnn_geometry_followup_settings.py` 指定了前序双指标评估与局部几何实验，导入链还读取更早的筛参任务。
- 固定版本的 GNINA 1.1 可执行文件。历史评分模块保留原绝对路径，迁移时须在单独运行副本中调整，或还原目录映射。

可将 `guidance/scripts/` 的源文件同步至已具备上述资源的实验工作区；新工作区则需要先准备对应目录和路径。不要以本发布目录的 `scripts/` 替换历史引导的同名依赖。

全量入口支持 `--run`、`--prepare-only`、`--worker` 和 `--finalize-only`。`--run` 只指定当前输出目录，不会重定向前序输入或固定的 CHECKS 路径。准备新的运行副本后再执行检查，避免向已完成的旧检查目录写入。

## 结果与校验

每组 30 个案例、每案例 100 次尝试。无引导、局部几何约束、加强几何约束、约束投影的双指标改善次数分别为 360、492、494、495；对应 12.00%、16.40%、16.47%、16.50%。成功必须同时满足新设计、结构约束、完整 22 项几何检查和两项评分相对参考分子改善，所有失败保留在分母中。

方案选择只读取有效新分子数和 CNN 改善数，Vina 在选择冻结后评估。这批案例已经用于开发和筛参，结果不作为独立测试集成绩；与 13 模型发布入口的新运行也不能直接合并。

[验收摘要](guidance_validation.json)记录原实验的通过状态、技术检查和统计来源哈希。完整分子、轨迹和评分日志不随代码发布。本次只做源码与导出检查，不重新执行生成或评分。

无需 GPU 或机器学习依赖即可核对代码版本：

```bash
python guidance/verify_sources.py
```

本版截止于周报使用的原生最终重建流程，后续最终核心保持、精度修复、独立复核及其结果均未收录。
