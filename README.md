# R-group design benchmark

比较 13 个分子生成方法在蛋白口袋中的单连接位点 R 基替换能力：保留其余原子的结构和坐标，只重新生成指定位置的一个 R 基。模型使用官方预训练权重，统一检查结构约束、几何质量和相对原配体的代理分数。

本任务参考 DiffSBDD 的骨架保持设计思路，是严格单连接位点的扩展任务。它不是论文原始实验的逐项复现，也不把代理分数当作真实 ΔΔG。

## 测试设置

- 固定 45 个案例，每个案例每个方法 **100 次原始生成**。案例是蛋白结构、原配体和一个指定替换位点的组合。
- 主表使用共同的 30 个多原子 R 基案例，覆盖 7 个靶标组。另有单原子 R 基案例；全部案例也单独汇总。案例列表在 `cohorts/single_site_45.json`。
- 保持核心的元素、形式电荷、芳香性、内部键、立体化学和蛋白坐标系中的位置；只允许在指定核心原子处连接一个 SINGLE 键。R 基重原子数与原被删除部分相同。
- 失败、尺寸错误和重复分子都计入原始尝试数，不补生成、不做 best-of-K。唯一分子数另外报告。
- 输入生成器的是蛋白、保留核心、连接位点和 R 基原子数。原 R 基的元素、键、坐标和评分不会传给生成器。

完整定义在 [configs/protocol.json](configs/protocol.json)。公开数据的训练重叠尚未排除，因此不能把结果称为严格独立测试集上的排名。

## 指标

| 指标 | 含义 |
|---|---|
| Strict success | 化学结构可解析，且核心、连接位点、连通性、原子数和位置都满足约束 |
| Valid new | Strict success，加上完整 PoseBusters 几何检查通过、评分成功，并且与原配体的化学结构不同 |
| Vina | 原生成构象的 Vina 打分，越低越好，单位 kcal/mol；不重新对接 |
| GNINA CNNaffinity | 原生成构象的神经网络亲和力预测，越高越好，输出尺度为 pK |
| Dual improvement | Valid new，且 Vina 比原配体低、CNNaffinity 比原配体高 |

成功率的分母是全部原始尝试。PoseBusters 缺失检查也不能记为通过。所有方法及原配体使用相同的加氢、仅氢原子优化和 CPU GNINA 1.1 评分。核心 RMSD 上限 0.1 Å，单原子最大位移上限 0.3 Å。

同时报告每个案例及靶标等权结果，避免一个靶标的案例多就主导总分。Vina 和 GNINA 是快速代理，不能直接解释成实验亲和力改善或自由能 ΔΔG。本仓库不运行 OpenFE。

## 目录

```text
benchmark.py       统一命令入口
scripts/           数据准备、13 个方法的适配器、生成、评分和汇总
configs/           测试规则、方法信息、官方资源及权重 SHA256
cohorts/           固定测试案例与官方 MOAD split
requirements/      依赖文件与实际运行环境的版本记录
patches/           在固定上游版本上使用的兼容补丁
licenses/          上游许可证原文
tests/             核心约束、固定预算、保存和评测规则的检查
docs/              安装、方法适配说明和已知问题
```

蛋白结构、权重、第三方完整仓库、生成结果、历史调参和联合引导实验没有放入发布目录。运行时下载的资源及输出由 `.gitignore` 排除。

## 使用

先按 [安装说明](docs/installation.md) 建立 CPU 评测环境和所选模型的生成环境。不同模型需要不同的 Lightning/PyTorch 版本，不建议把它们全部装进一个环境。

```bash
# 查看 13 个方法及对应环境
python benchmark.py models

# 获取固定版本的上游源码与该模型权重
python benchmark.py setup-sources --model AutoFragDiff
python benchmark.py assets download --model AutoFragDiff

# 从公开 MOAD 和 PDB CCD 下载、准备并核对固定案例
python benchmark.py prepare
python benchmark.py validate --unit-tests

# 在该模型的生成环境里运行；默认每案例 100 次
python benchmark.py generate --model AutoFragDiff --run benchmark_v1 --cohort primary

# 在 CPU 评测环境里评分。使用 GNINA 1.1
python benchmark.py score --model AutoFragDiff --run benchmark_v1 --gnina /path/to/gnina-1.1
python benchmark.py summarize --run benchmark_v1 --models AutoFragDiff
```

上述命令显式选择 30 个共同案例，所有模型都应使用相同 `--cohort primary`。默认 `--cohort all` 是 45 个准备案例，`--cohort single_atom` 是 14 个单原子案例。FLOWR.root 的已知不支持输入 1RVX 在主比较集以外；对全 45 个案例直接运行它会提前报错，不虚构未执行的尝试数。

如果已有权重，可按 `configs/assets.json` 的路径放好，再运行 `assets verify --model ...`。如果已有完整 MOAD/CCD 缓存，用 `prepare --from-cache`；已有任务池用 `prepare --freeze-only`。

默认资源和输出放在本仓库目录。也可设置 `RGROUP_BENCHMARK_ROOT=/path/to/workspace`，把数据、上游源码、工具、权重和结果放在独立工作目录；命令入口可以从任意工作目录执行。任务清单内部的相对路径以该工作目录为根。

小规模安装检查必须显式标记，不能并入正式结果：

```bash
python benchmark.py generate --model AutoFragDiff --run smoke_v1 --smoke --limit 1 --attempts 2
python benchmark.py score --model AutoFragDiff --run smoke_v1 --gnina /path/to/gnina-1.1
python benchmark.py summarize --run smoke_v1 --models AutoFragDiff --smoke
```

完成的案例支持续跑；中断的案例从原种子完整重跑。输入、代码、权重或预算变化时必须换 run 名称。运行清单记录输入、代码与权重校验值，并保存原始分子的完整 RDKit 状态，SDF 仅作为导出格式。

汇总输出在 `results/<run>/summary/`：`per_case.csv`、`comparison_primary_30.csv`、`comparison_all_cases.csv`、`per_target_primary.csv` 和 `summary.json`。`unique_*` 在总表里是各案例唯一数之和，不是跨口袋去重的分子总数。

## 方法和使用范围

模型、文章、许可证和适配说明见 [docs/methods.md](docs/methods.md)。MolJO、DiffGui、MolPilot 和 DELETE 的实际适配形式会在配置与输出中保留标识，不作为未经修改的原生采样器宣传。

这份整理采用新的输出布局，且修正了发布代码中已定位的 Pfizer diffusion 掩码绑定问题。历史表格没有随代码重新计算，也没有放进本仓库。已有结果不能直接混入这里的新运行；说明见 [docs/known_issues.md](docs/known_issues.md)。

许可按文件范围区分：独立的 benchmark 调度、数据准备和评测代码使用 MIT；上游适配与补丁保留对应许可证，特别是 MolJO/MolPilot 的非商业限制。权重许可独立于源码许可，详见 [THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md)。
