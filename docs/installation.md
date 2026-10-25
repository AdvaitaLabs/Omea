# 安装与资源

## CPU 数据准备和评测

本次运行使用 Python 3.12。安装同一套 RDKit、PoseBusters 和数据处理库，所有模型的最终输出都回到这个环境评分。

```bash
python3.12 -m venv .venv-eval
source .venv-eval/bin/activate
python -m pip install -r requirements/evaluation.txt
```

GNINA 使用已冻结的 **1.1** release，不能悄悄换成另一版本后合并表格。它单独下载，不随源码分发：

```bash
mkdir -p tools
curl -L https://github.com/gnina/gnina/releases/download/v1.1/gnina -o tools/gnina
chmod +x tools/gnina
```

可通过 `--gnina` 或 `GNINA_BIN` 指向自己的安装。Linux 上需要该 release 所需的运行库；本地评分会检查可执行文件并保存其 SHA256。评分固定 `--score_only --no_gpu --scoring vina --cnn_scoring rescore`；每个任务用 2 个 CPU 线程，默认同时 8 个任务。

## 模型生成环境

`benchmark.py models` 和 `configs/methods.json` 标明每个方法对应的环境。版本快照在 `requirements/*.versions.json`，普通 Python 依赖在同名 `.txt`。CUDA 和 PyG 扩展需匹配 PyTorch 的 wheel，不直接混入通用 requirements 文件。

| 环境 | 方法 | 已运行的核心版本 |
|---|---|---|
| generators | AutoFragDiff、DiffDec、DiffSBDD、MolJO-maskfix、PocketXMol、DiffGui-maskfix | PyTorch 2.8.0 / CUDA 12.8，pytorch-lightning 1.9.5 |
| additional | NEAT-POCKET | PyTorch 2.8.0 / CUDA 12.8，lightning 2.5.5；带 varlen 注意力兼容后端 |
| final_baselines | LDDM、MolPilot-inpaint、DELETE-singlepath、Pfizer 两个方法 | PyTorch 2.8.0 / CUDA 12.8，lightning 2.5.5 与 pytorch-lightning 1.9.5 分别保留 |
| flowr | FLOWR.root | PyTorch 2.11.0 / CUDA 13.0，lightning 2.6.5；上游独立环境 |
| delete_surface | DELETE 的表面预处理 | Python 3.7、PyMesh2 0.3、NumPy 1.21.6、SciPy 1.7.3 |

前三个生成环境的安装形式如下，把环境名称和 requirements 文件替换成需要的那一组：

```bash
python3.12 -m venv .venv-generators
source .venv-generators/bin/activate
python -m pip install torch==2.8.0 torchvision==0.23.0 --index-url https://download.pytorch.org/whl/cu128
python -m pip install torch_scatter==2.1.2+pt28cu128 torch_cluster==1.6.3+pt28cu128 -f https://data.pyg.org/whl/torch-2.8.0+cu128.html
python -m pip install -r requirements/generators.txt
```

`final_baselines` 还用到同一 wheel 索引中的 `torch_sparse==0.6.18+pt28cu128`。FLOWR.root 的依赖较多，先通过 `setup-sources --model FLOWR.root` 获取固定源码，按其 `pyproject.toml`/`uv.lock` 建立独立环境，再检查快照，最终评分仍使用公共 CPU 环境。

这些快照记录已有运行环境，并不是宣称所有组合都已在空机器上重新安装验证。上游的 LDDM metadata 限制 Python 3.11 并使用较旧 RDKit；本适配器在记录的 Python 3.12 环境中通过源码路径加载，不能直接执行其带依赖的 `pip install -e` 后期待保留公共环境版本。遇到模型依赖冲突时，单独建立它的生成环境，保持评测环境不变。

## 源码和权重

`setup-sources` 按 `configs/sources.json` 的 commit 获取源码，并应用 `patches/` 里的记录补丁。已有目录存在未记录改动时会报错，不自动覆盖。`--verify` 只核对，`--cache /path/to/upstream-repos` 可利用本地 Git 缓存。

`assets download --model ...` 只获取所选方法的官方资源，`assets verify --model ...` 检查全部所需权重与辅助文件的 SHA256。Google Drive 或 Figshare 若返回 HTML、限流或失败，请从 `configs/assets.json` 的官方来源下载到指定位置，再 verify；不会把下载错误当成模型生成失败。

NEAT 的 checkpoint 通过 Figshare ZIP 字节范围提取，PocketXMol 需要 release 中的训练配置，Pfizer 只选固定的 cutable 权重，不按 benchmark 分数挑 checkpoint。

## DELETE 表面预处理

DELETE 使用原生 MaSIF/MSMS/APBS/PARSE 表面特征，不能用其他模型的点云输入替代。需要另建 Python 3.7/PyMesh 环境，参考固定 DELETE 仓库的 `surf_maker_environment.yml`。MSMS、APBS、PDB2PQR 和 multivalue 按上游要求安装在工作目录的 `tools/delete/` 下；本仓库不打包这些二进制。

```bash
# 在公共 Python 环境导出保留核心的坐标，文件不含原 R 基坐标
python scripts/export_delete_inputs.py

# 切换到 DELETE 表面环境，使用原生特征生成脚本
/path/to/delete-surface-python scripts/prepare_delete_surfaces.py
```

脚本默认查找 `tools/delete/msms.x86_64Linux2.2.6.1`、`tools/delete/pdb2pqr-linux-bin64-2.1.1/pdb2pqr` 以及该目录下的 `bin/apbs` 和 `multivalue`。每个案例输出 `data/delete_surfaces/<case>/pocket.ply` 并保存输入及二进制校验值。完成后再运行 DELETE 生成。

## 检查

无需 GPU 或权重就能检查语法和指标规则。单元检查另外需要 CPU PyTorch。

```bash
python -m compileall -q benchmark.py scripts tests
python -m unittest discover -s tests -v
python benchmark.py validate
```

## CNN 引导实验

`guidance/` 保留周报版本的历史实验脚本，使用单独的生成和梯度环境，环境版本、源码来源、前序输入及路径要求见 [引导优化说明](guidance.md)。上面的 `benchmark.py` 安装命令只适用于原 13 模型 benchmark，不会自动准备引导实验的历史输入。
