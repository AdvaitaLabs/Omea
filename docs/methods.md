# 方法与引用

这里列出本 benchmark 的 13 个方法及实际运行形式。所有 checkpoint 冻结，不重新训练。文章链接与上游版本同时保存在 configs/methods.json。

| 方法 ID | 文章时间 | 发表处 | 源码许可 | 权重许可 |
|---|---|---|---|---|
| AutoFragDiff | 2023 | [NeurIPS GenBio Workshop (not main conference)](https://openreview.net/pdf/544ba60bdaf4d07e80e3da447013fbc9139e06ee.pdf) | [MIT](https://github.com/keiserlab/autofragdiff/blob/84f0885cb12e6ac4abc7558870f8d304c78c8a38/LICENSE) | no separate checkpoint terms located |
| DiffDec | 2024 | [Journal of Chemical Information and Modeling](https://doi.org/10.1021/acs.jcim.3c01466) | [MIT](https://github.com/biomed-AI/DiffDec/blob/916ae14207b2783a90336bb8509374535c5791f9/LICENSE) | CC-BY-4.0 |
| DiffSBDD | 2024 | [Nature Computational Science](https://doi.org/10.1038/s43588-024-00737-x) | [MIT](https://github.com/arneschneuing/DiffSBDD/blob/5d0d38d16c8932a0339fd2ce3f67ade98bbdff27/LICENSE) | CC-BY-4.0 |
| DELETE-singlepath | 2025 | [Nature Machine Intelligence; original preprint 2023](https://doi.org/10.1038/s42256-025-00997-w) | [MIT](https://github.com/OdinZhang/Delete/blob/ba57693822ac82950bf0aa081b788772039373de/LICENSE) | CC-BY-4.0 |
| MolJO-maskfix | 2025 | [ICML](https://proceedings.mlr.press/v267/qiu25h.html) | [CC-BY-NC-SA-4.0](https://github.com/GenSI-THUAIR/MolCRAFT/blob/743dfb5198c44ca0cec180822bd54258f9d386f1/MolJO/LICENSE) | predictors: CC-BY-NC-SA-2.0; backbone: separate terms not located |
| MolPilot-inpaint | 2025 | [ICML](https://proceedings.mlr.press/v267/qiu25g.html) | [CC-BY-NC-SA-4.0](https://github.com/GenSI-THUAIR/MolCRAFT/blob/743dfb5198c44ca0cec180822bd54258f9d386f1/MolPilot/LICENSE) | CC-BY-NC-SA-2.0 |
| DiffGui-maskfix | 2025 | [Nature Communications](https://doi.org/10.1038/s41467-025-63245-0) | [MIT](https://github.com/QiaoyuHu89/DiffGui/blob/f2f90b560ad32db9d584a3b7b153420dc1e8d258/LICENSE) | no separate checkpoint terms located |
| PocketXMol | 2026 | [Cell](https://pubmed.ncbi.nlm.nih.gov/41713417/) | [MIT](https://github.com/pengxingang/PocketXMol/blob/65488cf635c856101dbe703ac97e2f10f58e005c/LICENSE) | CC-BY-4.0 |
| Pfizer-Coupled-flow | 2026 | [Digital Discovery](https://doi.org/10.1039/D5DD00535C) | [Apache-2.0](https://github.com/pfizer-opensource/pilot-sbdd/blob/5b7e668ea88baac175536d804599e59236b60afd/LICENSE) | CC-BY-4.0 |
| Pfizer-Coupled-diffusion | 2026 | [Digital Discovery](https://doi.org/10.1039/D5DD00535C) | [Apache-2.0](https://github.com/pfizer-opensource/pilot-sbdd/blob/5b7e668ea88baac175536d804599e59236b60afd/LICENSE) | CC-BY-4.0 |
| FLOWR.root | 2026 | [Nature Communications](https://doi.org/10.1038/s41467-026-74130-9) | [MIT](https://github.com/jule-c/flowr_root/blob/739a33c2aab74d074dd24605b9a2e34dc809ae9c/LICENSE) | no separate checkpoint terms located |
| NEAT-POCKET | 2026-09 | [arXiv preprint](https://arxiv.org/abs/2609.05097) | [MIT](https://github.com/molinfo-vienna/NEAT-POCKET/blob/ce7de3f23d28b99f5316903220d4bacfa2757463/LICENSE) | MIT |
| LDDM | 2026-09 | [bioRxiv preprint](https://www.biorxiv.org/content/10.64898/2026.09.15.751537v1) | [MIT](https://github.com/LPDI-EPFL/lddm/blob/8fac7133c29ccdce576eb38bf7d43c09df9cb498/LICENSE.md) | CC-BY-NC-4.0 |


权重许可栏中的“no separate checkpoint terms located”表示没有从已收集材料找到单独权重授权，不能从源码 MIT 推导权重也可商用。MolJO/MolPilot 权重和 LDDM 当前 checkpoint 有明确非商业条款。

## 适配说明

- **AutoFragDiff**：直接调用官方单片段 denoiser，提供核心、连接原子与 R 基原子数；未做生成后核心图修复。
- **DiffDec**：原生 10 槽加 dummy 输出，尺寸由统一审计判断，不改 dummy logits。
- **DiffSBDD**：原生 inpaint，用保留核心选口袋；50 timesteps、20 resamplings 是单次原生采样内部参数，不是生成后 best-of-K。
- **DELETE-singlepath**：每次尝试只走一条自回归路径，单位置/元素提案，原生早停计入失败。
- **MolJO-maskfix**：修正核心状态保持；保留官方 Vina/SA critic 引导，未再训练。
- **MolPilot-inpaint**：增加核心坐标、类型、内部键和非连接核心边的推理条件；无额外性质引导。
- **DiffGui-maskfix**：修正固定核心 halfedge mask 和最终核心条件保持；使用官方性质条件和 guidance。
- **PocketXMol**：官方 maskfill 预设，单固定核心与已知连接位点，使用官方训练编码。
- **Pfizer-Coupled-flow**：固定 cutable kinase 权重；原生 flow 参数，采样前设置非连接核心边为已知 NOBOND。
- **Pfizer-Coupled-diffusion**：固定 cutable kinase 权重；补充核心电荷保持及复制采样器的非连接边掩码绑定。
- **FLOWR.root**：原生 fragment growing；保留原生 final inpaint，不使用补采样和生成后筛选。
- **NEAT-POCKET**：原生逐原子停止；限制蛋白氢松弛迭代；PyTorch 2.8 注意力兼容。
- **LDDM**：原生 design 模式及虚拟节点，加入单连接位点 NOBOND 条件；最终尺寸仍做统一审计。


不同模型对核心、连接位点和尺寸的原生支持不同。本任务统一最终验收规则，不能把这些适配结果直接当成各原论文的原生实验结果。采样参数以适配器和输出 config.json 为准。批大小与种子策略沿用历史设置；不同模型不宣称使用同一条随机噪声轨迹。

## 固定上游版本

| 源码目录 | commit |
|---|---|
| [AutoFragDiff](https://github.com/keiserlab/autofragdiff) | 84f0885cb12e6ac4abc7558870f8d304c78c8a38 |
| [DiffDec](https://github.com/biomed-AI/DiffDec) | 916ae14207b2783a90336bb8509374535c5791f9 |
| [DiffGui](https://github.com/QiaoyuHu89/DiffGui) | f2f90b560ad32db9d584a3b7b153420dc1e8d258 |
| [MolCRAFT](https://github.com/GenSI-THUAIR/MolCRAFT) | 743dfb5198c44ca0cec180822bd54258f9d386f1 |
| [NEAT-POCKET](https://github.com/molinfo-vienna/NEAT-POCKET) | ce7de3f23d28b99f5316903220d4bacfa2757463 |
| [PocketXMol](https://github.com/pengxingang/PocketXMol) | 65488cf635c856101dbe703ac97e2f10f58e005c |
| [flowr_root](https://github.com/jule-c/flowr_root) | 739a33c2aab74d074dd24605b9a2e34dc809ae9c |
| [LDDM](https://github.com/LPDI-EPFL/lddm) | 8fac7133c29ccdce576eb38bf7d43c09df9cb498 |
| [PfizerCoupled](https://github.com/pfizer-opensource/pilot-sbdd) | 5b7e668ea88baac175536d804599e59236b60afd |
| [DELETE](https://github.com/OdinZhang/Delete) | ba57693822ac82950bf0aa081b788772039373de |
| [DiffSBDD](https://github.com/arneschneuing/DiffSBDD) | 5d0d38d16c8932a0339fd2ce3f67ade98bbdff27 |
