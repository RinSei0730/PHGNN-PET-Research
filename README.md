# PHGNN + 原型引导的缺失 PET 提示研究

独立研究方向：以 PHGNN 为基座，在模态编码进入患者超图主干前，补充自然缺失的 Amyloid/Tau PET 内部表示，并通过消融实验判断原型库是否比邻居映射、普通个体映射更有帮助。

当前研究不使用 V4 的临床辅助监督、四类疾病阶段预训练、Ridge 融合或分数后处理。该仓库保存代码、实验设置和汇总结果；原始医疗数据、患者级预测、划分行号和包含患者输入的检查点仅保留在本地。

## 已完成实验

2026-10-15 完成 NACC AD/CN 第一轮：10 个预定条件 × 3 个种子（11、22、33）× 5 折，共 150 次训练，独立本地复核通过。

| 条件 | 平均 AUC | 平均 BACC | Amy/Tau 均缺失子组 AUC |
|---|---:|---:|---:|
| 本轮 PHGNN 基线 | 0.9349 | 86.40% | 0.9225 |
| 仅在观测 PET 节点构建 PET 超边 | 0.9386 | 86.62% | 0.9245 |
| 邻居 PET 映射 | 0.9365 | 86.90% | 0.9232 |
| MRI/人口学拼接查询的原型模块 | 0.9382 | 86.71% | 0.9246 |
| 相同输入、无原型的仿射映射 | 0.9375 | 86.96% | 0.9223 |

原型模块相对本轮基线有小幅平均增益，但 AD 敏感度由 85.26% 降至 84.01%，原型库相对普通映射的独立贡献仍需验证。图构建消融也有竞争力。当前不能宣称原型机制稳定有效或已达到 ICMR/ICME 论文要求。

完整的全部条件、负增益和限制见 [第一轮实验结论](reports/round01/实验结论.md)，机器可读统计见 [summary.json](reports/round01/summary.json)。15 折不是 15 份独立人群证据；该队列已在此前研究中查看过。

## 文件结构

- `src/pet_prompt.py`：生成器、供体池、模拟遮挡、重构评估及固定下游图的适配。
- `src/pipeline/`：NACC ROI 特征处理、训练折标准化、PHGNN 预训练与提示训练。
- `src/run_experiments.py`：预定条件运行、指标、完整本地工件和冻结协议保存。
- `tests/test_pipeline.py`、`tests/test_pet_prompt.py`：共 20 个无需真实数据的合成 CPU 检查。
- `reports/round01/`：仅含汇总结果、汇总复核报告、去路径协议和来源哈希。
- `docs/`：实验边界和文献核验。
- `run.ps1`：接受本机 Python 与队列路径的可移植 PowerShell 入口。

## 数据需要自己准备

本代码包不含 NACC/ADNI 数据或数据下载授权。真实训练需要有使用权限的、与当前 loader 字段兼容的处理后 `cohort.npz`。仅下载任意 NACC 表格不能直接运行本训练入口。

本轮原始队列为 6027 人；开发节点 4797 人，AD/CN 有标签开发患者 3379 人，未标注背景 1418 人。选择的模型输入为 MRI 240 维、Amyloid PET 45 维、Tau PET 43 维及人口学 7 维，共 335 维；编码后为 4×32 维，再加 2 个 PET 观测标记。临床量表不作为本轮输入。

队列字段及特征转换见 [data.py](src/pipeline/data.py)。MRI 使用 FreeSurfer 区域数值，PET 使用已提供的 SUVR 等数值；当前入口不是影像分割或原始 DICOM 预处理程序。本发布副本也不包含原项目的数据构建脚本。

## 软件

本轮实测 Python 3.12.10、NumPy 2.3.5、SciPy 1.15.3、scikit-learn 1.9.1、PyTorch 2.7.0+cu128，GPU 为 RTX 4070 Ti。CUDA 构建为 12.8，执行设置和源码 SHA 保存在 [协议派生副本](reports/round01/protocol_public.json)。

`requirements.txt` 记录实测核心依赖；PyTorch CPU/CUDA wheel 应按本机情况安装。合成 CPU 检查不需要 CUDA；复跑本轮 GPU 协议需匹配相应 CUDA PyTorch 环境。本仓库没有包含 Python 环境，也不会自动安装软件。

## 先运行无需数据的检查

在仓库根目录、已安装依赖的 Python 环境中执行：

```powershell
python -X utf8 -B tests/test_pipeline.py
python -X utf8 -B tests/test_pet_prompt.py
```

每个文件各 10 项检查，涵盖拟合边界、人工遮挡、邻居供体、观测 PET 保留、图归一化及提示检查点重放等。它们使用合成数据；通过这些检查不等于复现原 150 次医疗队列实验。

## 运行自己的数据实验

复制的冻结源码保持原始字节和 SHA。`src/run_experiments.py` 仍保留作者本机的旧默认队列路径，**使用时必须用 `--cohort` 明确覆盖**。

```powershell
python -X utf8 -B -u src/run_experiments.py --cohort "/自己的/cohort.npz" --out results/new_round
```

Windows 也可使用参数化入口：

```powershell
./run.ps1 -PythonPath ".venv/Scripts/python.exe" -CohortPath "data/cohort.npz"
```

默认是 10 个条件、seed 11/22/33、5 折、CUDA。查看实际配置而不训练：

```powershell
python -X utf8 -B src/run_experiments.py --cohort "/自己的/cohort.npz" --out results/new_round --check-only
```

保持数据、源码和实验参数不变后断点续跑：

```powershell
python -X utf8 -B -u src/run_experiments.py --cohort "/自己的/cohort.npz" --out results/new_round --resume
```

PowerShell 入口可对应加 `-Resume`。需要 CPU 时，可直接使用 Python 入口加 `--device cpu`；CPU 与 GPU 数值不保证逐位一致。

训练产生的 `results/new_round` 包含患者预测、患者划分和带完整节点输入的检查点，已加入 `.gitignore`。这些文件用于本地研究，不要手动强制提交。

## 本地完整结果审计

`tests/verify_round.py` 保留原始本地审计源码；它需要真实队列、原始冻结 `protocol.json`、本地 `origin_manifest.json` 与其历史文件，以及该轮的 `records.json`、`predictions/`、`checkpoints/`、`split_audits/` 等完整工件。本发布包不含这些工件，所以不能在仅下载仓库后直接完成该审计。`tests/write_round_summary.py` 同样依赖完整本地逐折记录。

`reports/round01/verification_independent.json` 是本轮完整本地工件已经通过审计的汇总证据。去路径 `protocol_public.json`、`provenance_public.json` 明确保留原始文件 SHA，供追溯使用；它们不替换原协议、不用于恢复原检查点。

## 方法与实验边界

标准化、PCA 和模态编码器的拟合遵循本轮训练折范围。生成器拟合和 PET 供体来自训练患者加未标注背景，排除验证和测试；早停、主干配置及阈值由验证集确定。测试标签不参与训练和选择。

这仍是传导式图：验证和测试患者的无标签特征参与图构建和传播。当前检查点的同图重放不等于实现了独立新患者部署。模块生成的是 32 维内部 PET 表示，不是真实 PET 图像或可替代临床检查。

详细边界见 [实验设计审计](docs/audit_experiment_design.md)，论文归因见 [文献核验](docs/references_check.md)，方法出处与许可状态见 [THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md)。
