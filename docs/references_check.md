# 本轮附件的文献核验（2026-10-15）

这份记录核验附件中的论文身份、方法归属和能否支持我们的实验主张。它不是新的模型设计要求；当前仍先验证 PHGNN 加缺失 PET 提示模块。已验证论文存在，不等于该论文证明我们的方案会提分。未能验证也不表示论文不存在。

## 已验证，但附件有错误归因

### PHGNN：正式版没有放弃 OASIS-3

正式发表版《Diagnosing Alzheimer’s disease using hypergraph neural networks with prompt tuning》，Pattern Recognition 176 (2026), 113290，第 4.4 节明确进行了 OASIS-3 的 AD/NC 外部验证，并讨论了约 90% PET 缺失。附件“因为 PET 缺失太多而直接放弃 OASIS-3”的表述不能用作研究动机。我们的动机应来自 NACC 实验中对真实缺失 PET 的处理方式及改进空间，而不是把原论文写成没有外部验证。

来源：[出版社正式版](https://www.sciencedirect.com/science/article/abs/pii/S0031320326002554)。

### PRA-PoE：原型是查询锚点，并非我们的 K 个 PET 原型库

论文真实存在。第 2.3 节为每个目标模态定义一个全局原型和观测/缺失词元，构造查询；键和值来自同一患者的其他模态，缺失来源被屏蔽。观测模态也经过门控校准，随后全体表示做 LayerNorm。第 2.4–2.5 节使用高斯专家 PoE、不确定性融合，以及交叉熵加 KL。它可以启发“区分观测与生成表示、屏蔽缺失来源”，但其注意力对象、原型数量、门控作用位置均不同于我们的人群邻居 PET 提示模块，不能称为直接复现；也不证明我们的超图、残差或 PET 原型组合必然提高分类。

来源：[作者论文第 2 节](https://arxiv.org/html/2605.13081v2)。

### CoPoE：确实存在，但不是 PET 表征生成方法

《CoPoE: Multimodal Fusion via Decomposable Disease-Coordinate Product-of-Experts for Missing-Modality Alzheimer’s Diagnosis》，Chihun An、Ikbeom Jang，arXiv:2610.11394，2026-10-08 提交。

实际方法将基因、体液、影像、临床输入映射成 R/P/N/S 四块高斯专家，只融合真实观测到的专家，缺失模态不产生插补专家。PET 不作为融合输入；PET 定义的淀粉样和 tau 阳性标签监督 P 块读出。其批内 InfoNCE 对齐的是影像专家的 N 块、体液专家的 P 块与对应融合块，不是“生成 PET 向量与真实 PET 向量”。去掉体液输入后病理读出仍有信号，不能改写成“仅靠 MRI 和人口学便能准确还原 PET”，更不是证明个体 ROI PET 可辨识。

来源：[arXiv 身份记录](https://arxiv.org/abs/2610.11394)、[方法第 III-D 节及限制](https://arxiv.org/html/2610.11394v1)。

## 部分已验证，指定身份或机制未能验证

### HiBrain：找到的是 KDD 2026 的层级原型脑图论文

作者网站确认《HiBrain: Hierarchical Prototype Learning on Multimodal Brain Graphs for Stage-Aware Biomarker Discovery》，Ren 等，KDD 2026。本轮未核实附件所称“Ma 等、OpenReview、ICLR 2027 审稿”的对应论文，也未验证解剖支持度校准、Ref. Softmax、条件准入控制属于这篇 HiBrain。ACM 页面在本轮访问失败，未把作者网站摘要当成全文机制证据。因此当前不据此引入参考 softmax，也不把“无邻居回退全局原型均值”归于 HiBrain。

来源：[第一作者论文列表](https://www.jingren.org/publications)、[合作者对框架的介绍](https://danleneurocom.github.io/)。可定位的 DOI 为 10.1145/3770855.3818900；指定 OpenReview 记录仍待补齐。

### 人群图审计：相关论文存在，日期、首作者和适用场景需修正

找到《Auditing Site-Dependent Performance in Transductive Population Graph Neural Networks for Multisite Autism fMRI》，Wan、Chen 等，bioRxiv 2026.09.15.751719 v2，页面标注 2026-09-23，研究的是多中心自闭症 fMRI，不是 AD 的 MRI/PET。其作者仓库当前 R3 记录指出，单向和双向图得到不同效果，结论依赖消息方向及协议。本轮未核实附件中 P0/P1 精确定义及“静态图 AUC 稳定提高 +0.022～+0.035”的原始位置，不能把这组数字搬到 NACC。论文全文端点访问失败，仓库可读。

来源：[bioRxiv 记录](https://www.biorxiv.org/content/10.64898/2026.09.15.751719v2)、[作者审计代码与当前结果限制](https://github.com/mindloop-lab/transductive-population-graph-audit)。

## 已验证，但不支持“必须加入且必然提分”

### SimpleShot：2019 年的少样本最近邻方法

SimpleShot（Wang 等，arXiv:1911.04623）是 2019 年工作，不属于全部 2025–2026 最新文献。它研究冻结图像特征的少样本最近邻/类中心分类，比较均值中心化、L2 归一化等变换。这不是所有原型分类器必须执行 CL2N 的定理，更没有证明我们的 pMCI/sMCI AUC、BACC 会立刻提高。若以后做距离原型头，应分别测试原特征、仅 L2、CL2N；均值只从预先指定的训练/参考集合估计，并从变换后的训练样本重算类别原型。

来源：[作者论文记录及摘要](https://arxiv.org/abs/1911.04623)。

## 对当前实验的实际影响

- 先把新研究的模型边界固定为 PHGNN 主干加缺失 PET 表征提示。附件提出的四类疾病原型、状态词元、门控、CL2N 和对比损失均为独立候选，不全部叠加。
- 区分患者级原始 PET、PET 编码和生成的内部表示。当前模块不输出医学可用的 PET 图像，也没有证明 PET 病理恢复。
- 淀粉样和 tau 分别使用真实观测掩码；模拟遮住的 PET 必须同时从供体池和相关 PET 关系来源移除，患者不得看到自己的重构答案。
- 类似“生成一定落在真实 PET 流形”“四类训练一定解耦疾病方向”“冻结提示精准捕捉 MCI 进展”均应写成待检验假设。
- 超图拓扑、特征统计拟合集、供体集和早停集合必须在协议中分别列出。测试特征参与传导图须披露，不能称为纯训练图。测试标签不用于训练或早停；已经查看过的测试集仍属于开发证据，后续正式验证需预留独立证据。
- 新增残差时保留相同输入和容量的无原型对照；新增对比损失时先确认批内数量、同疾病患者被当负例的影响和 PET 重构幅度。只用验证集选择候选，再进行固定实验。

本轮没有新增检索综述，也没有因这些文献修改训练代码。以上列出需要纠正的引用和可实验检验的启示，优先完成实际对照结果。
