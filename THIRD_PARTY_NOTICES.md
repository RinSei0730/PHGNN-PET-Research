# 方法出处与发布范围

本仓库是 NACC ROI 特征上的本地 PHGNN 方法重实现及独立缺失 PET 生成器研究，不是原作者官方代码仓库，也不声称复现了论文的全部影像预处理、数据协议或表格数值。

## PHGNN

- 论文：*Diagnosing Alzheimer’s disease using hypergraph neural networks with prompt tuning*，Pattern Recognition 176 (2026), 113290。
- [出版社页面](https://www.sciencedirect.com/science/article/abs/pii/S0031320326002554)
- [原作者仓库](https://github.com/theguardsgod/PHGNN)
- 本地历史对照的官方参考提交：`de4469007d2b388b185177da82061ed312e878db`。

本地实现保留 HGNN 主干、遮挡重构预训练和患者超图提示的研究框架。ROI 模态编码、NACC 队列与任务划分、训练折拟合预处理、验证集选择、确定性 CUDA 稀疏运算及提示检查点一致性属于本地适配；缺失 PET 原型库及消融生成器属于本轮研究实现。原作者的 Notebook、VAPFormer 文件和论文 PDF 未包含在本发布包。

当前原型为可学习的内部基向量；不把它描述为已证实的疾病亚型或真实 PET 病理恢复。其他论文对方案的启示和不适用之处见 [文献核验](docs/references_check.md)，不把启发称为完整复现。

## 许可状态

本地参考快照未发现原作者仓库的统一 LICENSE。当前存档未添加覆盖整个项目的 MIT 或其他未经核验的统一许可证。方法出处与署名不等于为第三方内容补授许可；如需公开再分发、商业使用或形成正式开源许可，仍需确认具体代码来源和适用条款。

## 数据与工件

NACC/ADNI 原始数据、处理后队列、患者级标签与预测、患者划分行号、完整患者图和包含患者输入的模型检查点未包含在仓库中。数据访问权限和使用条款由各数据提供方规定。
