"""Write the round-one report only after 150 runs and independent verification.

This script reads completed artifacts and never trains, ranks or selects arms.
Run it explicitly after tests/verify_experiment.py (or the independent verifier)
has produced results/round01_core/verification_independent.json.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
from statistics import fmean, pstdev


ROOT = Path(__file__).resolve().parents[1]
ARMS = (
    "baseline_naive", "baseline_observed", "neighbor", "prototype_avg",
    "prototype_concat", "mlp_concat", "prototype_residual",
    "prototype_pattern", "prototype_hypergraph", "prototype_sce_nce",
)
LABELS = {
    "baseline_naive": "PHGNN 新基线（原构图）",
    "baseline_observed": "PHGNN（PET 只在观测节点构图）",
    "neighbor": "邻居 PET 映射",
    "prototype_avg": "原型：MRI/人口学先取平均",
    "prototype_concat": "原型：MRI/人口学分别拼接",
    "mlp_concat": "无原型的仿射个体映射",
    "prototype_residual": "拼接原型＋个体残差",
    "prototype_pattern": "拼接原型＋A/T 缺失模式",
    "prototype_hypergraph": "拼接原型＋超图供体聚合",
    "prototype_sce_nce": "拼接原型＋SCE/InfoNCE 训练",
}
METRICS = ("auc", "bacc", "sen", "spe", "val_auc", "auc_pet", "auc_nopet")
COMPARISONS = (
    ("prototype_concat", "prototype_avg", "分别拼接是否优于取平均"),
    ("prototype_concat", "neighbor", "原型是否优于邻居映射"),
    ("prototype_concat", "mlp_concat", "原型是否优于无原型个体映射"),
    ("prototype_residual", "prototype_concat", "个体残差的增量"),
    ("prototype_residual", "mlp_concat", "原型＋残差是否优于无原型映射"),
    ("prototype_pattern", "prototype_concat", "A/T 缺失模式的增量"),
    ("prototype_hypergraph", "prototype_concat", "超图供体聚合的增量"),
    ("prototype_sce_nce", "prototype_concat", "SCE/InfoNCE 训练的增量"),
)


def read_json(path: Path):
    with path.open("r", encoding="utf-8-sig") as handle:
        return json.load(handle)


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def require(condition: bool, message: str):
    if not condition:
        raise RuntimeError(message)


def finite(value) -> bool:
    return isinstance(value, (float, int)) and math.isfinite(value)


def same_float(a, b) -> bool:
    return finite(a) and finite(b) and math.isclose(a, b, rel_tol=1e-10, abs_tol=1e-10)


def paired(left, right, metric):
    return [left[key][metric] - right[key][metric] for key in sorted(left)]


def counts(values):
    return (sum(value > 0 for value in values),
            sum(value == 0 for value in values),
            sum(value < 0 for value in values))


def seed_deltas(left, right, seeds, metric):
    return {seed: fmean(left[key][metric] - right[key][metric]
                        for key in sorted(left) if key[0] == seed) for seed in seeds}


def seed_text(values, scale=1.0, precision=4):
    return "；".join(f"{seed}: {value * scale:+.{precision}f}" for seed, value in values.items())


def gate(results: Path):
    """Require exact completed protocol, matching summaries and fresh audit."""
    paths = {name: results / f"{name}.json" for name in
             ("protocol", "records", "summary", "verification", "verification_independent")}
    for name, path in paths.items():
        require(path.is_file(), f"Missing {name}: {path}. No report was written.")
    data = {name: read_json(path) for name, path in paths.items()}
    protocol, records, summary = data["protocol"], data["records"], data["summary"]
    require(protocol.get("expected_runs") == 150, "Report is restricted to the predefined 150-run round.")
    require(protocol.get("tasks") == ["ad_cn"], "This report expects AD/CN only.")
    require(protocol.get("arms") == list(ARMS), "Arm list/order differs from the predefined round.")
    seeds, folds = protocol.get("seeds"), protocol.get("folds")
    require(seeds == [11, 22, 33] and folds == [0, 1, 2, 3, 4], "Seed/fold protocol changed.")
    require(isinstance(records, list) and len(records) == 150, "All 150 records are required.")
    require(summary.get("complete") is True and summary.get("completed_runs") == 150
            and summary.get("expected_runs") == 150, "summary.json is not complete.")
    require(summary.get("test_results_used_for_arm_selection") is False,
            "Across-arm selection on test scores cannot be described as this exploratory round.")
    for flag in ("clinical_supervision", "disease_stage_pretraining", "ridge_or_score_postprocessing"):
        require(summary.get(flag) is False, f"Unexpected enabled component: {flag}.")
    for name in ("verification", "verification_independent"):
        require(data[name].get("all_passed") is True, f"{name} did not pass.")
        if "completed_runs" in data[name]:
            require(data[name]["completed_runs"] == 150, f"{name} does not describe all 150 runs.")
        if "expected_runs" in data[name]:
            require(data[name]["expected_runs"] == 150, f"{name} expected-run count differs.")
    require(paths["verification_independent"].stat().st_mtime_ns >=
            max(paths["records"].stat().st_mtime_ns, paths["summary"].stat().st_mtime_ns),
            "Independent verification predates final records/summary; rerun the verifier first.")
    # Honor explicit input hashes if the independent verifier publishes them.
    audit = data["verification_independent"]
    for mapping_key in ("input_sha256", "artifact_sha256", "file_sha256"):
        mapping = audit.get(mapping_key, {})
        if isinstance(mapping, dict):
            for name in ("records", "summary", "protocol"):
                wanted = mapping.get(f"{name}.json", mapping.get(name))
                if isinstance(wanted, str):
                    require(wanted == sha256(paths[name]), f"Audited hash differs for {name}.json.")
    grouped = {arm: {} for arm in ARMS}
    for record in records:
        require(record.get("task") == "ad_cn" and record.get("arm") in ARMS,
                "Unexpected task/arm in records.")
        key = (record.get("seed"), record.get("fold"))
        require(key[0] in seeds and key[1] in folds and key not in grouped[record["arm"]],
                "Duplicate or unexpected seed/fold.")
        require(all(finite(record.get(metric)) for metric in METRICS), "Nonfinite or missing metric.")
        grouped[record["arm"]][key] = record
    expected_keys = {(seed, fold) for seed in seeds for fold in folds}
    baseline = grouped["baseline_naive"]
    task_summary = summary.get("tasks", {}).get("ad_cn", {})
    require(set(task_summary) == set(ARMS), "summary.json does not contain all ten arms.")
    for arm, rows in grouped.items():
        require(set(rows) == expected_keys, f"Incomplete split grid for {arm}.")
        item = task_summary[arm]
        require(item.get("completed_splits") == 15, f"Incomplete summary for {arm}.")
        for metric in METRICS:
            values = [r[metric] for r in rows.values()]
            require(same_float(item["metrics"][metric]["mean"], fmean(values))
                    and same_float(item["metrics"][metric]["std_population"], pstdev(values)),
                    f"Summary metric differs from records: {arm}/{metric}.")
            deltas = paired(rows, baseline, metric)
            require(same_float(item["paired_delta_vs_baseline_naive"][metric]["mean"], fmean(deltas)),
                    f"Paired summary differs from records: {arm}/{metric}.")
        actual_counts = counts(paired(rows, baseline, "auc"))
        stored_counts = item["auc_wins_ties_losses"]
        require(actual_counts == tuple(stored_counts[k] for k in ("wins", "ties", "losses")),
                f"Win/tie/loss counts differ for {arm}.")
        for key, record in rows.items():
            require(record["test_rows"] == baseline[key]["test_rows"]
                    and record["test_y"] == baseline[key]["test_y"], "Paired patients/labels differ.")
            if arm != "baseline_observed":
                require(record["graph_sha256"] == baseline[key]["graph_sha256"], "Fixed graph differs.")
            if arm not in ("baseline_naive", "baseline_observed"):
                for split in ("validation", "test"):
                    details = record.get(f"pet_reconstruction_{split}", {}).get("per_modality", {})
                    require(set(details) == {"amy", "tau"}, f"Missing reconstruction: {arm}/{split}.")
                    for modality in ("amy", "tau"):
                        d = details[modality]
                        require(d.get("count", 0) > 0 and all(finite(d.get(k)) for k in
                                ("mean_absolute_error", "cosine_similarity", "prediction_feature_std",
                                 "target_feature_std", "prediction_norm", "target_norm")),
                                f"Incomplete reconstruction diagnostics: {arm}/{split}/{modality}.")
    return protocol, records, summary, data["verification_independent"], grouped


def reconstruction(grouped, arm, split, modality):
    details = [record[f"pet_reconstruction_{split}"]["per_modality"][modality]
               for record in grouped[arm].values()]
    result = {}
    for metric in ("mean_absolute_error", "cosine_similarity", "prediction_feature_std",
                   "target_feature_std", "prediction_norm", "target_norm", "attention_entropy",
                   "attention_effective_prototypes", "attention_max_weight"):
        values = [d.get(metric) for d in details if finite(d.get(metric))]
        result[metric] = fmean(values) if values else None
    ratios = [d["prediction_feature_std"] / d["target_feature_std"]
              for d in details if d["target_feature_std"] > 0]
    result["dispersion_ratio"] = fmean(ratios) if ratios else None
    return result


def number(value, precision=3):
    return "—" if value is None else f"{value:.{precision}f}"


def direction(value):
    return "提高" if value > 0 else ("下降" if value < 0 else "持平")


def build_report(results: Path, protocol, records, summary, audit, grouped):
    seeds = protocol["seeds"]
    baseline = grouped["baseline_naive"]
    lines = [
        "# 第一轮实验结论：PHGNN 加缺失 PET 提示模块",
        "",
        "已经完成 150 次实验：10 个预先指定条件 × 3 个种子（11、22、33）× 5 折。"
        "本轮是同一 NACC 队列上的探索性比较；逐折指标、预测和独立复核已保存。"
        "本报告按预定条件顺序展示，不根据测试成绩选择最终方法。",
        "",
        "## 这轮实际做了什么",
        "",
        "只研究 PHGNN 与独立缺失 PET 表征生成器。没有加入临床辅助监督、四类疾病阶段预训练、"
        "Ridge 融合、分数借用或其他分数后处理。真实已观测 PET 与原可用性标记保留，生成器只替换自然缺失的 PET 编码。"
        "生成器供体和拟合患者来自当前训练折加未标注背景，排除验证、测试与留出患者；"
        "标准化及 PCA 拟合遵循本轮训练折协议。",
        "",
        "除 `baseline_observed` 的预定构图消融外，下游 PHGNN 使用相同的原始超图，"
        "因此 PET 补充与重建分类超图没有同时改变。`prototype_hypergraph` 改的是生成器的 MRI/人口学供体聚合图，"
        "下游分类图仍固定。`mlp_concat` 是两层没有中间激活的线性映射，整体属于仿射个体映射；"
        "它是相同输入的无原型结构对照，参数量并非严格相等，不能称为经过完整调优的非线性 MLP 基线。",
        "",
        "## 分类结果",
        "",
        "AUC 为 0～1；BACC、AD 敏感度 SEN、CN 特异度 SPE 用百分数。"
        "表中是 15 折指标的平均，AUC 的 ± 为折间总体标准差，不是独立样本的置信区间。"
        "无 PET 子组表示淀粉样与 tau PET 均缺失；本轮未输入 FDG，该命名不表示患者绝对没有任何类型 PET。"
        "子组均值也按折计算。",
        "",
        "| 条件 | AUC ± 标准差 | BACC | AD SEN | CN SPE | 无 PET AUC |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for arm in ARMS:
        metrics = summary["tasks"]["ad_cn"][arm]["metrics"]
        value = lambda key: metrics[key]["mean"]
        lines.append(f"| {LABELS[arm]} | {value('auc'):.4f} ± {metrics['auc']['std_population']:.4f} | "
                     f"{100 * value('bacc'):.2f}% | {100 * value('sen'):.2f}% | "
                     f"{100 * value('spe'):.2f}% | {value('auc_nopet'):.4f} |")
    base_metrics = summary["tasks"]["ad_cn"]["baseline_naive"]["metrics"]
    lines.extend([
        "",
        f"新基线无 PET AUC 为 **{base_metrics['auc_nopet']['mean']:.4f}**，AD 敏感度为 "
        f"**{100 * base_metrics['sen']['mean']:.2f}%**。评估缺失 PET 模块时应同时看这两项，"
        "不能只看总体 AUC。BACC 或敏感度变化还受到验证集选择的阈值影响。",
        "",
        "## 相对新基线的配对变化",
        "",
        "每个差值只比较本轮同一 seed/fold 的患者。BACC、SEN、SPE 差值为百分点（pp）。"
        "胜/平/负依据逐折 AUC 的未四舍五入值；种子方向依据该种子的五折平均 AUC 差。",
        "",
        "| 条件 | ΔAUC | ΔBACC pp | ΔAD SEN pp | ΔCN SPE pp | Δ无 PET AUC | AUC 胜/平/负 | 各种子 ΔAUC |",
        "|---|---:|---:|---:|---:|---:|---|---|",
    ])
    for arm in ARMS[1:]:
        rows = grouped[arm]
        delta = {metric: fmean(paired(rows, baseline, metric)) for metric in METRICS}
        win, tie, loss = counts(paired(rows, baseline, "auc"))
        lines.append(f"| {LABELS[arm]} | {delta['auc']:+.4f} | {100 * delta['bacc']:+.2f} | "
                     f"{100 * delta['sen']:+.2f} | {100 * delta['spe']:+.2f} | {delta['auc_nopet']:+.4f} | "
                     f"{win}/{tie}/{loss} | {seed_text(seed_deltas(rows, baseline, seeds, 'auc'))} |")
    lines.extend([
        "",
        "## 预先指定的模块内部对比",
        "",
        "下面每行是左侧条件减去参考条件，不把十个测试分数排序后宣布胜者。"
        "正的平均差仍需看折与种子方向；本轮没有据此宣称显著改进。",
        "",
        "| 研究问题（左侧 − 参考） | ΔAUC | ΔBACC pp | Δ无 PET AUC | ΔAD SEN pp | AUC 胜/平/负 | 各种子 ΔAUC |",
        "|---|---:|---:|---:|---:|---|---|",
    ])
    for arm, control, question in COMPARISONS:
        left, right = grouped[arm], grouped[control]
        delta = {metric: fmean(paired(left, right, metric)) for metric in METRICS}
        win, tie, loss = counts(paired(left, right, "auc"))
        lines.append(f"| {question}：`{arm}` − `{control}` | {delta['auc']:+.4f} | "
                     f"{100 * delta['bacc']:+.2f} | {delta['auc_nopet']:+.4f} | {100 * delta['sen']:+.2f} | "
                     f"{win}/{tie}/{loss} | {seed_text(seed_deltas(left, right, seeds, 'auc'))} |")
    lines.append("")
    for arm in ("prototype_concat", "prototype_residual"):
        delta_auc = fmean(paired(grouped[arm], baseline, "auc"))
        delta_nopet = fmean(paired(grouped[arm], baseline, "auc_nopet"))
        delta_sen = fmean(paired(grouped[arm], baseline, "sen"))
        seed_values = seed_deltas(grouped[arm], baseline, seeds, "auc")
        positives = sum(value > 0 for value in seed_values.values())
        negatives = sum(value < 0 for value in seed_values.values())
        lines.append(f"`{arm}` 相对新基线：总体 AUC {direction(delta_auc)} {abs(delta_auc):.4f}，"
                     f"无 PET AUC {direction(delta_nopet)} {abs(delta_nopet):.4f}，"
                     f"AD 敏感度{direction(delta_sen)} {100 * abs(delta_sen):.2f} 个百分点；"
                     f"种子平均 AUC 方向为 {positives} 个提高、{negatives} 个下降、"
                     f"{len(seeds) - positives - negatives} 个持平。这些是观测结果，不是下一轮的测试择优规则。")
        lines.append("")
    lines.extend([
        "## 生成表示是否接近均值、注意力是否发挥作用",
        "",
        "重构只在本来有 PET 的验证/测试患者上模拟遮住对应 PET 后评估。自然缺失患者没有真实 PET 答案，"
        "因此这些误差不能视为对其真实病理的验证。验证重构参与过生成器 epoch 选择；测试重构只在选定模型后记录。",
        "",
        "下表按模态分别给出 15 折均值。离散度比为每折生成表示跨患者逐维标准差与真实编码相应标准差的比值，再取均值。"
        "接近 0 表示患者间输出差异被压缩，接近 1 表示幅度相近；这不是重构精度指标。"
        "有效原型数是每名患者注意力熵的指数，8 表示在 8 个原型上近似均匀，1 表示单个原型主导。",
        "",
        "| 条件 / 模态 | 验证 MAE | 测试 MAE | 测试余弦 | 验证离散度比 | 测试离散度比 | 测试有效原型数 | 测试最大权重 |",
        "|---|---:|---:|---:|---:|---:|---:|---:|",
    ])
    for arm in ARMS[2:]:
        for modality in ("amy", "tau"):
            val = reconstruction(grouped, arm, "validation", modality)
            test = reconstruction(grouped, arm, "test", modality)
            lines.append(f"| `{arm}` / {modality} | {number(val['mean_absolute_error'])} | "
                         f"{number(test['mean_absolute_error'])} | {number(test['cosine_similarity'])} | "
                         f"{number(val['dispersion_ratio'])} | {number(test['dispersion_ratio'])} | "
                         f"{number(test['attention_effective_prototypes'], 2)} | {number(test['attention_max_weight'])} |")
    lines.append("")
    for modality in ("amy", "tau"):
        d = reconstruction(grouped, "prototype_concat", "test", modality)
        ratio = d["dispersion_ratio"]
        interpretation = ("明显低于真实编码，提示生成表示存在向共同模式收缩的现象" if ratio < 0.5 else
                          "低于真实编码，提示仍有一定收缩" if ratio < 0.9 else
                          "与真实编码相近；还需结合个体误差判断" if ratio <= 1.1 else
                          "高于真实编码；较大的变化本身不等于生成更准确")
        lines.append(f"拼接原型的 {modality}：测试有效原型数为 {number(d['attention_effective_prototypes'], 2)} / 8，"
                     f"最大权重为 {number(d['attention_max_weight'])}；生成/真实离散度比为 {number(ratio)}，"
                     f"{interpretation}。注意力不是完全均匀也可能出现输出接近共同均值，"
                     "还应结合值投影和个体残差检查；较低注意力熵不能证明学到了病理亚型。")
        lines.append("")
    lines.extend([
        "## 能说明什么，以及还不能说明什么",
        "",
        "本轮重新建立了可配对的新基线。旧练习结果 AUC 0.9332 使用 seed 1/2/3、旧预处理及旧提示 checkpoint 处理；"
        "本轮使用 seed 11/22/33、训练折拟合预处理并修正提示 checkpoint 保存。"
        "两批结果不能作逐折配对，也不能把新旧总分差全部归因于 PET 模块。",
        "",
        "本轮仍是传导式图：验证和测试患者的无标签特征参与图传播，标签不用于训练和早停。"
        "旧 NACC 队列及留出中心已经在此前研究中查看；换种子或建新目录不会变成独立外部验证。"
        "15 折共享训练数据且患者在三个种子下重复出现，不能把 15 个数当作完全独立的显著性证据。",
        "",
        "检查点保存了本轮模型和图，用于同图预测重放；通过重放校验不表示已经实现新患者或未见中心的部署推理。"
        "PET 原型是可学习内部向量，本轮没有病理亚型标签验证，不宣称真实 PET 图像生成或医学诊断可用。",
        "",
        "独立复核通过：`verification_independent.json`。本报告额外检查了 150 条记录的唯一性、"
        "配对患者、固定图和汇总数字；保留全部条件，包括负增益，不根据测试表删除方案。",
        "",
        "## 下一轮",
        "",
        "1. 用验证集分类、PET 重构与稳定性证据固定下一轮方向，并把配置写入协议。"
        "对没有超过无原型对照的方案，不把改善解释为原型机制已被证明；先检查输出收缩和邻居有效性。",
        "2. 固定方向后补充预先定义的其他任务与人工 PET 缺失率实验，核对 MCI 转归标签及随访窗口。"
        "这些任务当前还没有结果，不能把 AD/CN 一轮比较写成四任务结论。",
        "3. 准备字段兼容和推理协议后做 ADNI 确认，训练/验证阶段固定方法，再做外部评估。"
        "投稿 ICMR 或 ICME 的可行性需要后续的稳定增益、消融和外部证据，不由本轮最高测试分数决定。",
        "",
        "## 对应文件",
        "",
    ])
    for name, label in (("protocol.json", "预定协议"), ("records.json", "150 条逐折指标与诊断"),
                        ("summary.json", "汇总结果"), ("verification_independent.json", "独立复核")):
        path = (results / name).resolve().as_posix()
        lines.append(f"- [{label}]({path})；SHA256：`{sha256(results / name)}`。")
    lines.append("")
    return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results", type=Path, default=ROOT / "results" / "round01_core")
    args = parser.parse_args()
    results = args.results.resolve()
    protocol, records, summary, audit, grouped = gate(results)
    report = build_report(results, protocol, records, summary, audit, grouped)
    target = results / "实验结论.md"
    temporary = results / "实验结论.md.tmp"
    temporary.write_text(report, encoding="utf-8")
    temporary.replace(target)
    print(f"Written completed, independently verified report: {target}")


if __name__ == "__main__":
    main()
