"""Predefined exploratory PHGNN/missing-PET study; no clinical/V4 readout rules."""
from __future__ import annotations

import argparse
import copy
import hashlib
import json
import platform
import shutil
import sys
import time
from datetime import datetime
from pathlib import Path

sys.dont_write_bytecode = True
SOURCE = Path(__file__).resolve().parent
sys.path.insert(0, str(SOURCE / "pipeline"))

import numpy as np
import torch
from sklearn.metrics import roc_auc_score

import bench as B
import data
import device as D
import pet_prompt as A

STUDY = SOURCE.parent
COHORT = Path(r"D:\NACC_练习\NACC_我自己练习_20261014_202158_536917\processed\cohort.npz")
DEFAULT_ARMS = ("baseline_naive", "baseline_observed", "neighbor", "prototype_avg",
                "prototype_concat", "mlp_concat", "prototype_residual", "prototype_pattern",
                "prototype_hypergraph", "prototype_sce_nce")
METRICS = ("auc", "bacc", "sen", "spe", "val_auc", "auc_pet", "auc_nopet")


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding="utf-8")
    temporary.replace(path)


def source_hashes():
    return {p.relative_to(SOURCE).as_posix(): sha(p) for p in sorted(SOURCE.rglob("*.py"))}


def graph_hash(h):
    h = h.tocsr()
    digest = hashlib.sha256()
    digest.update(np.asarray(h.shape, dtype=np.int64).tobytes())
    for v in (h.indptr, h.indices, h.data):
        digest.update(v.tobytes())
    return digest.hexdigest()


def cpu_value(value):
    if isinstance(value, torch.Tensor):
        return value.detach().cpu()
    if isinstance(value, dict):
        return {k: cpu_value(v) for k, v in value.items()}
    if isinstance(value, list):
        return [cpu_value(v) for v in value]
    if isinstance(value, tuple):
        return tuple(cpu_value(v) for v in value)
    return value


def average(records):
    return {key: {"mean": float(np.nanmean([r[key] for r in records])),
                  "std_population": float(np.nanstd([r[key] for r in records]))} for key in METRICS}


def summarize(out, records, tasks, arms, expected):
    result = {"scope": "predefined exploratory study; same cohort previously studied",
              "test_results_used_for_arm_selection": False, "tasks": {},
              "completed_runs": len(records), "expected_runs": expected,
              "complete": len(records) == expected,
              "clinical_supervision": False, "disease_stage_pretraining": False,
              "ridge_or_score_postprocessing": False}
    for task in tasks:
        chosen = [r for r in records if r["task"] == task]
        baselines = {(r["seed"], r["fold"]): r for r in chosen if r["arm"] == "baseline_naive"}
        conditions = {}
        for arm in arms:
            rows = [r for r in chosen if r["arm"] == arm]
            if not rows:
                continue
            item = {"completed_splits": len(rows), "metrics": average(rows)}
            paired = [(r, baselines[(r["seed"], r["fold"])]) for r in rows
                      if (r["seed"], r["fold"]) in baselines]
            if paired:
                item["paired_delta_vs_baseline_naive"] = {
                    k: {"mean": float(np.nanmean([a[k] - b[k] for a, b in paired])),
                        "std_population": float(np.nanstd([a[k] - b[k] for a, b in paired]))}
                    for k in METRICS}
                item["auc_wins_ties_losses"] = {
                    "wins": sum(a["auc"] > b["auc"] for a, b in paired),
                    "ties": sum(a["auc"] == b["auc"] for a, b in paired),
                    "losses": sum(a["auc"] < b["auc"] for a, b in paired)}
            item["per_seed"] = []
            for seed in sorted({r["seed"] for r in rows}):
                seed_rows = [r for r in rows if r["seed"] == seed]
                item["per_seed"].append({"seed": seed, "splits": len(seed_rows),
                                         "metrics": average(seed_rows)})
            conditions[arm] = item
        result["tasks"][task] = conditions
    write_json(out / "summary.json", result)
    return result


def subgroup_results(td, sp, probability):
    a, t = td.observed["amy"][sp.test], td.observed["tau"][sp.test]
    masks = {"AT": a & t, "A-": a & ~t, "-T": ~a & t, "--": ~a & ~t}
    result = {}
    for name, mask in masks.items():
        y, p = td.y[sp.test][mask], probability[sp.test][mask, 1]
        result[name] = {"n": len(y), "negative": int((y == 0).sum()),
                        "positive": int((y == 1).sum()),
                        "auc": float(roc_auc_score(y, p)) if len(np.unique(y)) == 2 else None}
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cohort", type=Path, default=COHORT)
    parser.add_argument("--out", type=Path, default=STUDY / "results" / "round01_core")
    parser.add_argument("--tasks", nargs="+", default=["ad_cn"])
    parser.add_argument("--seeds", nargs="+", type=int, default=[11, 22, 33])
    parser.add_argument("--folds", nargs="+", type=int, default=[0, 1, 2, 3, 4])
    parser.add_argument("--arms", nargs="+", choices=DEFAULT_ARMS, default=list(DEFAULT_ARMS))
    parser.add_argument("--device", choices=["cpu", "cuda"], default="cuda")
    parser.add_argument("--check-only", action="store_true")
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()
    torch.set_num_threads(2)
    D.set_device(args.device)
    out, cohort_path = args.out.resolve(), args.cohort.resolve()
    if STUDY.resolve() not in out.parents:
        raise ValueError("Store each fresh run inside this study folder.")
    if len(set(args.arms)) != len(args.arms) or len(set(args.seeds)) != len(args.seeds):
        raise ValueError("Duplicate arms/seeds are not allowed.")
    if any(f not in range(5) for f in args.folds):
        raise ValueError("Folds select indices of the fixed five-fold split.")
    cohort_hash = sha(cohort_path)
    cohort = data.load_cohort(cohort_path)
    task_data, split_groups = {}, {}
    for task in args.tasks:
        td = data.make_task(cohort, task, scope="dev")
        if td.n_classes != 2:
            raise ValueError("This first-round runner supports binary tasks only.")
        task_data[task] = td
        split_groups[task] = [sp for sp in data.cv_splits(td, args.seeds) if sp.fold in args.folds]
    expected = sum(len(splits) for splits in split_groups.values()) * len(args.arms)
    fixed = {"study": "PHGNN + prototype-guided missing PET, restart round 1",
             "created_client_date": "2026-10-15", "tasks": args.tasks, "seeds": args.seeds,
             "folds": args.folds, "arms": args.arms, "expected_runs": expected,
             "cohort": str(cohort_path), "cohort_sha256": cohort_hash,
             "source_sha256": source_hashes(), "environment": D.describe(),
             "python": platform.python_version(), "adapter": A.PROTOCOL,
             "standardization_fit": "current labeled task training fold only",
             "encoder_pca_fallback_fit": "observed training fold only; never all observed patients",
             "mae_eligible": "training fold plus unlabeled background; validation/test excluded",
             "generator_fit_and_donors": "training fold plus unlabeled background; validation/test excluded",
             "graph_protocol": "transductive: unlabeled validation/test features participate in propagation",
             "baseline_graph": "naive modality-kNN union, including zero PET rows",
             "graph_ablation": "baseline_observed: only observed PET nodes create/join PET hyperedges",
             "module_downstream_graph": "same original naive graph, fixed before imputation",
             "hypergraph_provider_context": "MRI/tab graph only, no PET-valued edge construction",
             "selection": "generator epoch by common validation PET reconstruction loss; downstream config/epoch/threshold by validation labels",
             "across_arm_test_selection": False,
             "extra_clinical_targets_or_stage_labels": False,
             "ridge_fusion_score_borrowing": False,
             "interpretation": "exploratory comparison; this cohort including locked sites was examined in earlier work; no new independent external-validation claim",
             "future_confirmation": "freeze method before ADNI/site-level evaluation; no publication readiness assumed"}
    brief = {"out": str(out), "runs": expected, "tasks": args.tasks, "seeds": args.seeds,
             "folds": args.folds, "arms": args.arms, "environment": D.describe(),
             "data": {task: {"nodes": len(td.y), "labeled": int((td.y >= 0).sum()),
                              "background": int((td.y < 0).sum())} for task, td in task_data.items()}}
    print(json.dumps(brief, ensure_ascii=False, indent=2), flush=True)
    if args.check_only:
        print("Preflight passed; no training executed.", flush=True)
        return
    if out.exists():
        if not args.resume:
            raise FileExistsError("Existing output requires explicit --resume; never overwrite old results.")
        saved = json.loads((out / "protocol.json").read_text(encoding="utf-8"))
        if saved != fixed:
            raise ValueError("Cannot resume a different cohort/source/environment/protocol.")
        records = json.loads((out / "records.json").read_text(encoding="utf-8")) if (out / "records.json").exists() else []
    else:
        out.mkdir(parents=True)
        records = []
        write_json(out / "protocol.json", fixed)
        shutil.copytree(SOURCE, out / "source")
    for name in ("checkpoints", "predictions", "selection", "split_audits"):
        (out / name).mkdir(exist_ok=True)
    completed = {(r["task"], r["seed"], r["fold"], r["arm"]) for r in records}
    if len(completed) != len(records):
        raise ValueError("Duplicate saved records.")
    print(f"Protocol frozen before training: {out / 'protocol.json'}", flush=True)
    print(f"Resume: {len(completed)}/{expected} runs kept.", flush=True)
    started = time.perf_counter()
    for task, td in task_data.items():
        for split_index, sp in enumerate(split_groups[task]):
            if all((task, sp.seed, sp.fold, arm) in completed for arm in args.arms):
                continue
            split_tag = f"{task}_seed{sp.seed}_fold{sp.fold}"
            print(f"Preparing {split_tag} (train/val/test={len(sp.train)}/{len(sp.val)}/{len(sp.test)})", flush=True)
            ctx = B.Ctx(td, sp, feat="enc", graph_mode="naive")
            prepared = A.prepare(ctx)
            original_h = ctx.incidence()[0]
            original_hash = graph_hash(original_h)
            pool = prepared["pool"]
            audit = {"task": task, "seed": sp.seed, "fold": sp.fold,
                     "train_rows": td.node_ids[sp.train].tolist(), "val_rows": td.node_ids[sp.val].tolist(),
                     "test_rows": td.node_ids[sp.test].tolist(), "fit_and_donor_rows": td.node_ids[pool].tolist(),
                     "preprocessing_rows": td.node_ids[sp.train].tolist(),
                     "donor_validation_overlap": int(np.intersect1d(pool, sp.val).size),
                     "donor_test_overlap": int(np.intersect1d(pool, sp.test).size),
                     "graph_sha256": original_hash,
                     "mae_eligible_validation_overlap": int(np.intersect1d(ctx.mae_eligible, sp.val).size),
                     "mae_eligible_test_overlap": int(np.intersect1d(ctx.mae_eligible, sp.test).size)}
            assert all(audit[k] == 0 for k in ("donor_validation_overlap", "donor_test_overlap",
                                              "mae_eligible_validation_overlap", "mae_eligible_test_overlap"))
            write_json(out / "split_audits" / f"{split_tag}.json", audit)
            baseline_probability = None
            for arm in args.arms:
                key = (task, sp.seed, sp.fold, arm)
                if key in completed:
                    continue
                tag = f"{split_tag}_{arm}"
                torch.cuda.reset_peak_memory_stats() if args.device == "cuda" else None
                t0 = time.perf_counter()
                generator, selection = None, None
                if arm == "baseline_naive":
                    active = ctx
                elif arm == "baseline_observed":
                    active = copy.copy(ctx)
                    active.graph_mode = "observed"
                    active.cache = {}
                else:
                    generator, selection = A.train_generator(ctx, prepared, arm)
                    active = A.adapted_context(ctx, prepared, generator)
                if arm != "baseline_observed":
                    assert graph_hash(active.incidence()[0]) == original_hash
                _, probability, extra = B.run_phgnn_family(active, "phgnn")
                if not np.isfinite(probability).all():
                    raise FloatingPointError("Non-finite classification probabilities.")
                if arm == "baseline_naive":
                    baseline_probability = probability.copy()
                record = B.score(td, sp, probability)
                record.update(task=task, arm=arm, seed=sp.seed, fold=sp.fold,
                              seconds=round(time.perf_counter() - t0, 2),
                              cfg=extra, graph_sha256=graph_hash(active.incidence()[0]),
                              train=len(sp.train), validation=len(sp.val), test=len(sp.test),
                              test_rows=td.node_ids[sp.test].tolist(), test_y=td.y[sp.test].tolist(),
                              test_p=probability[sp.test, 1].astype(float).tolist(),
                              pet_patterns=subgroup_results(td, sp, probability),
                              gpu_peak_allocated_mb=round(torch.cuda.max_memory_allocated() / 2**20, 2) if args.device == "cuda" else None)
                if generator is not None:
                    record["adapter_parameters"] = selection["active_parameters"]
                    record["adapter_selection"] = {k: v for k, v in selection.items() if k != "history"}
                    record["pet_reconstruction_validation"] = A.evaluate_reconstruction(ctx, prepared, generator, sp.val)
                    record["pet_reconstruction_test"] = A.evaluate_reconstruction(ctx, prepared, generator, sp.test)
                    write_json(out / "selection" / f"{tag}.json", selection)
                checkpoint = active.cache[("selected_phgnn",)]
                checkpoint["study"] = {"task": task, "seed": sp.seed, "fold": sp.fold, "arm": arm,
                                       "cohort_sha256": cohort_hash, "protocol_sha256": sha(out / "protocol.json"),
                                       "source_sha256": fixed["source_sha256"]}
                if generator is not None:
                    checkpoint["pet_generator"] = {"arm": arm, "state": generator.state_dict(), "selection": record["adapter_selection"]}
                checkpoint_path = out / "checkpoints" / f"{tag}.pt"
                torch.save(cpu_value(checkpoint), checkpoint_path)
                np.savez_compressed(out / "predictions" / f"{tag}.npz", probability=probability,
                                    node_ids=td.node_ids, train=sp.train, val=sp.val, test=sp.test,
                                    y=td.y, observed_amy=td.observed["amy"], observed_tau=td.observed["tau"])
                if split_index == 0:
                    replay = B.replay_phgnn_checkpoint(cpu_value(checkpoint))
                    error = float(np.max(np.abs(replay - probability)))
                    if error > 2e-6:
                        raise AssertionError(f"Saved checkpoint cannot replay probabilities: {error}")
                    record["checkpoint_replay_max_error"] = error
                # One complete sham pipeline checks that replacing missing blocks with zero
                # preserves the new baseline, even after generator training consumes RNG.
                if arm == "prototype_avg" and baseline_probability is not None and not (out / "zero_output_check.json").exists():
                    sham = A.adapted_context(ctx, prepared, zero_output=True)
                    assert np.array_equal(sham.x_np, ctx.x_np)
                    _, zero_probability, _ = B.run_phgnn_family(sham, "phgnn")
                    error = float(np.max(np.abs(zero_probability - baseline_probability)))
                    check = {"task": task, "seed": sp.seed, "fold": sp.fold,
                             "passed": bool(np.array_equal(zero_probability, baseline_probability)),
                             "max_probability_error": error}
                    write_json(out / "zero_output_check.json", check)
                    if not check["passed"]:
                        raise AssertionError("Zero-output full-pipeline sham differs from new baseline.")
                    print("Zero-output full-pipeline identity check passed.", flush=True)
                    del sham
                records.append(record)
                completed.add(key)
                write_json(out / "records.json", records)
                summarize(out, records, args.tasks, args.arms, expected)
                print(f"[{len(records)}/{expected}] {tag}: AUC={record['auc']:.4f} BACC={record['bacc']:.4f} val={record['val_auc']:.4f} noPET={record['auc_nopet']:.4f} ({record['seconds']:.1f}s)", flush=True)
                del active, generator, checkpoint, probability
            del ctx, prepared
            if args.device == "cuda":
                torch.cuda.empty_cache()
    if sha(cohort_path) != cohort_hash or source_hashes() != fixed["source_sha256"]:
        raise AssertionError("Input cohort or experiment source changed during the run.")
    validation = {"all_passed": True, "completed_runs": len(records), "expected_runs": expected,
                  "input_cohort_unchanged": True, "experiment_source_unchanged": True,
                  "coverage": {}, "elapsed_seconds": round(time.perf_counter() - started, 2)}
    for task, td in task_data.items():
        for arm in args.arms:
            for seed in args.seeds:
                rs = [r for r in records if (r["task"], r["arm"], r["seed"]) == (task, arm, seed)]
                rows = [v for r in rs for v in r["test_rows"]]
                assert len(rows) == len(set(rows))
                if len(args.folds) == 5:
                    assert len(rows) == int((td.y >= 0).sum())
                validation["coverage"][f"{task}/{arm}/seed{seed}"] = {"splits": len(rs), "unique_test_patients": len(set(rows)), "each_patient_tested_once": True}
    assert len(records) == expected
    write_json(out / "verification.json", validation)
    result = summarize(out, records, args.tasks, args.arms, expected)
    print("Completed predefined comparisons; no across-arm selection on test scores.", flush=True)
    for task, conditions in result["tasks"].items():
        for arm, item in conditions.items():
            m = item["metrics"]
            print(f"{task} {arm:24s} AUC={m['auc']['mean']:.4f}+/-{m['auc']['std_population']:.4f} BACC={m['bacc']['mean']:.4f} noPET={m['auc_nopet']['mean']:.4f}", flush=True)
    print(f"Summary: {out / 'summary.json'}", flush=True)


if __name__ == "__main__":
    main()
