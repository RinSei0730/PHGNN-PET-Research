"""Focused leakage, graph-normalization and branch checks; no research data."""
from __future__ import annotations

import sys
import unittest
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import scipy.sparse as sp
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src" / "pipeline"))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
import pet_prompt as PP


class ToyContext(SimpleNamespace):
    def incidence(self):
        key = ("H", "naive")
        if key not in self.cache:
            self.cache[key] = (sp.eye(len(self.td.y), dtype=np.float32, format="csr"), None)
        return self.cache[key]

    def op(self):
        key = ("op", "naive")
        if key not in self.cache:
            self.cache[key] = object()
        return self.cache[key]


def toy_context():
    rng = np.random.default_rng(87)
    n, d = 28, 32
    z = {m: rng.normal(size=(n, d)).astype(np.float32) for m in ("mri", "amy", "tau", "tab")}
    observed = {"amy": np.arange(n) % 3 != 0, "tau": np.arange(n) % 4 != 0}
    for m in PP.PET:
        z[m][~observed[m]] = 0
    flags = np.stack([observed[m] for m in PP.PET], axis=1).astype(np.float32)
    x = np.concatenate([*z.values(), flags], axis=1)
    y = np.arange(n) % 2
    y[22:] = -1
    return ToyContext(x=torch.tensor(x), x_np=x, z=z, mods=list(z), cache={},
                           td=SimpleNamespace(y=y, observed=observed),
                           sp=SimpleNamespace(train=np.arange(14), val=np.arange(14, 18), test=np.arange(18, 22)),
                           raw={"mri": rng.normal(size=(n, 24)).astype(np.float32),
                                "tab": rng.normal(size=(n, 7)).astype(np.float32)}, seed=1)


class PETPromptTests(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(2)
        self.ctx = toy_context()
        self.prepared = PP.prepare(self.ctx)

    def test_pool_and_neighbors_exclude_val_test_self(self):
        pool = self.prepared["pool"]
        self.assertFalse(np.intersect1d(pool, np.union1d(self.ctx.sp.val, self.ctx.sp.test)).size)
        for m in PP.PET:
            nb = self.prepared["neighbors"][m].numpy()
            self.assertFalse(np.any(nb == np.arange(len(nb))[:, None]))
            self.assertTrue(np.isin(nb[nb < len(nb)], pool).all())

    def test_held_or_locked_background_cannot_supply_targets_or_donors(self):
        ctx = toy_context()
        ctx.sp.held = np.zeros(len(ctx.td.y), dtype=bool)
        ctx.sp.held[22] = True
        lock = np.zeros(len(ctx.td.y), dtype=bool)
        lock[23] = True
        ctx.td.extra = {"is_lock": lock}
        prepared = PP.prepare(ctx)
        self.assertNotIn(22, prepared["pool"])
        self.assertNotIn(23, prepared["pool"])
        for m in PP.PET:
            self.assertFalse(torch.isin(prepared["neighbors"][m], torch.tensor([22, 23])).any().item())

    def test_pca_fit_excludes_val_test(self):
        modified = toy_context()
        excluded = np.union1d(modified.sp.val, modified.sp.test)
        modified.raw["mri"][excluded] += 1000
        a, b = PP.neighbor_space(self.ctx), PP.neighbor_space(modified)
        np.testing.assert_array_equal(a[self.ctx.sp.train], b[modified.sp.train])

    def test_masked_nan_does_not_leak(self):
        usable = self.prepared["observed"]["amy"] & self.prepared["pool_mask"]
        usable[1] = False
        contaminated = self.prepared["tokens"]["amy"].clone()
        contaminated[~usable] = float("nan")
        for peer_fn in ("neighbor", "hypergraph"):
            if peer_fn == "neighbor":
                args = (usable, self.prepared["neighbors"]["amy"])
                fn = PP.neighbor_peer
            else:
                args = (usable, self.prepared["H"], self.prepared["Ht"])
                fn = PP.hypergraph_peer
            a, ac = fn(self.prepared["tokens"]["amy"], *args)
            b, bc = fn(contaminated, *args)
            torch.testing.assert_close(a, b, rtol=0, atol=0)
            torch.testing.assert_close(ac, bc, rtol=0, atol=0)
            self.assertTrue(torch.isfinite(b).all().item())

    def test_hypergraph_empty_provider_edges_are_excluded(self):
        # Node 0 belongs to both edges; second edge has no providers and must
        # not halve the value available from the first edge.
        h = sp.csr_matrix(np.array([[1, 1], [1, 0], [0, 1]], dtype=np.float32))
        ctx = SimpleNamespace(x=torch.zeros(3, 1))
        ht = PP._csr_tensor(h.T, ctx)
        htensor = PP._csr_tensor(h, ctx)
        tokens = torch.tensor([[float("nan")], [6.0], [float("nan")]])
        peer, count = PP.hypergraph_peer(tokens, torch.tensor([False, True, False]), htensor, ht)
        torch.testing.assert_close(peer, torch.tensor([[6.0], [6.0], [0.0]]))
        torch.testing.assert_close(count, torch.tensor([1.0, 1.0, 0.0]))

    def test_all_arm_outputs_gradients_and_attention(self):
        for arm in PP.ARMS:
            torch.manual_seed(120)
            model = PP.PETPrompt(arm)
            availability = {m: value.clone() for m, value in self.prepared["observed"].items()}
            availability["amy"][1] = False
            usable = availability["amy"] & self.prepared["pool_mask"]
            out, aux = model("amy", self.prepared, usable, availability, return_aux=True)
            self.assertEqual(tuple(out.shape), (28, 32))
            self.assertTrue(torch.isfinite(out).all().item())
            PP.token_loss(out[1:4], self.prepared["tokens"]["amy"][1:4], arm == "prototype_sce_nce").backward()
            self.assertTrue(any(p.grad is not None and torch.isfinite(p.grad).all().item() for p in model.parameters() if p.requires_grad))
            self.assertTrue(all(p.grad is None for p in model.parameters() if not p.requires_grad))
            if aux["attention"] is not None:
                torch.testing.assert_close(aux["attention"].sum(1), torch.ones(28))

    def test_pattern_requires_explicit_postmask_availability(self):
        model = PP.PETPrompt("prototype_pattern")
        with self.assertRaises(ValueError):
            model("amy", self.prepared, self.prepared["pool_mask"])
        a = {m: v.clone() for m, v in self.prepared["observed"].items()}
        a["amy"][1] = False
        usable = a["amy"] & self.prepared["pool_mask"]
        output = model("amy", self.prepared, usable, a)
        PP.token_loss(output[1:2], self.prepared["tokens"]["amy"][1:2]).backward()
        self.assertGreater(float(model.pattern["amy"].weight.grad.abs().sum()), 0)

    def test_short_generator_fit_is_repeatable_without_test_access(self):
        previous = PP.PROTOCOL["max_epochs"], PP.PROTOCOL["patience"]
        PP.PROTOCOL["max_epochs"], PP.PROTOCOL["patience"] = 4, 3
        try:
            a, sa = PP.train_generator(self.ctx, self.prepared, "prototype_concat")
            self.ctx.td.y[self.ctx.sp.test] = 99
            b, sb = PP.train_generator(self.ctx, self.prepared, "prototype_concat")
            self.assertEqual(sa, sb)
            for name in a.state_dict():
                torch.testing.assert_close(a.state_dict()[name], b.state_dict()[name], rtol=0, atol=0)
        finally:
            PP.PROTOCOL["max_epochs"], PP.PROTOCOL["patience"] = previous

    def test_replacement_retains_observed_flags_graph_and_clears_input_caches(self):
        self.ctx.cache["selected_phgnn"] = {"prediction": "old-inputs"}
        self.ctx.cache[("MAE", 1)] = "old-inputs"
        model = PP.PETPrompt("prototype_concat")
        clone = PP.adapted_context(self.ctx, self.prepared, model)
        self.assertIs(clone.incidence()[0], self.ctx.incidence()[0])
        self.assertIs(clone.op(), self.ctx.op())
        self.assertNotIn("selected_phgnn", clone.cache)
        self.assertNotIn(("MAE", 1), clone.cache)
        np.testing.assert_array_equal(clone.x_np[:, -2:], self.ctx.x_np[:, -2:])
        for j, m in enumerate(self.ctx.mods):
            if m in PP.PET:
                obs = self.ctx.td.observed[m]
                np.testing.assert_array_equal(clone.x_np[obs, j*32:(j+1)*32], self.ctx.x_np[obs, j*32:(j+1)*32])
        zero = PP.adapted_context(self.ctx, self.prepared, zero_output=True)
        np.testing.assert_array_equal(zero.x_np, self.ctx.x_np)

    def test_evaluation_uses_observed_targets_and_cannot_change_model(self):
        model = PP.PETPrompt("prototype_pattern")
        state = {name: value.clone() for name, value in model.state_dict().items()}
        result = PP.evaluate_reconstruction(self.ctx, self.prepared, model, self.ctx.sp.test)
        for m in PP.PET:
            self.assertEqual(result["per_modality"][m]["count"], int(self.ctx.td.observed[m][self.ctx.sp.test].sum()))
        for name, value in model.state_dict().items():
            torch.testing.assert_close(state[name], value, rtol=0, atol=0)


if __name__ == "__main__":
    unittest.main(verbosity=2)
