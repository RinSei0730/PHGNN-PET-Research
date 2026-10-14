"""CPU-only protocol tests; no cohort/GPU access required."""
from __future__ import annotations
import copy
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch
import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'src' / 'pipeline'))
import bench as B
import data as A
import device as D
import models as M
from hypergraph import multimodal_incidence


def synthetic(n=40):
    rng = np.random.default_rng(17)
    obs = np.zeros(n, bool); obs[[0, 1, 2, 3, 10, 11, 20, 21]] = True
    blocks = {m: rng.normal(size=(n, d)).astype(np.float32)
              for m, d in [('mri', 5), ('amy', 3), ('tau', 3), ('tab', 7)]}
    blocks['tab'][:, 2:] = (blocks['tab'][:, 2:] > 0).astype(np.float32)
    for m in ('amy', 'tau'):
        blocks[m][~obs] = np.nan
    y = np.arange(n) % 2; y[30:] = -1
    masks = {m: (obs.copy() if m in ('amy', 'tau') else np.ones(n, bool)) for m in blocks}
    td = A.TaskData('ad_cn', y, np.arange(n), blocks, masks,
                    {m: np.ones(n, np.int64) for m in ('amy', 'tau')},
                    {m: [f'{m}_{j}' for j in range(v.shape[1])] for m,v in blocks.items()},
                    np.zeros(n, np.int64), 2, {'is_lock': np.zeros(n, bool)})
    sp = A.Split(1, 0, np.arange(20), np.arange(20, 25), np.arange(25, 30))
    return td, sp


class PipelineProtocolTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        D.set_device('cpu'); torch.set_num_threads(1)

    def test_standardiser_excludes_validation_test_and_background(self):
        td, sp = synthetic()
        z, a = A.standardise(td.blocks, td.observed, td.tracer, sp.train, return_artifacts=True)
        changed = {m: v.copy() for m,v in td.blocks.items()}
        for v in changed.values():
            v[20:] = 100000
        z2, a2 = A.standardise(changed, td.observed, td.tracer, sp.train, return_artifacts=True)
        for m in td.blocks:
            np.testing.assert_array_equal(z[m][sp.train], z2[m][sp.train])
            for k in ('median', 'mean', 'std', 'fit_rows'):
                np.testing.assert_array_equal(a['modalities'][m]['pooled'][k],
                                              a2['modalities'][m]['pooled'][k])
            self.assertTrue(np.isin(a['modalities'][m]['pooled']['fit_rows'], sp.train).all())
        replay = A.apply_standardisation(td.blocks, td.observed, td.tracer, a)
        for m in z:
            np.testing.assert_array_equal(z[m], replay[m])

    def test_sparse_and_unseen_tracer_use_pooled_training_pet(self):
        x = np.array([[1.], [3.], [1000.], [2000.]], np.float32)
        obs = {'amy': np.ones(4, bool)}
        tr = {'amy': np.array([1, 1, 9, 9])}
        z, a = A.standardise({'amy': x}, obs, tr, np.array([0, 1]),
                             clip=1e9, return_artifacts=True)
        np.testing.assert_array_equal(a['modalities']['amy']['pooled']['fit_rows'], [0, 1])
        np.testing.assert_allclose(z['amy'][:, 0], [-1, 1, 998, 1998])

    def test_no_training_pet_returns_zero_without_fitting_evaluation_pet(self):
        obs = {'amy': np.array([False, False, True, True])}
        z, a = A.standardise({'amy': np.array([[np.nan], [np.nan], [1.], [9.]])},
                             obs, {'amy': np.ones(4, int)}, np.array([0, 1]),
                             return_artifacts=True)
        np.testing.assert_array_equal(z['amy'], np.zeros((4, 1)))
        self.assertTrue(a['modalities']['amy']['pooled']['zero_fallback'])

    def test_pca_fits_only_observed_training_and_replays(self):
        td, sp = synthetic()
        raw = A.standardise(td.blocks, td.observed, td.tracer, sp.train)
        out, a = B.modality_encoder(raw['amy'], td.y, td.observed['amy'],
                                    sp.train, sp.val, 2, 100, return_artifact=True)
        changed = raw['amy'].copy(); changed[20:] = 500
        out2, a2 = B.modality_encoder(changed, td.y, td.observed['amy'],
                                     sp.train, sp.val, 2, 100, return_artifact=True)
        self.assertEqual(a['kind'], 'pca')
        np.testing.assert_array_equal(a['fit_rows'], [0, 1, 2, 3, 10, 11])
        np.testing.assert_array_equal(a['components'], a2['components'])
        np.testing.assert_array_equal(out[sp.train], out2[sp.train])
        np.testing.assert_allclose(out, B.replay_modality_encoder(raw['amy'], td.observed['amy'], a),
                                   atol=2e-6)

    def test_no_training_observations_encoder_is_zero(self):
        x = np.random.default_rng(1).normal(size=(8, 3)).astype(np.float32)
        obs = np.array([False]*4 + [True]*4)
        out, a = B.modality_encoder(x, np.arange(8)%2, obs, np.arange(4), np.arange(4,6),
                                    2, 3, return_artifact=True)
        self.assertEqual(a['kind'], 'zero'); np.testing.assert_array_equal(out, 0)

    def test_test_labels_do_not_affect_supervised_encoder(self):
        rng = np.random.default_rng(8)
        x = rng.normal(size=(38, 4)).astype(np.float32)
        y = np.arange(38) % 2
        tr, va = np.arange(24), np.arange(24, 32)
        changed = y.copy(); changed[32:] = 1 - changed[32:]
        original_fit = M.fit
        def short_fit(*args, **kw):
            kw.update(epochs=12, patience=6)
            return original_fit(*args, **kw)
        with patch.object(M, 'fit', short_fit):
            z, a = B.modality_encoder(x, y, np.ones(38, bool), tr, va, 2, 123,
                                      return_artifact=True)
            z2, a2 = B.modality_encoder(x, changed, np.ones(38, bool), tr, va, 2, 123,
                                       return_artifact=True)
        self.assertEqual(a['kind'], 'mlp')
        np.testing.assert_array_equal(z, z2)
        for k in a['body_state']:
            self.assertTrue(torch.equal(a['body_state'][k], a2['body_state'][k]))
        np.testing.assert_allclose(z, B.replay_modality_encoder(x, np.ones(38, bool), a), atol=1e-7)

    def test_graph_modes_and_mri_tab_graph_ignore_pet_values(self):
        td, sp = synthetic()
        ctx = B.Ctx(td, sp, graph_mode='mri_tab')
        h, em = ctx.incidence()
        self.assertEqual(h.shape, (40, 80))
        expected, _ = multimodal_incidence({m:ctx.z[m] for m in ('mri','tab')}, td.observed, 30)
        self.assertEqual((h != expected).nnz, 0)
        naive, _ = ctx.incidence('naive')
        observed, _ = ctx.incidence('observed')
        self.assertEqual(naive.shape, (40, 160))
        self.assertEqual(observed.shape, (40, 96))
        for m in ('amy','tau'):
            ctx.z[m] = np.random.default_rng(7).normal(size=ctx.z[m].shape).astype(np.float32)
        ctx.cache.pop(('H','mri_tab'))
        self.assertEqual((h != ctx.incidence()[0]).nnz, 0)
        self.assertEqual(set(em), {0,1})

    def test_mae_eligibility_excludes_validation_test_and_held_background(self):
        td, sp = synthetic()
        td.extra['is_lock'][38:] = True
        ctx = B.Ctx(td, sp)
        np.testing.assert_array_equal(ctx.mae_eligible, np.r_[np.arange(20),np.arange(30,38)])
        self.assertEqual(len(np.intersect1d(ctx.mae_eligible, np.r_[sp.val,sp.test])), 0)
        captured = {}
        def fake_mae(x, op, eligible, hidden, **kw):
            captured['eligible'] = eligible.copy(); return 'ok'
        with patch.object(M, 'pretrain_mae', fake_mae):
            self.assertEqual(ctx.mae(8), 'ok')
        np.testing.assert_array_equal(captured['eligible'], ctx.mae_eligible)

    def test_selected_checkpoint_replays_post_update_token_graph_and_roundtrips(self):
        td, sp = synthetic()
        ctx = B.Ctx(td, sp, graph_mode='mri_tab')
        original_fit, original_mae, original_tune = M.fit, M.pretrain_mae, M.phgnn_prompt_tune
        def short_fit(*args, **kw):
            kw.update(epochs=9, patience=6); return original_fit(*args, **kw)
        def short_mae(*args, **kw):
            kw['epochs'] = 7; return original_mae(*args, **kw)
        def short_tune(*args, **kw):
            kw.update(epochs=12, patience=6); return original_tune(*args, **kw)
        with patch.object(M, 'fit', short_fit), patch.object(M, 'pretrain_mae', short_mae), \
                patch.object(M, 'phgnn_prompt_tune', short_tune), \
                patch.object(B, 'GRID', [dict(hidden=8,dropout=0.0,balanced=False)]):
            val, p, extra = B.run_phgnn_family(ctx)
        cp = ctx.cache[('selected_phgnn',)]
        self.assertEqual(cp['tokens'].shape, (16, ctx.x.shape[1]))
        self.assertEqual(cp['graph_mode'], 'mri_tab')
        replay = B.replay_phgnn_checkpoint(cp)
        np.testing.assert_allclose(p, replay, rtol=0, atol=2e-7)
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp)/'checkpoint.pt'
            torch.save(cp,path)
            loaded = torch.load(path,weights_only=False,map_location='cpu')
            np.testing.assert_array_equal(replay,B.replay_phgnn_checkpoint(loaded))

    def test_task_creation_has_no_clinical_or_stage_auxiliary_reads(self):
        n = 20
        c = {'dx':np.ones(n,int),'adc':np.repeat(np.arange(4),5),
             'y_ad_cn':np.arange(n)%2, 'mri_names':np.array(['ESTIMATEDTOTALINTRACRANIALVOL','CEREBRUMTCV']),
             'mri':np.tile([1.4e6,1e6],(n,1)), 'age':np.ones(n)*70,
             'educ':np.ones(n)*12,'sex':np.ones(n),'apoe_e4':np.zeros(n)}
        td = A.make_task(c,'ad_cn',modalities=('mri','tab'))
        self.assertEqual(set(td.extra), {'is_lock'})
        self.assertFalse(hasattr(A, 'privileged_targets'))


if __name__ == '__main__':
    unittest.main(verbosity=2)

