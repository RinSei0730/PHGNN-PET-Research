"""Device and repeatable CUDA execution for the isolated repair version.

    import device as D
    D.set_device('auto')        # 'cuda' when this PyTorch build can use a GPU, otherwise 'cpu'
    x = D.T(array)              # numpy array / list / tensor -> tensor on the run's device
    a = D.N(tensor)             # tensor -> numpy array (detached, on the CPU)

The CPU path retains the historical CSR arithmetic. CUDA defaults to deterministic
sparse bmm and strict deterministic settings. This aims at repeatability within a
fixed GPU environment; CPU/GPU dropout and arithmetic remain different. Results
record the execution protocol, and incompatible runs cannot silently be resumed.

Modules are created on the CPU and then moved, so parameters start from the same values on either device.
"""
from __future__ import annotations

import os
import hashlib
from functools import lru_cache
from pathlib import Path
import numpy as np
import torch

_DEVICE = torch.device('cpu')
_STABLE_CUDA = False


def set_device(name: str = 'auto', stable_cuda: bool = True) -> torch.device:
    """Select the device; this isolated version uses repeatable CUDA kernels by default.

    CPU computation keeps the historical CSR implementation. The caller can request
    ``stable_cuda=False`` only for an explicitly labelled legacy CUDA comparison.
    """
    global _DEVICE, _STABLE_CUDA
    if name == 'auto':
        name = 'cuda' if torch.cuda.is_available() else 'cpu'
    if name.startswith('cuda') and not torch.cuda.is_available():
        raise RuntimeError('--device cuda was asked for, but this PyTorch build cannot use a GPU '
                           f'(torch {torch.__version__}); use the CUDA environment or --device cpu')
    _DEVICE = torch.device(name)
    _STABLE_CUDA = bool(stable_cuda and _DEVICE.type == 'cuda')
    if _STABLE_CUDA:
        # Set before the runner's first CUDA matmul/linear operation creates a cuBLAS handle.
        workspace = os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG', ':4096:8')
        if workspace not in (':4096:8', ':16:8'):
            raise RuntimeError('Stable CUDA requires CUBLAS_WORKSPACE_CONFIG=:4096:8 or :16:8')
        torch.use_deterministic_algorithms(True)
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True
    return _DEVICE


def stable_cuda() -> bool:
    return _STABLE_CUDA


@lru_cache(maxsize=1)
def _source_hashes() -> dict:
    return {p.name: hashlib.sha256(p.read_bytes()).hexdigest()
            for p in sorted(Path(__file__).parent.glob('*.py'))}


def dev() -> torch.device:
    return _DEVICE


def T(a, dtype=None) -> torch.Tensor:
    """Tensor on the run's device (shares memory with a numpy array on the CPU, as torch.from_numpy does)."""
    return torch.as_tensor(a, dtype=dtype).to(_DEVICE)


def N(t: torch.Tensor) -> np.ndarray:
    return t.detach().cpu().numpy()


def describe() -> dict:
    """What a result file records about where it was computed."""
    import sklearn
    out = {'device': _DEVICE.type, 'torch': torch.__version__, 'numpy': np.__version__, 'sklearn': sklearn.__version__}
    if _DEVICE.type == 'cuda':
        out['gpu'] = torch.cuda.get_device_name(_DEVICE)
    out['execution'] = {
        'protocol': 'stable_cuda_bmm_v1' if _STABLE_CUDA else 'legacy_csr',
        'sparse_backend': 'coo_sparse_bmm' if _STABLE_CUDA else 'csr_sparse_mm',
        'deterministic_algorithms': torch.are_deterministic_algorithms_enabled(),
        'cublas_workspace_config': os.environ.get('CUBLAS_WORKSPACE_CONFIG') if _DEVICE.type == 'cuda' else None,
        'matmul_allow_tf32': torch.backends.cuda.matmul.allow_tf32 if _DEVICE.type == 'cuda' else None,
        'threads': torch.get_num_threads(),
    }
    if _STABLE_CUDA:
        out['execution']['source_sha256'] = _source_hashes()
    return out


def same_device(stored: dict | None) -> bool:
    """Resume/overwrite only a compatible device, kernel protocol and recorded environment."""
    actual = describe()
    stored = stored or {'device': 'cpu'}
    if stored.get('device', 'cpu') != _DEVICE.type:
        return False
    old_execution = stored.get('execution')
    if old_execution is None:
        if _STABLE_CUDA or torch.are_deterministic_algorithms_enabled():
            return False
    elif old_execution != actual['execution']:
        return False
    return all(stored[key] == actual.get(key) for key in ('torch', 'numpy', 'sklearn', 'gpu') if key in stored)


def guard(paths, overwrite: bool = False) -> None:
    """Refuse to replace result files that were computed on another device, unless ``overwrite``.

    The numbers of record up to 2026-10-06 are CPU results; a GPU run under the same name must not silently replace them.
    """
    import json
    for f in paths:
        if overwrite or not f.exists():
            continue
        try:
            env = json.loads(f.read_text(encoding='utf-8')).get('env')
        except Exception:                                   # unreadable or not a result file: nothing to protect
            continue
        if not same_device(env):
            raise SystemExit(f'{f} holds results with different device, environment or execution settings. '
                             'Give the run another name/output directory, or pass --overwrite to replace them.')
