"""Dependency-free ``scatter`` shim.

The only C-extension from the torch-scatter/sparse/cluster family this codebase
ever used is ``torch_scatter.scatter``. Those wheels are pinned to an exact
torch+CUDA build, which blocks moving to a newer CUDA (e.g. sm_120 / Blackwell).
Prefer the real extension when it happens to be installed, otherwise fall back
to the pure-PyTorch implementations already vendored in ``from_mfn.scatter``
(which, unlike ``torch.scatter_reduce``, handle complex dtypes).
"""

from typing import Optional

import torch

from .from_mfn.scatter import scatter_mean, scatter_sum

def _scatter_fallback(
    src: torch.Tensor,
    index: torch.Tensor,
    dim: int = -1,
    out: Optional[torch.Tensor] = None,
    dim_size: Optional[int] = None,
    reduce: str = 'sum',
) -> torch.Tensor:
    if reduce in ('sum', 'add'):
        return scatter_sum(src, index, dim, out, dim_size)
    if reduce == 'mean':
        return scatter_mean(src, index, dim, out, dim_size)
    raise NotImplementedError(f"scatter reduce='{reduce}' is not supported by the fallback shim")


try:  # pragma: no cover - depends on the installed environment
    from torch_scatter import scatter  # type: ignore
except ImportError:
    scatter = _scatter_fallback


def install_torch_scatter_shim():
    r''' Register a stand-in ``torch_scatter`` module in ``sys.modules`` when the
    real extension is absent.

    A trained model directory carries a frozen copy of the source it was built
    with (``<model>/src/maceh_*``), and those snapshots still do
    ``from torch_scatter import scatter``. Rewriting an archived checkpoint's
    source to load it on a newer CUDA would mean editing the provenance record of
    a completed training run, so instead satisfy the import with the same
    pure-PyTorch reduction. ``scatter_sum`` is a plain ``scatter_add_``, so the
    numerics are identical up to floating-point summation order.

    No-op when the real ``torch_scatter`` is importable. '''
    import sys
    if 'torch_scatter' in sys.modules:
        return
    try:
        import torch_scatter  # type: ignore  # noqa: F401
        return
    except ImportError:
        pass
    import types
    mod = types.ModuleType('torch_scatter')
    mod.scatter = _scatter_fallback
    mod.scatter_sum = scatter_sum
    mod.scatter_add = scatter_sum
    mod.scatter_mean = scatter_mean
    mod.__doc__ = 'Compatibility stand-in installed by maceh.compat_scatter.'
    mod.__maceh_shim__ = True
    sys.modules['torch_scatter'] = mod
