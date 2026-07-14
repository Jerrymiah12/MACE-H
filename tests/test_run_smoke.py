import subprocess
import sys

import numpy as np
import torch


def test_run_module_imports():
    from maceh.epc.run import run_epc, load_model_contexts, make_predict_fn
    assert callable(run_epc)


def test_atom_norb_from_model():
    from maceh.epc.run import atom_norb_from_model
    from maceh.kernel import DatasetInfo
    # species 0 = Z 6 with s+p (4 orbitals), species 1 = Z 79 with s (1 orbital)
    info = DatasetInfo(spinful=False, index_to_Z=[6, 79], orbital_types=[[0, 1], [0]])
    cumsum = atom_norb_from_model(info, np.array([79, 6, 6]))
    assert list(cumsum) == [0, 1, 5, 9]
    info_sp = DatasetInfo(spinful=True, index_to_Z=[6, 79], orbital_types=[[0, 1], [0]])
    cumsum_sp = atom_norb_from_model(info_sp, np.array([79, 6]))
    assert list(cumsum_sp) == [0, 2, 10]


def test_cli_help():
    out = subprocess.run([sys.executable, 'deephe3-epc.py', '--help'],
                         capture_output=True, text=True)
    assert out.returncode == 0
    assert 'electron-phonon' in out.stdout.lower()
