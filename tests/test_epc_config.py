import os

import pytest

from maceh.parse_configs import EPCConfig


def write_config(tmp_path):
    out_dir = tmp_path / 'out'
    cfg = f"""
[basic]
device = cpu
dtype = double
trained_model_dir = {tmp_path}
output_dir = {out_dir}

[data]
radius = 7.2

[epc]
structure_dir = {tmp_path}
q_grid = 2 2 2
k_grid = 4 4 4
delta = 0.02
atom_indices = 0 2
"""
    path = tmp_path / 'epc.ini'
    path.write_text(cfg)
    return str(path)


def test_epc_config_parses(tmp_path):
    config = EPCConfig(write_config(tmp_path))
    assert config.q_grid == (2, 2, 2)
    assert config.k_grid == (4, 4, 4)
    assert config.delta == pytest.approx(0.02)
    assert config.atom_indices == [0, 2]
    assert config.radius == pytest.approx(7.2)
    # defaults from epc_default.ini
    assert config.grad_threshold == pytest.approx(1e-10)
    assert config.save_derivatives is False
    assert config.inference is True


def test_epc_config_requires_radius(tmp_path):
    path = write_config(tmp_path)
    text = open(path).read().replace('radius = 7.2', 'radius = -1')
    open(path, 'w').write(text)
    with pytest.raises(AssertionError):
        EPCConfig(path)
