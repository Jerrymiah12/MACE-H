import numpy as np
import pytest

from mgo_lr import abacus_io
from mgo_lr.config import load_config
from mgo_lr.constants import ANGSTROM_TO_BOHR
from mgo_lr.structures import make_supercell, rocksalt_primitive

CFG = load_config("mgo_lr/configs/mgo.yaml")


def _sc():
    cell, frac, species = rocksalt_primitive(4.2)
    return make_supercell(cell, frac, species, 2)


def test_write_stru(tmp_path):
    sc = _sc()
    p = tmp_path / "STRU"
    abacus_io.write_stru(str(p), sc.cell, sc.cart, sc.species, CFG)
    text = p.read_text()
    lines = [l.strip() for l in text.splitlines()]
    assert "ATOMIC_SPECIES" in lines and "NUMERICAL_ORBITAL" in lines
    i = lines.index("LATTICE_CONSTANT")
    assert abs(float(lines[i + 1]) - ANGSTROM_TO_BOHR) < 1e-12
    assert "Direct" in lines
    # species blocks in config order with correct counts
    i_mg, i_o = lines.index("Mg", lines.index("Direct")), None
    i_o = lines.index("O", i_mg)
    assert int(lines[i_mg + 2]) == 8 and int(lines[i_o + 2]) == 8
    # a Direct coordinate row has 3 floats + "m 0 0 0"
    row = lines[i_mg + 3].split()
    assert len(row) == 7 and row[3] == "m"
    frac = np.array([float(x) for x in row[:3]])
    assert np.allclose(frac @ sc.cell, sc.cart[0], atol=1e-10)


def test_write_stru_rejects_bad_order(tmp_path):
    sc = _sc()
    bad = list(sc.species)
    bad[0], bad[8] = bad[8], bad[0]
    with pytest.raises(ValueError, match="species-major"):
        abacus_io.write_stru(str(tmp_path / "STRU"), sc.cell, sc.cart, bad, CFG)


def test_write_input(tmp_path):
    p = tmp_path / "INPUT"
    abacus_io.write_input(str(p), CFG, calculation="scf", out_mat_hs2=1,
                          suffix="MgO")
    text = p.read_text()
    assert text.startswith("INPUT_PARAMETERS")
    assert "gamma_only" in text and " 0" in text
    for key in ("basis_type", "ecutwfc", "scf_thr", "out_mat_hs2", "symmetry"):
        assert key in text
    with pytest.raises(ValueError, match="gamma_only"):
        abacus_io.write_input(str(p), CFG, gamma_only=1)


def test_write_kpt(tmp_path):
    p = tmp_path / "KPT"
    abacus_io.write_kpt(str(p), [4, 4, 4])
    assert p.read_text() == "K_POINTS\n0\nGamma\n4 4 4 0 0 0\n"


def test_write_job_script(tmp_path):
    p = tmp_path / "job.sh"
    abacus_io.write_job_script(str(p), CFG, ["snapshot_000001", "snapshot_000002"])
    text = p.read_text()
    assert text.startswith("#!/bin/bash")
    assert CFG["slurm"]["abacus_command"] in text
    assert "snapshot_000001" in text
