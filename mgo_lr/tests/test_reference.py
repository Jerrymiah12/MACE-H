import copy
import json
import os

import numpy as np
import pytest
import yaml

from mgo_lr import abacus_io, reference
from mgo_lr.config import load_config
from mgo_lr.snapshot import load_reference
from mgo_lr.structures import rocksalt_primitive
from mgo_lr.tests.test_gen_structures import Args

CFG = load_config("mgo_lr/configs/mgo.yaml")


def test_init_reference_decks(tmp_path):
    ws = str(tmp_path)
    assert reference.init_reference_stage(CFG, ws, Args()) == 0
    base = os.path.join(ws, "reference", "abacus")
    for e in CFG["reference"]["ecut_scan"]:
        d = os.path.join(base, f"ecut_{e}")
        assert os.path.isdir(d)
        assert f"ecutwfc" in open(os.path.join(d, "INPUT")).read()
        assert str(e) in open(os.path.join(d, "INPUT")).read()
    for mesh in CFG["reference"]["kmesh_scan"]:
        d = os.path.join(base, f"kmesh_{mesh[0]}x{mesh[1]}x{mesh[2]}")
        assert os.path.isdir(d)
    relax = open(os.path.join(base, "cell_relax", "INPUT")).read()
    assert "cell-relax" in relax
    final = open(os.path.join(base, "final_scf", "INPUT")).read()
    assert "out_mat_hs2" in final


def test_lattice_constant_from_cell():
    cell, _, _ = rocksalt_primitive(4.19)
    assert abs(reference.lattice_constant_from_cell(cell) - 4.19) < 1e-12


def test_collect_reference_with_override(tmp_path):
    ws = str(tmp_path)
    reference.init_reference_stage(CFG, ws, Args())
    cfg = copy.deepcopy(CFG)
    cfg["material"]["lattice_constant_relaxed"] = 4.19
    assert reference.collect_reference_stage(cfg, ws, Args()) == 0
    ref = load_reference(ws)
    assert abs(reference.lattice_constant_from_cell(ref["prim_cell"]) - 4.19) < 1e-10
    assert list(ref["atomic_numbers"]) == [12, 8]
    assert ref["species"] == ["Mg", "O"]
    ref_dir = os.path.join(ws, "reference")
    ot = open(os.path.join(ref_dir, "orbital_types.dat")).read().splitlines()
    assert len(ot) == 2                       # one line per atom (2-atom cell)
    assert ot[0].split() == [str(l) for l in CFG["abacus"]["orbital_types"]["Mg"]]
    assert os.path.exists(os.path.join(ref_dir, "primitive.cif"))
    settings = yaml.safe_load(open(os.path.join(ref_dir, "dft_settings.yaml")))
    assert abs(settings["lattice_constant_relaxed"] - 4.19) < 1e-10
    # final_scf STRU regenerated at the relaxed constant
    cell2, _, _ = abacus_io.parse_stru(
        os.path.join(ref_dir, "abacus", "final_scf", "STRU"))
    assert abs(reference.lattice_constant_from_cell(cell2) - 4.19) < 1e-8


def test_collect_reference_parses_relaxed_stru(tmp_path):
    ws = str(tmp_path)
    reference.init_reference_stage(CFG, ws, Args())
    out = os.path.join(ws, "reference", "abacus", "cell_relax", "OUT.MgO")
    os.makedirs(out)
    cell, frac, species = rocksalt_primitive(4.213)
    abacus_io.write_stru(os.path.join(out, "STRU_ION_D"),
                         cell, frac @ cell, species, CFG)
    assert reference.collect_reference_stage(CFG, ws, Args()) == 0
    ref = load_reference(ws)
    assert abs(reference.lattice_constant_from_cell(ref["prim_cell"]) - 4.213) < 1e-8


def test_collect_reference_scan_summary(tmp_path):
    ws = str(tmp_path)
    reference.init_reference_stage(CFG, ws, Args())
    e0 = CFG["reference"]["ecut_scan"][0]
    out = os.path.join(ws, "reference", "abacus", f"ecut_{e0}", "OUT.MgO")
    os.makedirs(out)
    with open(os.path.join(out, "running_scf.log"), "w") as f:
        f.write("charge density convergence is achieved\n"
                "!FINAL_ETOT_IS -7000.5 eV\n")
    cfg = copy.deepcopy(CFG)
    cfg["material"]["lattice_constant_relaxed"] = 4.19
    reference.collect_reference_stage(cfg, ws, Args())
    summary = json.load(open(os.path.join(ws, "reference",
                                          "scan_summary.json")))
    assert summary["ecut"][str(e0)]["etot_ev"] == -7000.5
    assert any("missing" in w for w in summary["warnings"])


def test_collect_reference_no_relax_no_override_fails(tmp_path):
    ws = str(tmp_path)
    reference.init_reference_stage(CFG, ws, Args())
    with pytest.raises(SystemExit, match="relax"):
        reference.collect_reference_stage(CFG, ws, Args())
