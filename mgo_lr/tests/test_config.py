import copy
import subprocess
import sys

import pytest

from mgo_lr import config, constants

CFG_PATH = "mgo_lr/configs/mgo.yaml"


def test_constants_values():
    assert abs(constants.RY_TO_EV - 13.605693122994) < 1e-9
    assert abs(constants.C_COUL - 14.399645478) < 1e-6
    assert constants.LR_SIGN == -1.0
    assert abs(constants.BOHR_TO_ANGSTROM * constants.ANGSTROM_TO_BOHR - 1.0) < 1e-14


def test_load_default_config():
    cfg = config.load_config(CFG_PATH)
    assert cfg["material"]["species"] == ["Mg", "O"]
    assert cfg["abacus"]["gamma_only_algorithm"] is False
    assert cfg["supercells"] == {"pilot": 2, "main": 3, "large": 4}
    assert isinstance(cfg["lr"]["ewald_lambda"], float)


def test_missing_field_raises(tmp_path):
    cfg = config.load_config(CFG_PATH)
    bad = copy.deepcopy(cfg)
    del bad["lr"]["ewald_lambda"]
    import yaml
    p = tmp_path / "bad.yaml"
    p.write_text(yaml.safe_dump(bad))
    with pytest.raises(KeyError, match="lr.ewald_lambda"):
        config.load_config(str(p))


def test_gamma_only_true_rejected(tmp_path):
    cfg = config.load_config(CFG_PATH)
    bad = copy.deepcopy(cfg)
    bad["abacus"]["gamma_only_algorithm"] = True
    import yaml
    p = tmp_path / "bad.yaml"
    p.write_text(yaml.safe_dump(bad))
    with pytest.raises(ValueError, match="gamma_only"):
        config.load_config(str(p))


def test_save_resolved(tmp_path):
    cfg = config.load_config(CFG_PATH)
    out = config.save_resolved(cfg, str(tmp_path), "unit-test")
    assert "generation_logs" in out
    import yaml
    assert yaml.safe_load(open(out))["material"]["name"] == "MgO"


def test_cli_unknown_stage_fails():
    r = subprocess.run([sys.executable, "-m", "mgo_lr", "no-such-stage",
                        "--workspace", "/tmp/x"], capture_output=True)
    assert r.returncode != 0


def test_cli_help_lists_stages():
    r = subprocess.run([sys.executable, "-m", "mgo_lr", "--help"],
                       capture_output=True, text=True)
    assert r.returncode == 0
    assert "gen-structures" in r.stdout
