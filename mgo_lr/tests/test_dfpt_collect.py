import json
import os

import numpy as np
import pytest

from mgo_lr import dfpt
from mgo_lr.config import load_config
from mgo_lr.tests.test_gen_structures import Args, make_fake_reference

CFG = load_config("mgo_lr/configs/mgo.yaml")

PH_OUT = """
     Computing the dielectric constant

          Dielectric constant in cartesian axis

          (       3.135573418       0.000000000       0.000000000 )
          (       0.000000000       3.135573418       0.000000000 )
          (       0.000000000       0.000000000       3.135573418 )

          Effective charges (d Force / dE) in cartesian axis without asr

           atom      1   Mg
      Ex  (        1.97120        0.00000        0.00000 )
      Ey  (        0.00000        1.97120        0.00000 )
      Ez  (        0.00000        0.00000        1.97120 )
           atom      2   O
      Ex  (       -1.95320        0.00000        0.00000 )
      Ey  (        0.00000       -1.95320        0.00000 )
      Ez  (        0.00000        0.00000       -1.95320 )

          Effective charges (d P / du) in cartesian axis apply asr

           atom      1   Mg
      Px  (        9.99999        0.00000        0.00000 )
      Py  (        0.00000        9.99999        0.00000 )
      Pz  (        0.00000        0.00000        9.99999 )
           atom      2   O
      Px  (       -9.99999        0.00000        0.00000 )
      Py  (        0.00000       -9.99999        0.00000 )
      Pz  (        0.00000        0.00000       -9.99999 )
"""


def test_parse_ph_output():
    eps, zstar, labels = dfpt.parse_ph_output(PH_OUT)
    assert eps.shape == (3, 3)
    assert abs(eps[0, 0] - 3.135573418) < 1e-9
    assert zstar.shape == (2, 3, 3)          # d P / du block NOT parsed
    assert labels == ["Mg", "O"]
    assert abs(zstar[0, 0, 0] - 1.97120) < 1e-9
    assert abs(zstar[1, 2, 2] + 1.95320) < 1e-9


def test_parse_missing_born_raises():
    with pytest.raises(ValueError, match="Born"):
        dfpt.parse_ph_output("no tensors here\n")


def test_apply_asr_exact():
    _, zstar, _ = dfpt.parse_ph_output(PH_OUT)
    corrected = dfpt.apply_asr(zstar)
    assert np.abs(corrected.sum(axis=0)).max() < 1e-13
    # symmetric correction: each atom shifted by half the raw sum
    assert abs(corrected[0, 0, 0] - (1.97120 - 0.017 / 2 * 2 / 2)) < 1e-2


def _ws_with_ph_out(tmp_path, text):
    ws = str(tmp_path)
    make_fake_reference(ws)
    qdir = os.path.join(ws, "reference", "qe")
    os.makedirs(qdir, exist_ok=True)
    with open(os.path.join(qdir, "ph.out"), "w") as f:
        f.write(text)
    return ws


def test_collect_dfpt_stage(tmp_path):
    ws = _ws_with_ph_out(tmp_path, PH_OUT)
    assert dfpt.collect_dfpt_stage(CFG, ws, Args()) == 0
    ref = os.path.join(ws, "reference")
    z = np.load(os.path.join(ref, "born_effective_charges.npy"))
    e = np.load(os.path.join(ref, "dielectric_infinity.npy"))
    assert z.shape == (2, 3, 3) and e.shape == (3, 3)
    assert np.abs(z.sum(axis=0)).max() < 1e-13          # ASR applied
    assert z[0, 0, 0] > 0 > z[1, 0, 0]
    assert os.path.exists(os.path.join(ref, "qe_dfpt_output.out"))
    checks = json.load(open(os.path.join(ref, "dfpt_checks.json")))
    assert checks["hard_failures"] == []


def test_collect_dfpt_sign_flip_fails(tmp_path):
    flipped = PH_OUT.replace("1.97120", "-1.97120").replace("--1.97120",
                                                            "1.97120")
    ws = _ws_with_ph_out(tmp_path, flipped)
    with pytest.raises(SystemExit):
        dfpt.collect_dfpt_stage(CFG, ws, Args())
