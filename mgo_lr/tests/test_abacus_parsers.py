import numpy as np
import pytest
import scipy.sparse

from mgo_lr import abacus_io
from mgo_lr.config import load_config
from mgo_lr.structures import make_supercell, rocksalt_primitive

CFG = load_config("mgo_lr/configs/mgo.yaml")

SCF_LOG = """
 Charge Density Convergence is achieved
 charge density convergence is achieved
 !FINAL_ETOT_IS -7524.123456789 eV
 EFERMI = 5.4321 eV
"""


def write_csr(path, dim, blocks, name="H", step_line=True):
    """Fabricate an ABACUS out_mat_hs2 sparse file (shared with later tests)."""
    lines = []
    if step_line:
        lines.append("STEP: 0")
    lines.append(f"Matrix Dimension of {name}(R): {dim}")
    lines.append(f"Matrix number of {name}(R): {len(blocks)}")
    for R, dense in blocks.items():
        m = scipy.sparse.csr_matrix(dense)
        lines.append(f"{R[0]} {R[1]} {R[2]} {m.nnz}")
        if m.nnz:
            lines.append(" ".join(f"{v:.12e}" for v in m.data))
            lines.append(" ".join(str(i) for i in m.indices))
            lines.append(" ".join(str(i) for i in m.indptr))
    with open(path, "w") as f:
        f.write("\n".join(lines) + "\n")


def test_parse_running_scf(tmp_path):
    p = tmp_path / "running_scf.log"
    p.write_text(SCF_LOG)
    out = abacus_io.parse_running_scf(str(p))
    assert out["converged"] is True
    assert abs(out["etot_ev"] + 7524.123456789) < 1e-9
    assert abs(out["fermi_ev"] - 5.4321) < 1e-9
    p.write_text("scf failed horribly\n")
    assert abacus_io.parse_running_scf(str(p))["converged"] is False


def test_parse_csr_roundtrip(tmp_path):
    rng = np.random.default_rng(0)
    a = rng.standard_normal((4, 4))
    a[np.abs(a) < 0.7] = 0.0
    blocks = {(0, 0, 0): a, (1, 0, -1): np.zeros((4, 4)),
              (-1, 0, 1): a.T.copy()}
    p = tmp_path / "data-HR-sparse_SPIN0.csr"
    write_csr(str(p), 4, blocks)
    dim, parsed = abacus_io.parse_csr(str(p))
    assert dim == 4
    assert set(parsed) == {(0, 0, 0), (-1, 0, 1)}    # nnz=0 block dropped
    assert np.allclose(parsed[(0, 0, 0)].toarray(), a)
    # no STEP line (older ABACUS) also parses
    write_csr(str(p), 4, blocks, step_line=False)
    dim2, parsed2 = abacus_io.parse_csr(str(p))
    assert dim2 == 4 and set(parsed2) == set(parsed)


def test_parse_csr_rejects_nan(tmp_path):
    bad = np.array([[np.nan, 1.0], [0.0, 2.0]])
    p = tmp_path / "bad.csr"
    write_csr(str(p), 2, {(0, 0, 0): bad})
    with pytest.raises(ValueError, match="NaN"):
        abacus_io.parse_csr(str(p))


def test_parse_csr_rejects_duplicate_R(tmp_path):
    p = tmp_path / "dup.csr"
    a = np.eye(2)
    lines = ["Matrix Dimension of H(R): 2", "Matrix number of H(R): 2"]
    for _ in range(2):
        m = scipy.sparse.csr_matrix(a)
        lines.append("0 0 0 2")
        lines.append(" ".join(f"{v:.3e}" for v in m.data))
        lines.append(" ".join(str(i) for i in m.indices))
        lines.append(" ".join(str(i) for i in m.indptr))
    p.write_text("\n".join(lines) + "\n")
    with pytest.raises(ValueError, match="duplicate"):
        abacus_io.parse_csr(str(p))


def test_parse_stru_roundtrip(tmp_path):
    cell, frac, species = rocksalt_primitive(4.19)
    sc = make_supercell(cell, frac, species, 2)
    p = tmp_path / "STRU"
    abacus_io.write_stru(str(p), sc.cell, sc.cart, sc.species, CFG)
    cell2, cart2, species2 = abacus_io.parse_stru(str(p))
    assert np.allclose(cell2, sc.cell, atol=1e-9)
    assert np.allclose(cart2, sc.cart, atol=1e-8)
    assert species2 == sc.species
