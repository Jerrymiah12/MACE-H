import numpy as np
import pytest

from maceh.epc.derivative import image_free_radius, contamination_profile


def test_image_free_radius_cubic():
    lat = np.diag([10.0, 10.0, 10.0])
    assert image_free_radius(lat) == pytest.approx(5.0)


def test_image_free_radius_uses_thinnest_direction():
    lat = np.diag([10.0, 4.0, 20.0])
    assert image_free_radius(lat) == pytest.approx(2.0)


def test_image_free_radius_is_perpendicular_not_vector_length():
    r''' for a sheared cell the perpendicular thickness is smaller than |a|; using the
    vector length would overstate how far the run is trustworthy '''
    lat = np.array([[10.0, 0.0, 0.0], [9.0, 1.0, 0.0], [0.0, 0.0, 10.0]])
    # the b direction is only 1 A thick perpendicular to a
    assert image_free_radius(lat) == pytest.approx(0.5)


def _toy(norb_per_atom=2, n_atoms=2):
    norb_cumsum = np.arange(n_atoms + 1) * norb_per_atom
    positions = np.array([[0.0, 0.0, 0.0], [2.0, 0.0, 0.0]])[:n_atoms]
    uc = np.diag([8.0, 8.0, 8.0])
    return positions, uc, norb_cumsum


def test_contamination_profile_distances_and_values():
    positions, uc, norb_cumsum = _toy()
    n = norb_cumsum[-1]
    dense = np.zeros((n, n))
    dense[0:2, :] = 3.0      # rows of atom 0
    dense[2:4, :] = 7.0      # rows of atom 1
    group = {((0, 0, 0), (0, 0, 0)): dense}
    d, v = contamination_profile(group, 0, positions, uc, uc, norb_cumsum)
    order = np.argsort(d)
    d, v = d[order], v[order]
    # displaced atom 0 is at the origin: 0 A from itself, 2 A from atom 1
    assert d == pytest.approx([0.0, 2.0])
    assert v == pytest.approx([3.0, 7.0])


def test_contamination_profile_uses_minimum_image():
    r''' a separation longer than half the cell must fold back through the boundary '''
    positions = np.array([[0.0, 0.0, 0.0], [7.0, 0.0, 0.0]])
    uc = np.diag([8.0, 8.0, 8.0])
    norb_cumsum = np.array([0, 1, 2])
    dense = np.ones((2, 2))
    group = {((0, 0, 0), (0, 0, 0)): dense}
    d, _ = contamination_profile(group, 0, positions, uc, uc, norb_cumsum)
    assert sorted(d) == pytest.approx([0.0, 1.0])   # 7 A folds to 1 A, not 7


def test_contamination_profile_offsets_by_cell_p():
    r''' p shifts the displaced atom by whole unit cells before the distance is taken '''
    positions, uc, norb_cumsum = _toy()
    dense = np.ones((norb_cumsum[-1], norb_cumsum[-1]))
    sc = np.diag([24.0, 24.0, 24.0])          # 3x1x1 supercell, no folding at 8 A
    group = {((1, 0, 0), (0, 0, 0)): dense}
    d, _ = contamination_profile(group, 0, positions, uc, sc, norb_cumsum)
    # atom 0 displaced in cell (1,0,0) sits at x = 8; bra atoms are at x = 0 and 2
    assert sorted(d) == pytest.approx([6.0, 8.0])


def test_contamination_profile_empty_group():
    positions, uc, norb_cumsum = _toy()
    d, v = contamination_profile({}, 0, positions, uc, uc, norb_cumsum)
    assert d.size == 0 and v.size == 0


def _decay_report(peaks, r_safe=5.0, capsys=None):
    r''' drive _report_decay with one value per 1 A bin '''
    from maceh.epc.run import _report_decay
    dists = np.array([i + 0.5 for i in range(len(peaks))])
    _report_decay(dists, np.array(peaks, dtype=float), r_safe, probe=0)


def test_report_decay_flags_a_real_rebound(capsys):
    # falls to 0.05 then climbs back to 0.5, well above the 1% floor of peak 10
    _decay_report([10.0, 5.0, 1.0, 0.05, 0.5, 0.6])
    out = capsys.readouterr().out
    assert 'RISES' in out and 'periodic-image artefact' in out
    # the trough quoted must be the one preceding the rebound, not a later minimum
    assert '3-4 A' in out


def test_report_decay_ignores_wiggles_in_the_noise_floor(capsys):
    # the late rise is 0.002 against a peak of 10 -- 0.02%, pure numerical floor
    _decay_report([10.0, 5.0, 1.0, 0.01, 0.001, 0.002, 0.0015])
    out = capsys.readouterr().out
    assert 'RISES' not in out
    assert 'decays monotonically' in out


def test_report_decay_monotonic_reports_where_it_reaches_the_floor(capsys):
    _decay_report([10.0, 5.0, 1.0, 0.05, 0.01, 0.005])
    out = capsys.readouterr().out
    assert 'decays monotonically' in out
    assert 'below 1% of peak beyond' in out


def test_report_decay_handles_empty_input(capsys):
    from maceh.epc.run import _report_decay
    _report_decay(np.array([]), np.array([]), 5.0, probe=0)
    assert capsys.readouterr().out == ''
