"""Tier-3 dataset-level physics and locality diagnostics.

Never a per-snapshot rejection: results are reports under
generation_logs/locality/ feeding the dataset-level approval decision
(F_SR(r) < F_full(r) over the long-distance region before scaling up).
Physics comparisons are made only within matched comparison_family_id
groups.
"""
import json
import os

import numpy as np

from .config import atomic_write_text
from .convert import parse_key, read_blocks
from .lr import blocks_norm
from .snapshot import SnapshotStore


def block_distance(key, cart, cell):
    r0, r1, r2, i, j = parse_key(key)
    shift = np.array([r0, r1, r2], float) @ np.asarray(cell, float)
    cart = np.asarray(cart, float)
    return float(np.linalg.norm(cart[j - 1] + shift - cart[i - 1]))


def frobenius_inner(a, b):
    return float(sum(np.sum(a[k] * b[k]) for k in set(a) & set(b)))


def odd_response(h_plus, h_minus, h_lr, delta):
    """ΔH_DFT = (H(+A) - H(-A))/2 compared against H_LR(+A).  Diagnostic
    only: ΔH_DFT = ΔH_SR + H_LR, so no exact match is expected."""
    dh = {}
    for k in set(h_plus) | set(h_minus):
        p, m = h_plus.get(k), h_minus.get(k)
        if p is None:
            p = np.zeros_like(m)
        if m is None:
            m = np.zeros_like(p)
        dh[k] = 0.5 * (p - m)
    n_dh, n_lr = blocks_norm(dh), blocks_norm(h_lr)
    return {"cos_theta": frobenius_inner(dh, h_lr) / (n_dh * n_lr + delta),
            "r_lr": n_lr / (n_dh + delta)}


def tail_fractions(blocks, cart, cell, radii):
    dw = [(block_distance(k, cart, cell), float(np.sum(v * v)))
          for k, v in blocks.items()]
    total = sum(w for _, w in dw)
    if total <= 0.0:
        return [0.0 for _ in radii]
    return [sum(w for d, w in dw if d > r) / total for r in radii]


def binned_norms(blocks, cart, cell, bin_width):
    bins = {}
    for k, v in blocks.items():
        b = int(block_distance(k, cart, cell) // bin_width)
        bins.setdefault(b, []).append(float(np.linalg.norm(v)))
    return [{"r_lo": b * bin_width, "r_hi": (b + 1) * bin_width,
             "count": len(ns), "mean": float(np.mean(ns)),
             "median": float(np.median(ns)), "max": float(np.max(ns))}
            for b, ns in sorted(bins.items())]


def locality_report_stage(cfg, workspace, args):
    if getattr(args, "set_name", None) is None:
        raise SystemExit("locality-report requires --set pilot|main|large")
    delta = float(cfg["validation"]["delta"])
    bin_width = float(cfg["locality"]["bin_width"])
    store = SnapshotStore(workspace, args.set_name)
    sids = [s for s in store.list()
            if store.read_status(s)["state"] == "validated"]
    if not sids:
        print(f"{args.set_name}: no validated snapshots; nothing to report")
        return 0
    metas, h_full, h_lr, h_sr = {}, {}, {}, {}
    for sid in sids:
        folder = store.folder(sid)
        with open(os.path.join(folder, "displacement_metadata.json")) as f:
            metas[sid] = json.load(f)
        h_full[sid] = read_blocks(os.path.join(folder, "hamiltonians_full.h5"))
        h_lr[sid] = read_blocks(os.path.join(folder, "hamiltonians_lr.h5"))
        h_sr[sid] = read_blocks(os.path.join(folder, "hamiltonians_sr.h5"))

    folder0 = store.folder(sids[0])
    cell = np.loadtxt(os.path.join(folder0, "lat.dat")).T   # columns -> rows
    cart0 = np.loadtxt(os.path.join(folder0, "site_positions.dat")).T
    rmax = max(block_distance(k, cart0, cell) for k in h_full[sids[0]])
    radii = [bin_width * i for i in range(1, int(rmax // bin_width) + 2)]

    tails = {"full": [], "lr": [], "sr": []}
    for sid in sids:
        cart = np.loadtxt(os.path.join(store.folder(sid),
                                       "site_positions.dat")).T
        tails["full"].append(tail_fractions(h_full[sid], cart, cell, radii))
        tails["lr"].append(tail_fractions(h_lr[sid], cart, cell, radii))
        tails["sr"].append(tail_fractions(h_sr[sid], cart, cell, radii))
    f_mean = {k: np.mean(np.array(v), axis=0).tolist()
              for k, v in tails.items()}
    upper = slice(len(radii) // 2, None)      # long-distance region
    f_sr_ok = bool(all(s <= f + 1e-12 for s, f in
                       zip(f_mean["sr"][upper], f_mean["full"][upper])))

    odd = []
    for sid in sids:
        m = metas[sid]
        partner = m.get("sign_partner_id")
        amp = float(m.get("amplitude") or 0.0)
        if partner in metas and amp > 0.0:
            entry = odd_response(h_full[sid], h_full[partner], h_lr[sid],
                                 delta)
            entry.update({"sids": [sid, partner], "amplitude": amp,
                          "family": m.get("comparison_family_id")})
            odd.append(entry)

    families = {}
    for sid in sids:
        m = metas[sid]
        fam = families.setdefault(
            m.get("comparison_family_id"),
            {"q_magnitude": m.get("q_magnitude"), "members": []})
        fam["members"].append({
            "sid": sid, "polarization_class": m.get("polarization_class"),
            "amplitude": m.get("amplitude"),
            "lr_norm": blocks_norm(h_lr[sid])})
    for fam in families.values():
        by_class = {}
        for e in fam["members"]:
            by_class.setdefault(e["polarization_class"] or "none",
                                []).append(e["lr_norm"])
        fam["mean_lr_norm_by_class"] = {c: float(np.mean(v))
                                        for c, v in by_class.items()}

    report = {"set": args.set_name, "n_snapshots": len(sids),
              "tail": {"radii": radii, "F_full": f_mean["full"],
                       "F_lr": f_mean["lr"], "F_sr": f_mean["sr"],
                       "f_sr_below_f_full": f_sr_ok},
              "binned": {"full": binned_norms(h_full[sids[0]], cart0, cell,
                                              bin_width),
                         "lr": binned_norms(h_lr[sids[0]], cart0, cell,
                                            bin_width),
                         "sr": binned_norms(h_sr[sids[0]], cart0, cell,
                                            bin_width)},
              "odd_response": odd, "families": families}
    out_dir = os.path.join(workspace, "generation_logs", "locality")
    os.makedirs(out_dir, exist_ok=True)
    atomic_write_text(os.path.join(out_dir,
                                   f"locality_{args.set_name}.json"),
                      json.dumps(report, indent=1))
    verdict = "PASS" if f_sr_ok else "NOT YET"
    print(f"{args.set_name}: locality report for {len(sids)} snapshots; "
          f"F_SR < F_full over long distances: {verdict}")
    return 0
