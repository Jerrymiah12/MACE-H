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


def long_range_localizes(f_full, f_sr, floor, min_improvement):
    """Dataset-level approval criterion for `H^SR` localization.

    Over the long-distance half of the radius grid, and only where `H_full`
    still carries meaningful weight (`F_full > floor` — radii past the largest
    nonzero block are all-zero and carry no evidence), `F_SR` must be *measurably*
    below `F_full` (by at least `min_improvement`, relatively).  Equality is NOT
    a pass: a subtracted-LR Hamiltonian with the same tail as the full one shows
    no localization gain.
    """
    n = len(f_full)
    pairs = [(f_sr[i], f_full[i]) for i in range(n // 2, n) if f_full[i] > floor]
    return bool(pairs) and all(s <= f * (1.0 - min_improvement)
                               for s, f in pairs)


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
    def blocks(sid, kind):
        return read_blocks(os.path.join(store.folder(sid),
                                        f"hamiltonians_{kind}.h5"))

    # Radius grid, cell, and per-snapshot binned stats come from the first
    # validated snapshot; load its blocks once and release them.
    folder0 = store.folder(sids[0])
    cell = np.loadtxt(os.path.join(folder0, "lat.dat")).T   # columns -> rows
    cart0 = np.loadtxt(os.path.join(folder0, "site_positions.dat")).T
    h_full0 = blocks(sids[0], "full")
    rmax = max(block_distance(k, cart0, cell) for k in h_full0)
    radii = [bin_width * i for i in range(1, int(rmax // bin_width) + 2)]
    binned = {"full": binned_norms(h_full0, cart0, cell, bin_width),
              "lr": binned_norms(blocks(sids[0], "lr"), cart0, cell, bin_width),
              "sr": binned_norms(blocks(sids[0], "sr"), cart0, cell, bin_width)}
    del h_full0

    # Stream one snapshot at a time: accumulate tail fractions and the scalar
    # H_LR norms, holding only the current snapshot's blocks in memory (the
    # planned 400-structure set would need many GB if held all at once).
    metas, lr_norms = {}, {}
    tails = {"full": [], "lr": [], "sr": []}
    for sid in sids:
        folder = store.folder(sid)
        with open(os.path.join(folder, "displacement_metadata.json")) as f:
            metas[sid] = json.load(f)
        cart = np.loadtxt(os.path.join(folder, "site_positions.dat")).T
        hl = blocks(sid, "lr")
        tails["full"].append(tail_fractions(blocks(sid, "full"), cart, cell,
                                            radii))
        tails["lr"].append(tail_fractions(hl, cart, cell, radii))
        tails["sr"].append(tail_fractions(blocks(sid, "sr"), cart, cell, radii))
        lr_norms[sid] = blocks_norm(hl)
    f_mean = {k: np.mean(np.array(v), axis=0).tolist()
              for k, v in tails.items()}
    f_sr_ok = long_range_localizes(
        f_mean["full"], f_mean["sr"],
        float(cfg["locality"].get("tail_floor", 1e-6)),
        float(cfg["locality"].get("min_tail_improvement", 0.05)))

    # Odd-response pairs: load only the ± pair members' blocks, momentarily.
    odd = []
    for sid in sids:
        m = metas[sid]
        partner = m.get("sign_partner_id")
        amp = float(m.get("amplitude") or 0.0)
        if partner in metas and amp > 0.0:
            entry = odd_response(blocks(sid, "full"), blocks(partner, "full"),
                                 blocks(sid, "lr"), delta)
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
            "lr_norm": lr_norms[sid]})
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
              "binned": binned,
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
