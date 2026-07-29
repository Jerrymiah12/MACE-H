import os
import tempfile
import time
import warnings

import numpy as np
import torch

from ..kernel import DeepHE3Kernel, NetOutInfo
from ..graph import Collater, get_edge_fea
from ..parse_configs import EPCConfig
from .supercell import load_structure, build_supercell, uniform_grid
from .derivative import (build_supercell_graph, finite_difference_pair,
                         stream_finite_difference, acoustic_sum_rule, hermitize_blocks,
                         image_free_radius, contamination_profile)
from .assembler import HoppingAssembler
from .chunked_tp import chunked_tensor_products
from .assemble import write_epc_cartesian_h5


def atom_norb_from_model(dataset_info, numbers):
    r''' per-atom orbital counts (doubled when spinful) derived from the trained
    model's dataset_info; returns the cumulative-sum slice boundaries '''
    norb_per_species = [sum(2 * l + 1 for l in types) for types in dataset_info.orbital_types]
    factor = 2 if dataset_info.spinful else 1
    norb = []
    for Z in numbers:
        species = int(dataset_info.Z_to_index[int(Z)])
        assert species >= 0, f'element Z={Z} unknown to the model'
        norb.append(factor * norb_per_species[species])
    return np.concatenate([[0], np.cumsum(norb)])


def load_model_contexts(config):
    r''' one (kernel, net, construct_kernel) per trained model, mirroring
    DeepHE3Kernel.eval; multiple models each predict a subset of targets and
    their blocks are merged by update_hopping '''
    contexts = []
    for model_path in DeepHE3Kernel.find_model(config.model_dir):
        kernel = DeepHE3Kernel()
        kernel.load_config(train_config_path=os.path.join(model_path, 'src/train.ini'))
        assert kernel.train_config.target == config.target, \
            f'model predicts {kernel.train_config.target} but EPC requires {config.target}'
        # EPC precision is decoupled from the recorded training dtype: the network is
        # built and its checkpoint loaded at the EPC dtype (load_state_dict casts), and
        # train_config's dtype fields must follow so update_hopping allocates hopping
        # blocks at the same precision instead of rounding derivatives back to float32
        kernel.train_config.set_dtype('double' if config.torch_dtype == torch.float64 else 'float')
        kernel.eval_config = config
        kernel.dataset_info = NetOutInfo.from_json(os.path.join(model_path, 'src')).dataset_info
        if contexts:
            assert kernel.dataset_info == contexts[0][0].dataset_info, \
                'all models must share the same dataset_info (species/orbital layout)'
        kernel.config_set_target()
        construct_kernel = kernel.register_constructor(device=config.device)
        net = kernel.load_model(os.path.join(model_path, 'src'), device=config.device)
        checkpoint = torch.load(os.path.join(model_path, 'best_model.pkl'), map_location='cpu')
        net.load_state_dict(checkpoint['state_dict'])
        net.eval()
        contexts.append((kernel, net, construct_kernel))
    return contexts


def make_predict_fn(contexts, data, config, debug=False):
    r''' returns predict_fn(positions) -> {str([Rx,Ry,Rz,I,J]): np.ndarray} on the
    fixed supercell graph; only edge_attr is recomputed from the positions.

    The graph is constant for the whole finite-difference sweep, so everything that
    depends on it alone is hoisted out of the per-call path: the batch is collated
    and moved to the device once, and the hopping-key / species-block / reverse-edge
    bookkeeping is precompiled into a HoppingAssembler. Only the positions travel to
    the device per call, and only the predicted values come back. '''
    dtype = torch.get_default_dtype()
    collate = Collater()
    batch = collate([data]).to(device=config.device)
    # CPU copies of the fixed integer topology, taken once rather than per call
    edge_key_cpu = data.edge_key.cpu()
    edge_index_cpu = data.edge_index.cpu()
    x_cpu = data.x.cpu()
    lattice_cpu = data.lattice[0].cpu()

    assembler = HoppingAssembler(contexts, edge_key_cpu.numpy(), edge_index_cpu.numpy(),
                                 x_cpu.numpy(), debug=debug)
    if not assembler.supported:
        # spinful models keep the general path: update_hopping's four spin blocks
        # have no vectorised equivalent here and no coverage to justify one
        warnings.warn('spinful model: falling back to the unvectorised hopping assembly')

    def predict_fn(positions):
        # edge_attr is the only input that changes. It is deliberately still built on
        # the CPU, as before: evaluating it on the device instead shifts the norm and
        # matmul rounding and moves g by ~1e-12, and on a CPU device that is the
        # difference between a reproducible run and a non-reproducible one. (On a CUDA
        # device the network's own reductions already vary by ~1e-12 between identical
        # runs, so nothing there is bit-reproducible either way.)
        edge_attr = get_edge_fea(positions, lattice_cpu, dtype, edge_key_cpu)
        batch.edge_attr = edge_attr.to(device=config.device, non_blocking=True)
        H_preds = []
        # chunked_tensor_products slices the per-edge tensor products, whose transients
        # are what a large supercell runs out of device memory on. It is exact, so it is
        # always on rather than a fallback after an OOM.
        with torch.no_grad(), chunked_tensor_products():
            for kernel, net, construct_kernel in contexts:
                _, output_edge = net(batch)
                H_preds.append(construct_kernel.get_H(output_edge).cpu().numpy())
                del output_edge
        if assembler.supported:
            # already Hermitized: the assembler folds in the same symmetrization
            H = assembler(H_preds)
        else:
            H = {}
            for (kernel, _, _), H_pred in zip(contexts, H_preds):
                kernel.update_hopping(H, H_pred, x_cpu, edge_index_cpu,
                                      edge_key_cpu, debug=debug)
            # differentiate the same symmetrized Hamiltonian the band postprocessing uses
            H = hermitize_blocks(H)
        if not debug:
            msg = ('Nonfinite prediction: NaN means some orbitals are not predicted '
                   '(option --debug fills them with 0); inf means the model itself '
                   'produced nonfinite output.')
            for hopping in H.values():
                # explicit raise, not assert: must survive python -O
                if not np.isfinite(hopping).all():
                    raise ValueError(msg)
        return H

    return predict_fn


# a rebound in the decay only counts as image contamination above this fraction of the
# peak |dH|; below it the curve is in the numerical floor, where wiggles mean nothing
FLOOR_FRACTION = 0.01


def _report_decay(dists, vals, r_safe, probe):
    r''' print how |dH| falls off with distance from the displaced atom, and whether it
    is still falling where periodic images take over. A physical derivative decays
    monotonically; if the far field is no smaller than the value at the image-free
    radius, the long-range part of this run is image contamination rather than
    physics, and only pairs inside r_safe should be used. '''
    if dists.size == 0:
        return
    edges = np.arange(0.0, dists.max() + 1.0, 1.0)
    print(f'dH decay from displaced atom {probe} (image-free radius {r_safe:.2f} A):')
    bins = []
    for lo, hi in zip(edges[:-1], edges[1:]):
        m = (dists >= lo) & (dists < hi)
        if not m.any():
            continue
        peak = float(vals[m].max())
        bins.append((lo, hi, peak))
        flag = '' if hi <= r_safe else '   <- beyond image-free radius'
        print(f'    {lo:4.1f}-{hi:4.1f} A   max |dH| = {peak:.3e} eV/A{flag}')
    # A physical dH/dtau falls off monotonically. Look for the decay turning back up:
    # that rebound is the displaced atom meeting its own periodic images, and it is what
    # actually bounds the usable range -- comparing only the two ends would miss it,
    # because the far field can still sit below the near field while being pure artefact.
    # Wiggles down in the numerical floor are not that signal, so a rebound only counts
    # once it is an appreciable fraction of the peak; otherwise the run has simply
    # decayed as far as this precision resolves.
    peak_all = max(p for _, _, p in bins)
    floor = FLOOR_FRACTION * peak_all
    trough = None          # running minimum, frozen at the moment a rebound is seen
    rebound = None
    for lo, hi, peak in bins:
        if trough is None or peak < trough[2]:
            if rebound is None:
                trough = (lo, hi, peak)
            continue
        if rebound is None and trough[1] > 1.0 and peak > 1.5 * trough[2] and peak > floor:
            rebound = (lo, hi, peak)
    if rebound is not None:
        print(f'    |dH| bottoms out at {trough[2]:.3e} eV/A near {trough[0]:.0f}-'
              f'{trough[1]:.0f} A, then RISES to {rebound[2]:.3e} at {rebound[0]:.0f}-'
              f'{rebound[1]:.0f} A: decay is non-monotonic, so dH beyond '
              f'~{trough[0]:.0f} A is periodic-image artefact, not physics.')
    else:
        reached = next((lo for lo, _, p in bins if p <= floor), None)
        tail = (f' and is below {100 * FLOOR_FRACTION:g}% of peak beyond {reached:.0f} A'
                if reached is not None else '')
        print(f'    |dH| decays monotonically over the sampled range{tail}.')
    inner = vals[dists < r_safe]
    outer = vals[dists >= r_safe]
    if inner.size and outer.size:
        frac = 100 * float(outer.max()) / max(float(inner.max()), 1e-300)
        print(f'    max |dH| inside image-free radius {float(inner.max()):.3e} eV/A, '
              f'outside {float(outer.max()):.3e} eV/A ({frac:.1f}%)')
        # Beyond r_safe a pair at nominal distance d also sits at L-d through the cell
        # boundary, so the stored block superposes the two. That superposition is exactly
        # the sum a q = Gamma displacement pattern calls for -- moving every image
        # together is what Gamma means -- so it is correct there and only limits how
        # finely q can be resolved. Saying it is "untrustworthy" would be wrong at Gamma.
        if frac > 1.0:
            print('    (that outer part is physics and its periodic images superposed: '
                  'exact in the q=Gamma sum, but the individual q dependence it encodes '
                  'is only resolved by a denser q_grid)')


def run_epc(config_path, debug=False):
    config = EPCConfig(config_path)
    torch.set_default_dtype(config.torch_dtype)
    if config.torch_dtype != torch.float64:
        warnings.warn('finite differences with float32 are noisy; '
                      'dtype = double is strongly recommended for EPC')

    struct = load_structure(config.structure_dir)

    print('\n------- Loading trained model(s) -------')
    contexts = load_model_contexts(config)
    kernel0 = contexts[0][0]
    norb_cumsum = atom_norb_from_model(kernel0.dataset_info, struct.numbers)

    print('\n------- Stage 1: finite-difference Hamiltonian derivatives -------')
    sc_struct, smap = build_supercell(struct, config.q_grid)
    receptive = (kernel0.train_config.num_blocks + 1) * kernel0.train_config.cutoff_radius
    r_safe = image_free_radius(sc_struct.lattice)
    inv_lat = np.linalg.inv(sc_struct.lattice)
    thicknesses = [1.0 / np.linalg.norm(inv_lat[:, a]) for a in range(3)]
    if min(thicknesses) < 2 * receptive:
        # the strict criterion (thickness >= 2 x receptive field) needs a supercell that
        # is usually out of reach, so state the radius the run is actually good to and
        # let the measured decay below decide whether that is enough, rather than
        # implying the warning can be silenced by a denser q_grid
        warnings.warn(
            f'supercell thicknesses {", ".join(f"{t:.2f}" for t in thicknesses)} A are '
            f'below twice the model receptive field ({2 * receptive:.2f} A): derivatives '
            f'between atoms further apart than the image-free radius {r_safe:.2f} A '
            f'include the displaced atom coupling to its own periodic images. Judge this '
            f'by the dH decay reported below, not by this warning -- satisfying it '
            f'strictly would need a q_grid of about '
            f'{int(np.ceil(2 * receptive / min(thicknesses)))} in each direction.')

    # Neighbour-list skin. The graph is built once and held fixed while atoms are
    # displaced, so a pair sitting just outside the cutoff can never enter it, and its
    # contribution to dH is silently lost. Displacing one atom by delta changes any pair
    # distance by at most delta, so a skin of 2*delta captures every pair that could come
    # inside. It costs nothing at a sane delta (2e-4 A admits almost no extra pairs) and
    # removes a systematic error that grows with delta -- at delta=1e-2 it was distorting
    # the derivatives of atom pairs near the cutoff by tens of percent.
    graph_radius = config.radius + 2.0 * config.delta
    data = build_supercell_graph(sc_struct, graph_radius, torch.get_default_dtype())
    data.x = kernel0.dataset_info.Z_to_index[data.x]
    assert torch.all(data.x >= 0), 'structure contains elements unknown to the model'
    predict_fn = make_predict_fn(contexts, data, config, debug=debug)
    positions0 = data.pos.clone()

    n_displaced = len(config.atom_indices) if config.atom_indices else smap.n_uc_atoms
    stru_id = os.path.basename(os.path.normpath(config.structure_dir))
    out_dir = os.path.join(config.out_dir, stru_id)
    os.makedirs(out_dir, exist_ok=True)
    fd_scratch, scratch_path = tempfile.mkstemp(dir=out_dir, prefix='epc_dH.', suffix='.h5')
    os.close(fd_scratch)
    try:
        begin = time.time()
        deriv = stream_finite_difference(predict_fn, positions0, smap, norb_cumsum,
                                         config.delta, scratch_path,
                                         atom_indices=config.atom_indices,
                                         grad_threshold=config.grad_threshold)
        print(f'Finished {6 * n_displaced} forward passes on the supercell, '
              f'cost {time.time() - begin:.2f} seconds.')

        # delta-convergence report on the first displaced atom, one Cartesian
        # direction at a time so at most one half-delta and one full-delta group
        # are resident at once
        probe = config.atom_indices[0] if config.atom_indices else 0
        dev = 0.0
        prof_d, prof_v = [], []
        for alpha in range(3):
            half = finite_difference_pair(predict_fn, positions0, smap, norb_cumsum,
                                          config.delta / 2, probe, alpha,
                                          grad_threshold=config.grad_threshold)
            full = deriv.group(probe, alpha)
            for pR in set(full) | set(half):
                a = full.get(pR)
                b = half.get(pR)
                if a is None:
                    dev = max(dev, float(np.abs(b).max()))
                elif b is None:
                    dev = max(dev, float(np.abs(a).max()))
                else:
                    dev = max(dev, float(np.abs(a - b).max()))
            # measured on the group already in hand, so this costs no extra reads
            d, v = contamination_profile(full, probe, struct.positions, struct.lattice,
                                         sc_struct.lattice, norb_cumsum)
            prof_d.append(d)
            prof_v.append(v)
            del half, full
        # dH(d) = D + C d^2 gives dH(d) - dH(d/2) = (3/4) C d^2, so the error of the
        # derivative actually returned is 4/3 of the measured difference, not the
        # difference itself. Reporting the raw difference understates the error, and
        # quoting it in eV/A alone gives no sense of scale, so relate it to the peak.
        peak = max((float(np.abs(m).max()) for m in prof_v if m.size), default=0.0)
        est = 4.0 * dev / 3.0
        rel = f', {100 * est / peak:.3f}% of peak |dH|' if peak > 0 else ''
        print(f'delta-convergence: max |dH(delta) - dH(delta/2)| = {dev:.3e} eV/A '
              f'(delta = {config.delta} A)')
        print(f'    => estimated error of the returned dH ~ {est:.3e} eV/A{rel}. '
              f'Error falls as delta^2 at no extra cost (the pass count is independent '
              f'of delta) until it reaches the model\'s evaluation noise.')
        _report_decay(np.concatenate(prof_d), np.concatenate(prof_v), r_safe, probe)
        if config.atom_indices is None:
            print(f'acoustic sum rule violation: {acoustic_sum_rule(deriv):.3e} eV/A')

        print('\n------- Stage 2: Fourier transform to g_ij,ka(k, q) -------')
        out_path = os.path.join(out_dir, 'epc_cartesian_pred.h5')
        write_epc_cartesian_h5(out_path, struct, deriv,
                               kpts=uniform_grid(config.k_grid),
                               qpts=uniform_grid(config.q_grid),
                               attrs=dict(
            units='g in eV/Angstrom; k, q fractional; lattice, positions in Angstrom',
            spinful=kernel0.dataset_info.spinful, delta=config.delta,
            model_dir=config.model_dir,
            note='Cartesian AO coupling g_ij,ka(k,q) = [dH(k,q)/dtau_ka]_ij; phonon-mode '
                 'contraction and band transformation (incl. possible dS/dtau handling) '
                 'are left for downstream postprocessing',
            convention='g[k,q,a,alpha,i,j] = sum_{R,p} e^{2 pi i k.R} e^{2 pi i q.p} '
                       'dH_ij(R)/dtau_{a,alpha}(p), cell-phase gauge (phases carry the '
                       'lattice vector, not the atom position). The bra index i belongs '
                       'to Bloch state k+q and the ket index j to Bloch state k: this is '
                       '<phi_i^{k+q}| dH/dtau |phi_j^k>. Downstream band rotations must '
                       'use eigenvectors at k+q on the left and k on the right.',
            # provenance: what the run is good to, without re-deriving it from the config
            q_grid=np.asarray(config.q_grid, dtype=int),
            k_grid=np.asarray(config.k_grid, dtype=int),
            graph_radius=float(config.radius),
            receptive_field=float(receptive),
            image_free_radius=float(r_safe),
            supercell_thicknesses=np.asarray(thicknesses, dtype=float),
            reproducibility='network reductions on a CUDA device vary by ~1e-12 between '
                            'identical runs; results are not bit-reproducible there',
            date=time.strftime('%Y-%m-%d %H:%M:%S')),
                               save_derivatives=config.save_derivatives)
        print(f'\nEPC written to "{out_path}"')
    finally:
        if os.path.exists(scratch_path):
            os.remove(scratch_path)
