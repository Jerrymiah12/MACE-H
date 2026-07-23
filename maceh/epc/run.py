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
from .derivative import (build_supercell_graph, finite_difference, stream_finite_difference,
                         acoustic_sum_rule, hermitize_blocks)
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
    fixed supercell graph; only edge_attr is recomputed from the positions '''
    dtype = torch.get_default_dtype()
    collate = Collater()

    def predict_fn(positions):
        data.edge_attr = get_edge_fea(positions, data.lattice[0], dtype, data.edge_key)
        batch = collate([data])
        H = {}
        for kernel, net, construct_kernel in contexts:
            with torch.no_grad():
                _, output_edge = net(batch.to(device=config.device))
            H_pred = construct_kernel.get_H(output_edge).cpu().numpy()
            kernel.update_hopping(H, H_pred, batch.x.cpu(), batch.edge_index.cpu(),
                                  batch.edge_key.cpu(), debug=debug)
        if not debug:
            msg = ('Nonfinite prediction: NaN means some orbitals are not predicted '
                   '(option --debug fills them with 0); inf means the model itself '
                   'produced nonfinite output.')
            for hopping in H.values():
                # explicit raise, not assert: must survive python -O
                if not np.isfinite(hopping).all():
                    raise ValueError(msg)
        # differentiate the same symmetrized Hamiltonian the band postprocessing uses
        return hermitize_blocks(H)

    return predict_fn


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
    inv_lat = np.linalg.inv(sc_struct.lattice)
    for a in range(3):
        thickness = 1.0 / np.linalg.norm(inv_lat[:, a])
        if thickness < 2 * receptive:
            warnings.warn(f'supercell thickness along axis {a} ({thickness:.2f} A) is below '
                          f'twice the model receptive field ({receptive:.2f} A); periodic '
                          f'images of displaced atoms may contaminate dH/dR. '
                          f'Consider a denser q_grid.')

    data = build_supercell_graph(sc_struct, config.radius, torch.get_default_dtype())
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
    begin = time.time()
    deriv = stream_finite_difference(predict_fn, positions0, smap, norb_cumsum,
                                     config.delta, scratch_path,
                                     atom_indices=config.atom_indices,
                                     grad_threshold=config.grad_threshold)
    print(f'Finished {6 * n_displaced} forward passes on the supercell, '
          f'cost {time.time() - begin:.2f} seconds.')

    try:
        # delta-convergence report on the first displaced atom
        probe = [config.atom_indices[0]] if config.atom_indices else [0]
        deriv_half = finite_difference(predict_fn, positions0, smap, norb_cumsum,
                                       config.delta / 2, atom_indices=probe,
                                       grad_threshold=config.grad_threshold)
        dev = 0.0
        for alpha in range(3):
            full = deriv.group(probe[0], alpha)
            half = deriv_half.group(probe[0], alpha)
            for pR in set(full) | set(half):
                a = full.get(pR)
                b = half.get(pR)
                if a is None:
                    dev = max(dev, float(np.abs(b).max()))
                elif b is None:
                    dev = max(dev, float(np.abs(a).max()))
                else:
                    dev = max(dev, float(np.abs(a - b).max()))
        print(f'delta-convergence: max |dH(delta) - dH(delta/2)| = {dev:.3e} eV/A '
              f'(delta = {config.delta} A)')
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
            date=time.strftime('%Y-%m-%d %H:%M:%S')),
                               save_derivatives=config.save_derivatives)
        print(f'\nEPC written to "{out_path}"')
    finally:
        if os.path.exists(scratch_path):
            os.remove(scratch_path)
