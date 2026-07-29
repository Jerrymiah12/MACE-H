r''' Vectorised hopping assembly for the EPC finite-difference driver.

``DeepHE3Kernel.update_hopping`` walks the edge list in Python and, for every
edge, rebuilds the hopping key, looks the species pair up by string comparison
against every equivariant target block, and allocates a fresh matrix.
``hermitize_blocks`` then re-parses each key with ``json.loads`` to find the
reverse hopping. Both are re-executed on every forward pass even though the
graph -- and therefore every one of those lookups -- is identical across the
whole finite-difference sweep, where the only thing that changes is the
predicted values.

This module hoists all of that structure into a plan built once, so a forward
pass reduces to a handful of vectorised numpy writes. The result is the same
dict of Hermitized hopping blocks the original path returns; see
``tests/test_assembler.py``, which asserts bit-for-bit equality against
``update_hopping`` + ``hermitize_blocks`` on a randomised graph.
'''

import numpy as np


def _edge_key_strings(edge_key):
    r''' str([Rx, Ry, Rz, I, J]) per edge, matching the keys update_hopping builds
    with str(edge_key[e].tolist()) '''
    return [str([int(v) for v in row]) for row in edge_key]


class HoppingAssembler:
    r''' Precomputed replacement for update_hopping + hermitize_blocks on a fixed
    graph. Non-spinful models only -- ``supported`` is False otherwise and callers
    must keep using the general path. '''

    def __init__(self, contexts, edge_key, edge_index, species, debug=False):
        kernel0 = contexts[0][0]
        info = kernel0.dataset_info
        self.supported = not info.spinful
        self.debug = debug
        self.n_edges = int(edge_key.shape[0])
        self.keys = _edge_key_strings(edge_key)
        assert len(set(self.keys)) == self.n_edges, \
            'duplicate hopping keys in the graph; the assembler assumes one block per edge'
        if not self.supported:
            return

        self.dtype = kernel0.train_config.np_dtype
        norb_per_species = [sum(2 * l + 1 for l in types) for types in info.orbital_types]
        index_to_Z = [int(z) for z in info.index_to_Z]

        sp_i = species[edge_index[0]]
        sp_j = species[edge_index[1]]

        # one group per ordered species pair: every edge in a group writes the same
        # slices into the same matrix shape, so the whole group is one numpy write
        self.groups = {}
        pos_in_group = np.empty(self.n_edges, dtype=np.int64)
        for pair in {(int(a), int(b)) for a, b in zip(sp_i, sp_j)}:
            sel = np.flatnonzero((sp_i == pair[0]) & (sp_j == pair[1]))
            self.groups[pair] = sel
            pos_in_group[sel] = np.arange(sel.size)
        self.pos_in_group = pos_in_group

        # (group, matrix shape, [(model, out_slice, row_slice, col_slice, lr, lc)])
        self.shape_of = {}
        self.plan = {}
        for (a, b) in self.groups:
            self.shape_of[(a, b)] = (norb_per_species[a], norb_per_species[b])
            want = f'{index_to_Z[a]} {index_to_Z[b]}'
            steps = []
            for im, (kernel, _, _) in enumerate(contexts):
                noi = kernel.net_out_info
                for it, equivariant_block in enumerate(noi.blocks):
                    for N_M_str, bs in equivariant_block.items():
                        if N_M_str == want:
                            steps.append((im,
                                          slice(noi.slices[it], noi.slices[it + 1]),
                                          slice(bs[0], bs[1]), slice(bs[2], bs[3]),
                                          bs[1] - bs[0], bs[3] - bs[2]))
            self.plan[(a, b)] = steps

        # reverse hopping of each edge: [R, I, J] -> [-R, J, I]
        lookup = {}
        for e, row in enumerate(edge_key):
            lookup[(int(row[0]), int(row[1]), int(row[2]), int(row[3]), int(row[4]))] = e
        partner = np.empty(self.n_edges, dtype=np.int64)
        for e, row in enumerate(edge_key):
            rev = (-int(row[0]), -int(row[1]), -int(row[2]), int(row[4]), int(row[3]))
            f = lookup.get(rev)
            assert f is not None, \
                f'missing reverse hopping partner for edge {self.keys[e]}'
            partner[e] = f
        self.partner = partner

        # position, inside the transposed partner group, of each edge's partner
        self.partner_pos = {}
        for (a, b), sel in self.groups.items():
            self.partner_pos[(a, b)] = pos_in_group[partner[sel]]

    def __call__(self, H_preds):
        r''' H_preds: one (n_edges, n_out) array per model context.
        Returns {key_str: Hermitized block}, identical to
        hermitize_blocks(update_hopping(...)) over the same predictions. '''
        fill = 0 if self.debug else np.nan
        raw = {}
        for pair, sel in self.groups.items():
            arr = np.full((sel.size,) + self.shape_of[pair], fill, dtype=self.dtype)
            for im, so, sr, sc, lr, lc in self.plan[pair]:
                arr[:, sr, sc] = H_preds[im][sel, so].reshape(sel.size, lr, lc)
            raw[pair] = arr

        out = {}
        for (a, b), sel in self.groups.items():
            # H_ij(R) <- (H_ij(R) + H_ji(-R)^dagger) / 2, the same symmetrization the
            # band postprocessing applies, done as one array op over the whole group
            rev = raw[(b, a)][self.partner_pos[(a, b)]].conj().transpose(0, 2, 1)
            herm = (raw[(a, b)] + rev) / 2.0
            for pos, e in enumerate(sel):
                out[self.keys[e]] = herm[pos]
        return out
