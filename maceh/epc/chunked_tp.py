r''' Memory-bounded evaluation of e3nn tensor products.

The EPC supercell grows as the cube of the q-grid, and the equivariant tensor
products in the interaction blocks are evaluated once per edge with a large
hidden dimension. At 512 atoms / ~117k edges in float64 a single tensor-product
intermediate reaches ~20 GiB, which does not fit on a 32 GiB card even though
the physics needs only the same arithmetic done in pieces.

``e3nn.o3.TensorProduct`` maps row i of ``x`` and row i of ``y`` (and row i of
``weight`` when the weights are not shared) to row i of the output, with no
coupling between rows. Splitting that leading dimension is therefore exact --
not an approximation, and not a change to the model -- so this module patches
``forward`` to evaluate long inputs a chunk at a time and concatenate.

Patching e3nn rather than the model keeps trained-model source archives
untouched: a checkpoint carries a frozen copy of the code it was trained with,
and editing it to fit a bigger supercell would rewrite the provenance of a
finished training run.
'''

import contextlib

import torch

# rows per chunk are chosen so one chunk's output stays near this budget; the
# transient inside the tensor product is a multiple of it, hence the small value
CHUNK_BYTES = 256 * 1024 ** 2


def _rows(t):
    return t.shape[0] if t is not None and t.dim() > 1 else None


@contextlib.contextmanager
def chunked_tensor_products(chunk_bytes=CHUNK_BYTES, min_rows=1024):
    r''' Within this context, e3nn TensorProduct calls whose leading dimension is
    long are evaluated in slices. Results are identical to the unchunked call:
    every row is computed by the same kernel with the same inputs. '''
    from e3nn.o3 import TensorProduct

    original = TensorProduct.forward

    def forward(self, x, y, weight=None):
        nx, ny = _rows(x), _rows(y)
        n = max(nx or 0, ny or 0)
        if n == 0:
            return original(self, x, y, weight)
        out_dim = getattr(self, '_out_dim', None) or self.irreps_out.dim
        itemsize = x.element_size()
        rows = max(min_rows, int(chunk_bytes // max(out_dim * itemsize, 1)))
        if n <= rows:
            return original(self, x, y, weight)
        nw = _rows(weight)
        pieces = []
        for lo in range(0, n, rows):
            hi = min(lo + rows, n)
            xs = x[lo:hi] if nx == n else x
            ys = y[lo:hi] if ny == n else y
            ws = weight[lo:hi] if nw == n else weight
            pieces.append(original(self, xs, ys, ws))
        return torch.cat(pieces, dim=0)

    TensorProduct.forward = forward
    try:
        yield
    finally:
        TensorProduct.forward = original
