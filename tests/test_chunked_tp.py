import torch
from e3nn.o3 import TensorProduct, FullyConnectedTensorProduct, Irreps

from maceh.epc.chunked_tp import chunked_tensor_products


def _tp():
    return FullyConnectedTensorProduct('4x0e + 4x1o', '1x0e + 1x1o', '4x0e + 4x1o',
                                       shared_weights=False, internal_weights=False)


def test_chunking_is_exact_for_per_row_weights():
    torch.manual_seed(0)
    tp = _tp()
    n = 5000
    x = torch.randn(n, tp.irreps_in1.dim, dtype=torch.float64)
    y = torch.randn(n, tp.irreps_in2.dim, dtype=torch.float64)
    w = torch.randn(n, tp.weight_numel, dtype=torch.float64)
    tp = tp.double()
    ref = tp(x, y, w)
    with chunked_tensor_products(chunk_bytes=1024, min_rows=64):
        got = tp(x, y, w)
    assert got.shape == ref.shape
    assert torch.equal(got, ref), 'chunking must be bit-for-bit exact'


def test_chunking_handles_shared_weights():
    torch.manual_seed(1)
    tp = FullyConnectedTensorProduct('4x0e + 2x1o', '1x0e + 1x1o', '4x0e + 2x1o',
                                     shared_weights=True, internal_weights=True).double()
    n = 3000
    x = torch.randn(n, tp.irreps_in1.dim, dtype=torch.float64)
    y = torch.randn(n, tp.irreps_in2.dim, dtype=torch.float64)
    ref = tp(x, y)
    with chunked_tensor_products(chunk_bytes=1024, min_rows=32):
        got = tp(x, y)
    assert torch.equal(got, ref)


def test_chunking_handles_broadcast_second_operand():
    r''' y may be a single row broadcast over all of x; it must not be sliced '''
    torch.manual_seed(2)
    tp = _tp().double()
    n = 2048
    x = torch.randn(n, tp.irreps_in1.dim, dtype=torch.float64)
    y = torch.randn(1, tp.irreps_in2.dim, dtype=torch.float64)
    w = torch.randn(n, tp.weight_numel, dtype=torch.float64)
    ref = tp(x, y, w)
    with chunked_tensor_products(chunk_bytes=1024, min_rows=64):
        got = tp(x, y, w)
    assert torch.equal(got, ref)


def test_short_input_takes_the_unchunked_path():
    torch.manual_seed(3)
    tp = _tp().double()
    n = 10
    x = torch.randn(n, tp.irreps_in1.dim, dtype=torch.float64)
    y = torch.randn(n, tp.irreps_in2.dim, dtype=torch.float64)
    w = torch.randn(n, tp.weight_numel, dtype=torch.float64)
    with chunked_tensor_products(chunk_bytes=1 << 30):
        got = tp(x, y, w)
    assert torch.equal(got, tp(x, y, w))


def test_forward_is_restored_after_the_context():
    tp = _tp()
    before = TensorProduct.forward
    with chunked_tensor_products():
        assert TensorProduct.forward is not before
    assert TensorProduct.forward is before


def test_forward_is_restored_after_an_exception():
    before = TensorProduct.forward
    try:
        with chunked_tensor_products():
            raise RuntimeError('boom')
    except RuntimeError:
        pass
    assert TensorProduct.forward is before
