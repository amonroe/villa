from __future__ import annotations

import pytest
import torch
import torch.nn.functional as F
from torch.nn.attention import SDPBackend, sdpa_kernel

from vesuvius.models.build.pretrained_backbones.dinovol_2_eva import EvaAttention, _sdpa_head_dim_aligned


@pytest.mark.parametrize("head_dim", [54, 30, 64])
def test_padded_attention_matches_unpadded(head_dim: int):
    torch.manual_seed(0)
    q, k, v = (torch.randn(2, 4, 37, head_dim, dtype=torch.float64) for _ in range(3))

    expected = F.scaled_dot_product_attention(q, k, v)
    actual = _sdpa_head_dim_aligned(q, k, v)

    assert actual.shape == expected.shape
    torch.testing.assert_close(actual, expected)


def test_padded_attention_matches_unpadded_with_mask():
    torch.manual_seed(0)
    q, k, v = (torch.randn(2, 4, 37, 54, dtype=torch.float64) for _ in range(3))
    mask = torch.rand(2, 1, 1, 37) > 0.3
    mask[..., 0] = True

    torch.testing.assert_close(
        _sdpa_head_dim_aligned(q, k, v, attn_mask=mask),
        F.scaled_dot_product_attention(q, k, v, attn_mask=mask),
    )


def test_padded_attention_gradients_match_unpadded():
    torch.manual_seed(0)
    inputs = [torch.randn(1, 2, 19, 54, dtype=torch.float64, requires_grad=True) for _ in range(3)]

    grads_expected = torch.autograd.grad(F.scaled_dot_product_attention(*inputs).sum(), inputs)
    grads_actual = torch.autograd.grad(_sdpa_head_dim_aligned(*inputs).sum(), inputs)

    for actual, expected in zip(grads_actual, grads_expected):
        torch.testing.assert_close(actual, expected)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")
def test_fp16_head_dim_54_uses_memory_efficient_kernel():
    torch.manual_seed(0)
    attention = EvaAttention(dim=864, num_heads=16, num_prefix_tokens=5).cuda().eval()
    x = torch.randn(1, 4096, 864, device="cuda")

    with torch.no_grad():
        expected = attention(x)
        # Before the fix this raised "No available kernel": head_dim 54 is not
        # supported by the memory-efficient kernel in fp16.
        with sdpa_kernel([SDPBackend.EFFICIENT_ATTENTION]):
            actual = attention.half()(x.half())

    assert torch.isfinite(actual).all()
    cosine = F.cosine_similarity(actual.float(), expected, dim=-1)
    assert cosine.min().item() > 0.9999
