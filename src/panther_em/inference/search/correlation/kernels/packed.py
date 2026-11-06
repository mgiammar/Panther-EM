"""The fused kernel's packed ``(value, index)`` maximum and its decoding.

The register-resident kernel keeps each pixel's running best as one int64:
the float value made order-preserving in the high 32 bits (so an unsigned
``atomicMax`` finds the maximum) and the flat ``(hypothesis, psi)`` index in the low
32 bits (mirroring ``float_to_sortable_u32`` / ``pack_val_idx`` in
``fused_irfft_and_stats_kernel.cuh``). Both the statistics and the kernel loader
decode it, so it lives here.
"""

from __future__ import annotations

import torch


def encode_argmax_packed(val: torch.Tensor, idx: torch.Tensor) -> torch.Tensor:
    """Pack ``(val, idx)`` into a single sortable ``int64``.

    Parameters
    ----------
    val : torch.Tensor
        float32 tensor of values to pack.
    idx : torch.Tensor
        Integer tensor of indices to pack (must fit in 32 bits), same shape as ``val``.

    Returns
    -------
    torch.Tensor
        int64 tensor, same shape as ``val``. Comparing two packed values via ordinary
        *unsigned* 64-bit comparison recovers the ``(val, idx)`` lexicographic order --
        see the module-level note on :func:`decode_argmax_packed` about why a plain
        ``torch.max`` on this tensor (a signed int64 view of what is conceptually
        unsigned data) is unsafe.
    """
    bits = val.contiguous().view(torch.int32).to(torch.int64) & 0xFFFFFFFF
    sign_set = (bits & 0x80000000) != 0
    mask = torch.where(
        sign_set, torch.full_like(bits, 0xFFFFFFFF), torch.full_like(bits, 0x80000000)
    )
    sortable = (bits ^ mask) & 0xFFFFFFFF
    return (sortable << 32) | (idx.to(torch.int64) & 0xFFFFFFFF)


def decode_argmax_packed(packed: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Inverse of :func:`encode_argmax_packed`; also decodes the fused kernel's output.

    Parameters
    ----------
    packed : torch.Tensor
        int64 tensor produced by :func:`encode_argmax_packed`, or the fused kernel's raw
        ``argmax_packed`` output.

    Returns
    -------
    tuple[torch.Tensor, torch.Tensor]
        ``(val, idx)``, dtypes ``(float32, int64)``, same shape as ``packed``.

    Notes
    -----
    Never reduce (``.max()``/``>``/etc.) a **packed** tensor directly: the encoding's
    order-preserving trick assumes *unsigned* 64-bit comparison, which is what CUDA's
    ``atomicMax`` on ``unsigned long long`` performs, but PyTorch has no native uint64
    dtype -- comparing the raw ``int64`` view as *signed* silently inverts the ordering
    whenever the packed value's top bit is set. Always decode first (elementwise, safe
    regardless of tensor shape), then reduce the decoded float ``val`` in ordinary
    floating-point comparison space.
    """
    idx = (packed & 0xFFFFFFFF).to(torch.int64)
    sortable = ((packed >> 32) & 0xFFFFFFFF).to(torch.int64)

    sign_set = (sortable & 0x80000000) != 0
    mask = torch.where(
        sign_set,
        torch.full_like(sortable, 0x80000000),
        torch.full_like(sortable, 0xFFFFFFFF),
    )
    bits = (sortable ^ mask) & 0xFFFFFFFF
    val = bits.to(torch.int32).view(torch.float32)
    return val, idx
