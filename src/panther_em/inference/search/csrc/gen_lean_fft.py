"""Generate the unrolled, register-resident inverse-DFT code for ``lean_irfft_stats.cuh``.

The lean reduce kernel keeps one complex 64-point spectrum per thread entirely in
registers. For that to compile to straight-line FMA code, every loop has to be
unrolled with compile-time indices and every twiddle factor has to be a literal.
This script emits that code::

    python gen_lean_fft.py --out lean_fft_gen.cuh          # regenerate the header
    python gen_lean_fft.py --check                          # numpy self-tests only

What is emitted (all ``__device__ __forceinline__``, operating on ``float re[N], im[N]``):

``ifftN_inplace(re, im)``
    Unnormalized inverse DFT, ``X[j] = sum_k x[k] e^{+2 pi i k j / N}``, mixed-radix
    decimation-in-time (radix-4 stages plus one radix-2 stage when ``N = 2 * 4^p``).
    Input must be loaded in the digit-reversed order ``IFFTN_PERM``; output is natural.
``twiddle_residue_<n_psi>(re, im)``
    Multiplies slot ``k`` by ``e^{+2 pi i k / n_psi}`` -- the shift that turns the
    residue-0 spectrum into the residue-1 spectrum (see below).
``hermitian_pack_inplace(re, im)``
    Packs two real residue transforms into one complex 64-point IDFT (see below).

The maths, for one (pixel, hypothesis) pair with spectrum ``C_k``, ``k < F``:

    corr[psi] = Re(C_0) + 2 sum_{k>=1} Re(C_k e^{2 pi i k psi / n_psi}),  psi < n_psi = 64 R
    D_k       = a_k C_k,  a_0 = 1, a_k = 2, a_{n_psi/2} = 1 with Im dropped (Nyquist)
    corr[R j + r] = Re IDFT_64( X_r )[j],  X_r[k'] = sum_m D_{k' + 64 m} t^{(k' + 64 m) r}

with ``t = e^{2 pi i / n_psi}``. The IDFT kernel ``e^{2 pi i k j / 64}`` is periodic in
``k``, so bins ``k >= 64`` fold onto ``k' = k mod 64`` exactly: any ``F <= n_psi/2 + 1``
is handled by the same 64-point transform. Two real residues ``r`` and ``r + R/2`` are
Hermitian-packed into ONE complex IDFT, ``Z = H_r + i H_{r+R/2}`` with
``H[k'] = (X[k'] + conj X[64-k']) / 2``, so its real output is residue ``r`` and its
imaginary output residue ``r + R/2``. Since ``t^{k' R/2} = e^{2 pi i k'/128}``, the
second residue's spectrum is the first's times ``u_{k'} = e^{2 pi i k'/128}`` (applied
to the alternately-signed fold, which the kernel forms while loading).

A 128-point transform is also emitted. It is NOT used by the reduce kernel: a
128-point complex working set is 256 fp32 registers, over the 255-register limit, and
folding makes it unnecessary. It exists for the on-device generator test
(``lean_debug_ifft``) and for future kernels with a different thread mapping.
"""

from __future__ import annotations

import argparse
import cmath
import math
import sys
from dataclasses import dataclass

# --------------------------------------------------------------------------- #
# Transform plan: mixed-radix decimation-in-time
# --------------------------------------------------------------------------- #


@dataclass(frozen=True)
class Butterfly:
    """One radix-``len(indices)`` butterfly: inputs at ``indices``, input ``q`` pre-multiplied
    by ``twiddles[q]`` (``twiddles[0] == 1``), outputs written back to ``indices``."""

    indices: tuple[int, ...]
    twiddles: tuple[complex, ...]


@dataclass(frozen=True)
class Plan:
    """Fully unrolled schedule for an unnormalized inverse DFT of size ``n``.

    ``permutation[i]`` is the natural-order input index that must be loaded into
    working slot ``i``; ``stages`` run in order, each a list of independent butterflies
    over the working array; the output ends up in natural order.
    """

    n: int
    radices: tuple[int, ...]
    permutation: tuple[int, ...]
    stages: tuple[tuple[Butterfly, ...], ...]


def choose_radices(n: int) -> tuple[int, ...]:
    """Outermost-first radices: ``(2, 4, 4, ...)`` for ``n = 2 * 4^p``, else all 4s."""
    if n < 2 or n & (n - 1):
        raise ValueError(f"n must be a power of two >= 2, got {n}")
    p, rem = divmod(n.bit_length() - 1, 2)  # n = 2^(2p + rem)
    return ((2,) if rem else ()) + (4,) * p


def _permutation(n: int, radices: tuple[int, ...]) -> list[int]:
    """Digit-reversed load order for a DIT split by ``radices[0]`` first.

    Splitting ``x`` by its outermost radix ``r`` gives the subsequences ``x[q + r m]``,
    each transformed recursively and laid out consecutively in the working array.
    """
    if not radices:
        return [0]
    r, inner = radices[0], radices[1:]
    sub = _permutation(n // r, inner)
    return [q + r * m for q in range(r) for m in sub]


def _stages(n: int, radices: tuple[int, ...], offset: int = 0) -> list[list[Butterfly]]:
    """Butterfly stages for the block ``[offset, offset + n)``, innermost stage first."""
    if not radices:
        return []
    r, inner = radices[0], radices[1:]
    span = n // r
    stages: list[list[Butterfly]] = []
    # Recurse into the r sub-blocks; their stages are independent and interleave.
    for q in range(r):
        for depth, sub_stage in enumerate(_stages(span, inner, offset + q * span)):
            if depth == len(stages):
                stages.append([])
            stages[depth].extend(sub_stage)
    # Then the combining stage of this level: twiddle w_n^{j q}, inverse sign (+).
    combine = [
        Butterfly(
            indices=tuple(offset + j + q * span for q in range(r)),
            twiddles=tuple(cmath.exp(2j * math.pi * j * q / n) for q in range(r)),
        )
        for j in range(span)
    ]
    stages.append(combine)
    return stages


def make_plan(n: int) -> Plan:
    """Build the unrolled inverse-DFT plan for size ``n``."""
    radices = choose_radices(n)
    return Plan(
        n=n,
        radices=radices,
        permutation=tuple(_permutation(n, radices)),
        stages=tuple(tuple(stage) for stage in _stages(n, radices)),
    )


def run_plan(plan: Plan, x: list[complex]) -> list[complex]:
    """Reference execution of ``plan`` in Python (the emitter mirrors this exactly)."""
    work = [x[i] for i in plan.permutation]
    for stage in plan.stages:
        for bf in stage:
            a = [work[i] * w for i, w in zip(bf.indices, bf.twiddles, strict=True)]
            for out, idx in enumerate(bf.indices):
                # inverse DFT of the r inputs: y[out] = sum_q a[q] e^{+2 pi i q out / r}
                work[idx] = sum(
                    a[q] * cmath.exp(2j * math.pi * q * out / len(a)) for q in range(len(a))
                )
    return work


# --------------------------------------------------------------------------- #
# C emission helpers
# --------------------------------------------------------------------------- #


def flit(v: float) -> str:
    """A float literal with enough digits to round-trip fp32."""
    return f"{v:.9e}f"


def emit_cmul(dst_re: str, dst_im: str, src_re: str, src_im: str, w: complex) -> list[str]:
    """``dst = src * w`` with exact shortcuts for ``w`` in ``{1, -1, i, -i}``."""
    eps = 1e-12
    if abs(w - 1) < eps:
        return [f"const float {dst_re}={src_re}, {dst_im}={src_im};"]
    if abs(w + 1) < eps:
        return [f"const float {dst_re}=-{src_re}, {dst_im}=-{src_im};"]
    if abs(w - 1j) < eps:
        return [f"const float {dst_re}=-{src_im}, {dst_im}={src_re};"]
    if abs(w + 1j) < eps:
        return [f"const float {dst_re}={src_im}, {dst_im}=-{src_re};"]
    c, s = flit(w.real), flit(w.imag)
    return [
        f"const float {dst_re}={c}*{src_re} - {s}*{src_im}, "
        f"{dst_im}={s}*{src_re} + {c}*{src_im};"
    ]


def emit_butterfly(bf: Butterfly) -> list[str]:
    """Straight-line C for one radix-2 or radix-4 inverse butterfly."""
    lines = ["{"]
    for q, (idx, w) in enumerate(zip(bf.indices, bf.twiddles, strict=True)):
        lines += ["  " + s for s in emit_cmul(f"a{q}r", f"a{q}i", f"re[{idx}]", f"im[{idx}]", w)]
    i = bf.indices
    if len(i) == 2:
        lines += [
            f"  re[{i[0]}]=a0r+a1r; im[{i[0]}]=a0i+a1i;",
            f"  re[{i[1]}]=a0r-a1r; im[{i[1]}]=a0i-a1i;",
        ]
    elif len(i) == 4:
        # y0 = a0+a1+a2+a3 ; y1 = a0 + i a1 - a2 - i a3 ; y2 = a0-a1+a2-a3 ; y3 = a0 - i a1 - a2 + i a3
        lines += [
            "  const float t0r=a0r+a2r, t0i=a0i+a2i, t1r=a0r-a2r, t1i=a0i-a2i;",
            "  const float t2r=a1r+a3r, t2i=a1i+a3i, t3r=a1r-a3r, t3i=a1i-a3i;",
            f"  re[{i[0]}]=t0r+t2r; im[{i[0]}]=t0i+t2i;",
            f"  re[{i[1]}]=t1r-t3i; im[{i[1]}]=t1i+t3r;",
            f"  re[{i[2]}]=t0r-t2r; im[{i[2]}]=t0i-t2i;",
            f"  re[{i[3]}]=t1r+t3i; im[{i[3]}]=t1i-t3r;",
        ]
    else:
        raise ValueError(f"unsupported radix {len(i)}")
    lines.append("}")
    return lines


def emit_ifft(plan: Plan) -> str:
    """``ifftN_inplace`` plus the ``IFFTN_PERM`` load-order macro."""
    n = plan.n
    out = [
        f"// Unnormalized inverse DFT of size {n}, radices {plan.radices} (outermost first).",
        f"// Load slot i from natural-order index IFFT{n}_PERM[i]; output is natural order.",
        f"#define IFFT{n}_PERM_LIST " + ",".join(map(str, plan.permutation)),
        f"__device__ __forceinline__ void ifft{n}_inplace(float* __restrict__ re,"
        f" float* __restrict__ im) {{",
    ]
    for depth, stage in enumerate(plan.stages):
        out.append(f"  // ---- stage {depth}: {len(stage)} radix-{len(stage[0].indices)} butterflies")
        for bf in stage:
            out += ["  " + s for s in emit_butterfly(bf)]
    out.append("}")
    return "\n".join(out)


def emit_residue_twiddle(n_psi: int, slots: int = 64) -> str:
    """Multiply slot ``k`` by ``e^{+2 pi i k / n_psi}`` (residue shift by one)."""
    out = [
        f"// X[k] *= e^(+2 pi i k / {n_psi}), k = 1..{slots - 1}: residue-0 -> residue-1 spectrum.",
        f"__device__ __forceinline__ void twiddle_residue_{n_psi}(float* __restrict__ re,"
        f" float* __restrict__ im) {{",
    ]
    for k in range(1, slots):
        w = cmath.exp(2j * math.pi * k / n_psi)
        out.append("  { " + " ".join(emit_cmul("xr", "xi", f"re[{k}]", f"im[{k}]", w)) +
                   f" re[{k}]=xr; im[{k}]=xi; }}")
    out.append("}")
    return "\n".join(out)


def emit_hermitian_pack(slots: int = 64) -> str:
    """``Z[k] = H_a[k] + i H_b[k]`` in place, from ``X_a`` in ``(re, im)`` and ``X_b = X_a * u``.

    ``H[k] = (X[k] + conj X[N-k]) / 2`` and ``u_k = e^{+2 pi i k / 128}``. Pairs
    ``(k, N-k)`` are processed together so the update is in place: with
    ``h1 = H_a[k]`` and ``h2 = H_b[k]``, ``Z[k] = h1 + i h2`` and
    ``Z[N-k] = conj(h1) + i conj(h2)``.

    The kernel calls this with ``(re, im)`` holding ``X_a`` and relies on the identity
    ``X_b[k] = X_a[k] u_k`` -- valid when the two residues come from the same fold. For
    the general fold (``F > 64``) the kernel forms ``Z`` itself while loading; see the
    kernel for that variant, which uses the same pair structure.
    """
    n = slots

    def u(k: int) -> complex:
        return cmath.exp(2j * math.pi * k / 128)

    out = [
        "// In-place Hermitian pack of residues (a, b = a shifted by R/2):",
        "//   Z[k] = H_a[k] + i H_b[k],  H[k] = (X[k] + conj X[N-k]) / 2,  X_b[k] = X_a[k] u_k.",
        "__device__ __forceinline__ void hermitian_pack_inplace(float* __restrict__ re,"
        " float* __restrict__ im) {",
        "  { const float r=re[0]; re[0]=r; im[0]=r; }  // k=0: H_a = Re X0, H_b = Re(X0 u_0) = Re X0",
    ]
    half = n // 2
    uh = u(half)  # e^{i pi/2} = i for slots=64
    if abs(uh - 1j) < 1e-12:
        out.append(f"  {{ const float r=re[{half}], i=im[{half}]; re[{half}]=r; im[{half}]=-i; }}"
                   f"  // k={half}: H_a = Re X, H_b = Re(i X) = -Im X")
    else:
        raise NotImplementedError("hermitian_pack_inplace assumes 64 slots (u_32 = i)")
    for k in range(1, half):
        kp = n - k
        out.append("  {")
        out.append(f"    const float ar=re[{k}], ai=im[{k}], br=re[{kp}], bi=im[{kp}];")
        out.append("    const float h1r=0.5f*(ar+br), h1i=0.5f*(ai-bi);")
        out += ["    " + s for s in emit_cmul("a2r", "a2i", "ar", "ai", u(k))]
        out += ["    " + s for s in emit_cmul("b2r", "b2i", "br", "bi", u(kp))]
        out.append("    const float h2r=0.5f*(a2r+b2r), h2i=0.5f*(a2i-b2i);")
        out.append(f"    re[{k}]=h1r-h2i; im[{k}]=h1i+h2r;")
        out.append(f"    re[{kp}]=h1r+h2i; im[{kp}]=-h1i+h2r;")
        out.append("  }")
    out.append("}")
    return "\n".join(out)


def emit_constant_lookup(name: str, values: list[complex], comment: str) -> str:
    """``float2 name(unsigned k)`` returning literal constants via a switch.

    Called with a compile-time ``k`` (every caller is fully unrolled) it folds to an
    immediate, which is what lets the F > 64 fold path in the kernel apply per-slot
    twiddles without holding a second 128-register array.
    """
    out = [f"// {comment}", f"__device__ __forceinline__ float2 {name}(unsigned k) {{", "  switch (k) {"]
    for k, w in enumerate(values):
        out.append(f"    case {k}: return make_float2({flit(w.real)}, {flit(w.imag)});")
    out += ["    default: return make_float2(1.f, 0.f);", "  }", "}"]
    return "\n".join(out)


def emit_header(sizes: tuple[int, ...]) -> str:
    """The whole generated header."""
    parts = [
        "// AUTO-GENERATED by gen_lean_fft.py -- do not edit by hand. Regenerate with",
        f"//   python gen_lean_fft.py --out lean_fft_gen.cuh --sizes {' '.join(map(str, sizes))}",
        "// See that script's docstring for the maths and the conventions.",
        "#pragma once",
        "",
    ]
    for n in sizes:
        parts += [emit_ifft(make_plan(n)), ""]
    parts += [emit_residue_twiddle(256), "", emit_hermitian_pack(64), ""]
    parts += [
        emit_constant_lookup(
            "lean_tw256", [cmath.exp(2j * math.pi * k / 256) for k in range(64)],
            "e^(+2 pi i k / 256), k < 64: the residue-1 twiddle at n_psi = 256."),
        "",
        emit_constant_lookup(
            "lean_u128", [cmath.exp(2j * math.pi * k / 128) for k in range(64)],
            "e^(+2 pi i k / 128), k < 64: shifts a residue spectrum by R/2 residues."),
        "",
    ]
    return "\n".join(parts)


# --------------------------------------------------------------------------- #
# Self-tests (numpy)
# --------------------------------------------------------------------------- #


def _check_plans(sizes: tuple[int, ...]) -> None:
    import numpy as np

    rng = np.random.default_rng(0)
    for n in sizes:
        plan = make_plan(n)
        assert sorted(plan.permutation) == list(range(n)), "permutation is not a bijection"
        x = rng.standard_normal(n) + 1j * rng.standard_normal(n)
        got = np.array(run_plan(plan, list(x)))
        ref = np.fft.ifft(x) * n
        err = np.abs(got - ref).max()
        print(f"  ifft{n}: radices {plan.radices}, {sum(map(len, plan.stages))} butterflies, "
              f"max |err| vs numpy = {err:.1e}")
        assert err < 1e-11 * n


def lean_corr_reference(c, n_psi: int) -> "np.ndarray":
    """Pure-numpy model of the kernel's maths (fold + residue twiddle + Hermitian pack)."""
    import numpy as np

    f = len(c)
    r_count = n_psi // 64
    t_threads = r_count // 2
    assert f <= n_psi // 2 + 1
    d = np.array(c, dtype=complex)
    d[1:] *= 2
    if f == n_psi // 2 + 1:  # Nyquist: coefficient 1, imaginary part dropped
        d[-1] = c[-1].real
    t = cmath.exp(2j * math.pi / n_psi)
    out = np.zeros(n_psi)
    ks = np.arange(64)
    for h in range(t_threads):
        # folded spectra for residues h and h + T
        a = np.zeros(64, complex)
        b = np.zeros(64, complex)
        for k in range(f):
            kp, m = k % 64, k // 64
            fac = cmath.exp(2j * math.pi * m * h / r_count)  # t^{64 m h}
            a[kp] += fac * d[k]
            b[kp] += fac * (-1) ** m * d[k]
        xa = a * t ** (ks * h)
        xb = b * t ** (ks * h) * np.exp(2j * np.pi * ks / 128)

        def herm(x):
            return 0.5 * (x + np.conj(x[(-ks) % 64]))

        z = np.fft.ifft(herm(xa) + 1j * herm(xb)) * 64
        out[r_count * ks + h] = z.real
        out[r_count * ks + h + t_threads] = z.imag
    return out


def _check_lean_math() -> None:
    import numpy as np

    rng = np.random.default_rng(1)
    for n_psi, fs in ((128, (16, 37, 64, 65)), (256, (16, 64, 65, 80, 100, 128, 129))):
        for f in fs:
            c = rng.standard_normal(f) + 1j * rng.standard_normal(f)
            c[0] = c[0].real
            padded = np.zeros(n_psi // 2 + 1, complex)
            padded[:f] = c
            ref = np.fft.irfft(padded, n=n_psi) * n_psi  # norm="forward"
            got = lean_corr_reference(c, n_psi)
            err = np.abs(got - ref).max()
            print(f"  n_psi={n_psi} F={f:3d}: fold/pack model vs irfft(norm=forward) max |err| = {err:.1e}")
            assert err < 1e-10


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--out", help="header to write (default: print nothing, just check)")
    ap.add_argument("--sizes", type=int, nargs="+", default=[64, 128])
    ap.add_argument("--check", action="store_true", help="run the numpy self-tests")
    args = ap.parse_args(argv)
    sizes = tuple(args.sizes)
    if args.check or not args.out:
        print("plans:")
        _check_plans(sizes)
        print("kernel maths:")
        _check_lean_math()
    if args.out:
        with open(args.out, "w") as fh:
            fh.write(emit_header(sizes))
        print(f"wrote {args.out} ({sum(1 for _ in open(args.out))} lines)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
