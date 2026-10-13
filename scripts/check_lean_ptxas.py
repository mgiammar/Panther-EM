"""Register and spill report for the lean iRFFT + statistics kernels.

Builds (or reuses) the two JIT extensions with ``-Xptxas -v`` and prints one row per
kernel instantiation: registers, spill bytes and whether the entry meets the register
cap implied by its ``__launch_bounds__`` occupancy (``min_blocks_per_sm`` in
``csrc/lean_irfft_stats.cuh``). Exits non-zero when any lean entry exceeds its cap
or spills more than the tolerances below (a few bytes of spill at the 168/255-register
caps are known and harmless: the ``<64,4>``/``<56,4>`` fp32 dense entries and the
fp16 ``n_psi=256`` emit entries with ``FMAX`` in {120, 128}).

Usage::

    CUDA_VISIBLE_DEVICES=1 TORCH_EXTENSIONS_DIR=... python scripts/check_lean_ptxas.py

Set ``--rebuild`` to force a clean rebuild of both extensions (needed after editing the
kernels so a fresh ptxas report is produced; a cached build prints nothing).
"""

from __future__ import annotations

import argparse
import os
import re
import shutil
import subprocess
import sys

ENTRY_RE = re.compile(
    r"Compiling entry function '([^']+)' for 'sm_(\d+)'(.*?)"
    r"(?=ptxas info\s+: Compiling entry|\Z)",
    re.S,
)
LEAN_RE = re.compile(r"lean_irfft_stats_transposed<(\d+)u, (\d+)u, (\w+), (true|false)")
REGS_PER_SM = 65536
# bytes of spill stores + loads tolerated per entry before the check fails
DENSE_SPILL_TOLERANCE = 16
EMIT_SPILL_TOLERANCE = 128


def _demangle(name: str) -> str:
    try:
        out = subprocess.run(["c++filt", name], capture_output=True, text=True).stdout
    except OSError:
        return name
    return re.sub(r"\(.*", "", out.strip()) or name


def _register_cap(min_blocks: int, threads: int = 128) -> int:
    """Registers per thread allowed by ``__launch_bounds__(threads, min_blocks)``."""
    cap = REGS_PER_SM // (min_blocks * threads)
    cap -= cap % 8  # ptxas allocates in units of 8
    return min(255, cap)


def _min_blocks_per_sm(fmax: int, r: int) -> int:
    """Mirror of the header's table."""
    if fmax > 64:
        return 2
    if r == 2 and fmax in (24, 40, 48, 56):
        return 2
    return 3


_BUILD_SNIPPET = """
import sys
from panther_em.inference.search import fused_kernel_loader as loader
module = getattr(loader, sys.argv[1])()
sys.exit(0 if module is not None else 3)
"""


def _build_logs(rebuild: bool) -> tuple[str, str]:
    """Return the verbose build logs of the pairs and dense extensions.

    Each build runs in a child interpreter so that ninja's ptxas output (written to
    the process's real file descriptors) is captured.
    """
    if rebuild:
        ext_dir = os.environ.get("TORCH_EXTENSIONS_DIR")
        if ext_dir is None:
            sys.exit("--rebuild needs TORCH_EXTENSIONS_DIR to be set")
        for name in ("panther_em_fused_irfft_stats", "panther_em_lean_pairs"):
            shutil.rmtree(os.path.join(ext_dir, name), ignore_errors=True)
    env = {**os.environ, "PANTHER_EM_FUSED_KERNEL_VERBOSE": "1"}
    logs = []
    for fn_name in ("_try_compile_pairs", "_try_compile"):
        proc = subprocess.run(
            [sys.executable, "-c", _BUILD_SNIPPET, fn_name],
            capture_output=True,
            text=True,
            env=env,
        )
        if proc.returncode != 0:
            sys.exit(f"{fn_name} failed:\n{(proc.stdout + proc.stderr)[-4000:]}")
        logs.append(proc.stdout + proc.stderr)
    return logs[0], logs[1]


def main() -> int:
    """Build, parse the ptxas reports and print the table; see the module docstring."""
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--rebuild", action="store_true")
    parser.add_argument(
        "--all", action="store_true", help="list every entry, not just lean"
    )
    args = parser.parse_args()

    pairs_log, dense_log = _build_logs(args.rebuild)
    failures = 0
    rows = 0
    header = f"{'module':6s} {'sm':5s} {'regs':>4s} {'cap':>4s} {'spill(st/ld)':>13s}"
    print(f"{header}  kernel")
    for module, log in (("pairs", pairs_log), ("dense", dense_log)):
        for mangled, sm, body in ENTRY_RE.findall(log):
            name = _demangle(mangled)
            lean = LEAN_RE.search(name)
            if lean is None and not args.all:
                continue
            regs_m = re.search(r"Used (\d+) registers", body)
            spill_m = re.search(
                r"(\d+) bytes spill stores, (\d+) bytes spill loads", body
            )
            regs = int(regs_m.group(1)) if regs_m else -1
            spill = (
                (int(spill_m.group(1)), int(spill_m.group(2))) if spill_m else (0, 0)
            )
            cap = ""
            flag = ""
            if lean is not None:
                fmax, r, _dtype, emit = (
                    int(lean.group(1)),
                    int(lean.group(2)),
                    lean.group(3),
                    lean.group(4) == "true",
                )
                cap_n = _register_cap(_min_blocks_per_sm(fmax, r))
                cap = str(cap_n)
                if regs > cap_n:
                    flag = "  OVER CAP"
                    failures += 1
                tolerance = EMIT_SPILL_TOLERANCE if emit else DENSE_SPILL_TOLERANCE
                if any(spill):
                    flag += "  spill (emit)" if emit else "  spill (dense)"
                    if sum(spill) > tolerance:
                        flag += " OVER TOLERANCE"
                        failures += 1
            rows += 1
            spill_s = f"{spill[0]:6d}/{spill[1]:<6d}"
            cells = f"{module:6s} sm_{sm:3s} {regs:4d} {cap:>4s} {spill_s}"
            print(f"{cells}  {name[:100]}{flag}")
    if rows == 0:
        print(
            "no ptxas output captured: the extensions were already built; rerun with "
            "--rebuild to get a fresh report"
        )
        return 2
    print(f"\n{rows} entries, {failures} failures")
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
