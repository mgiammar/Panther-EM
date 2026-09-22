"""Build the experimental fused kernel (wide EPT/FPB config grid) as a separate extension."""
import os, functools, torch
from panther_em.inference.search import fused_kernel_loader as fkl
HERE=os.path.dirname(os.path.abspath(__file__))
@functools.lru_cache(maxsize=1)
def load(name="panther_exp_irfft_stats", src="irfft_stats_exp.cu"):
    from torch.utils.cpp_extension import load as _load
    incs=fkl._find_cufftdx_includes(); defines,gencode=fkl._arch_flags()
    return _load(name=name, sources=[os.path.join(HERE,src)],
        extra_include_paths=[HERE, os.path.join(HERE,"include"), *incs],
        extra_cuda_cflags=["-O3","-std=c++17","--expt-relaxed-constexpr","-U__CUDA_NO_HALF_OPERATORS__","-U__CUDA_NO_HALF_CONVERSIONS__","-U__CUDA_NO_HALF2_OPERATORS__","-U__CUDA_NO_BFLOAT16_CONVERSIONS__",*defines,*gencode],
        extra_cflags=["-O3","-std=c++17"], verbose=bool(int(os.environ.get("EXP_VERBOSE","0"))))
