import warnings, statistics, torch
warnings.simplefilter("always")
from panther_em.inference.search import fused_kernel_loader as fkl
from panther_em.inference.search.fused_statistics import FusedPixelStats
dev=torch.device("cuda:0"); torch.cuda.set_device(dev)
mod=fkl._try_compile()
def bench(fn,reps=9,inner=5):
    for _ in range(3): fn()
    torch.cuda.synchronize(); a,b=torch.cuda.Event(True),torch.cuda.Event(True); ts=[]
    for _ in range(reps):
        a.record()
        for _ in range(inner): fn()
        b.record(); torch.cuda.synchronize(); ts.append(a.elapsed_time(b)/inner)
    return statistics.median(ts)
F,P,Q=64,512,2048
base=torch.complex(torch.randn(F,P,Q,device=dev),torch.randn(F,P,Q,device=dev)); base[0].imag.zero_()
spec=base.permute(1,2,0)
print("module lean raw      :", bench(lambda: mod.lean_irfft_stats_transposed(base,256,0,[])), "ms")
print("module cufftdx raw   :", bench(lambda: mod.fused_irfft_stats_transposed(base,256)), "ms")
print("loader lean decode=F :", bench(lambda: fkl.lean_irfft_stats_transposed(base,256,decode=False)), "ms")
print("loader lean decode=T :", bench(lambda: fkl.lean_irfft_stats_transposed(base,256,decode=True)), "ms")
fs=FusedPixelStats(P,device=dev)
r=fs._try_fused_reduce(spec,256,decode=True); print("try_fused_reduce returned", None if r is None else [t.dtype for t in r])
print("FusedPixelStats._reduce:", bench(lambda: fs._reduce(spec,256)), "ms")
print("decode alone         :", bench(lambda: fkl.decode_argmax_packed(torch.zeros(P,dtype=torch.int64,device=dev))), "ms")
import inspect; print(inspect.getsource(fs._try_fused_reduce)[:400])
