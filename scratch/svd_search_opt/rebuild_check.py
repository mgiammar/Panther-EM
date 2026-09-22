import os, statistics, torch
os.environ["PANTHER_EM_FUSED_KERNEL_VERBOSE"]="1"
from panther_em.inference.search import fused_kernel_loader as fkl
dev=torch.device("cuda:0"); torch.cuda.set_device(dev)
mod=fkl._try_compile(); assert mod
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
b16=torch.view_as_complex(torch.view_as_real(base).half())
for npsi in (128,256):
    print(f"n_psi={npsi}: lean c64 {bench(lambda: mod.lean_irfft_stats_transposed(base,npsi,0,[])):.3f} ms | lean c32 {bench(lambda: mod.lean_irfft_stats_transposed(b16,npsi,0,[])):.3f} ms | cufftdx c64 {bench(lambda: mod.fused_irfft_stats_transposed(base,npsi)):.3f} ms")
