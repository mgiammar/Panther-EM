import statistics, sys, torch
sys.path.insert(0, "scratch/svd_search_opt/kernels")
import exp_loader
from panther_em.inference.search import fused_kernel_loader as fkl
from panther_em.inference.search.statistics import decode_argmax_packed
dev=torch.device("cuda:0"); torch.cuda.set_device(dev)
mod=exp_loader.load()
def bench(fn,reps=15,inner=5):
    for _ in range(5): fn()
    torch.cuda.synchronize()
    a,b=torch.cuda.Event(True),torch.cuda.Event(True); ts=[]
    for _ in range(reps):
        a.record()
        for _ in range(inner): fn()
        b.record(); torch.cuda.synchronize(); ts.append(a.elapsed_time(b)/inner)
    return statistics.median(ts)
P,N,K=512,2048,64
g=torch.Generator(device=dev).manual_seed(0)
c=torch.complex(torch.randn(K,P,N,generator=g,device=dev),torch.randn(K,P,N,generator=g,device=dev)); c[0].imag.zero_()
cp=c.permute(1,2,0).contiguous()
for npsi in (128,256):
    ref=fkl.fused_irfft_stats_transposed(c,npsi,decode=True)
    print(f"\n=== n_psi={npsi}  (production config fpb=8 ept=8) ===")
    print(f"{'fpb':>4}{'ept':>4} | {'transposed ms':>14}{'Gcorr/s':>9} | {'regular ms':>11}{'Gcorr/s':>9} | check")
    rows=[]
    for (psi,freq,ept,fpb) in sorted(set(mod.get_supported_configs())):
        if psi!=npsi: continue
        try:
            out=mod.fused_irfft_stats_transposed(c,npsi,fpb,ept)
            s1,s2,pk=out; vmax,amax=decode_argmax_packed(pk)
            ok = torch.allclose(s1,ref[0],rtol=1e-4,atol=1e-2) and torch.allclose(vmax,ref[2],rtol=1e-4,atol=1e-3) and int((amax!=ref[3]).sum())==0
            tt=bench(lambda: mod.fused_irfft_stats_transposed(c,npsi,fpb,ept))
            tr=bench(lambda: mod.fused_irfft_stats(cp,npsi,fpb,ept))
            print(f"{fpb:>4}{ept:>4} | {tt:14.3f}{P*N*npsi/tt/1e6:9.1f} | {tr:11.3f}{P*N*npsi/tr/1e6:9.1f} | {'ok' if ok else 'MISMATCH'}")
        except Exception as e:
            print(f"{fpb:>4}{ept:>4} | failed: {str(e).splitlines()[0][:90]}")
