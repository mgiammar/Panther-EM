"""Compile the production extension and check the lean path end to end through FusedPixelStats."""
import statistics, torch
from panther_em.inference.search import fused_kernel_loader as fkl
from panther_em.inference.search.fused_statistics import FusedPixelStats
from panther_em.inference.search.statistics import PixelStats, decode_argmax_packed
dev=torch.device("cuda:0"); torch.cuda.set_device(dev)
mod=fkl._try_compile(); assert mod is not None, "compile failed"
print("configs:", fkl.get_supported_configs()[:3], "... lean sentinel:", fkl.lean_sentinel_packed())
def bench(fn,reps=15,inner=5):
    for _ in range(4): fn()
    torch.cuda.synchronize()
    a,b=torch.cuda.Event(True),torch.cuda.Event(True); ts=[]
    for _ in range(reps):
        a.record()
        for _ in range(inner): fn()
        b.record(); torch.cuda.synchronize(); ts.append(a.elapsed_time(b)/inner)
    return statistics.median(ts)
def cerr(a,b): return ((a-b).abs().max()/b.abs().max()).item()
g=torch.Generator(device=dev).manual_seed(0)
for F in (64, 37, 16):
    P,Q=512,2048
    base=torch.complex(torch.randn(F,P,Q,generator=g,device=dev),torch.randn(F,P,Q,generator=g,device=dev)); base[0].imag.zero_()
    spec=base.permute(1,2,0)                                   # (P,Q,F) bmm-layout view
    spec16=torch.view_as_complex(torch.view_as_real(base).half()).permute(1,2,0)   # complex32 view, same layout
    for npsi in (128,256):
        ref=PixelStats(P,device=dev)._reduce(spec,npsi)
        fs=FusedPixelStats(P,device=dev)
        out=fs._reduce(spec,npsi); out16=fs._reduce(spec16,npsi)
        # which kernel ran? time it
        t=bench(lambda: fs._reduce(spec,npsi)); t16=bench(lambda: fs._reduce(spec16,npsi)); tt=bench(lambda: PixelStats(P,device=dev)._reduce(spec,npsi))
        print(f"F={F:2d} n_psi={npsi}: c64 lean vs torch: vmax {cerr(out[2],ref[2]):.1e} s2 {cerr(out[1],ref[1]):.1e} mism {int((out[3]!=ref[3]).sum())} | c32: vmax {cerr(out16[2],ref[2]):.1e} mism {int((out16[3]!=ref[3]).sum())} | fused c64 {t:.3f} ms, c32 {t16:.3f} ms, torch {tt:.3f} ms")
# accumulate mode across 4 hyp batches == one-shot over concatenated batch
print("\n--- accumulate mode (4 batches, hyp_offset) vs single call ---")
F,P,Q=64,256,4*300
base=torch.complex(torch.randn(F,P,Q,generator=g,device=dev),torch.randn(F,P,Q,generator=g,device=dev)); base[0].imag.zero_()
for npsi in (128,256):
    s1a,s2a,pka=fkl.lean_irfft_stats_transposed(base.contiguous(),npsi,decode=False)
    outs=[torch.zeros(P,device=dev),torch.zeros(P,device=dev),torch.full((P,),fkl.lean_sentinel_packed(),dtype=torch.int64,device=dev)]
    for i in range(4):
        r=fkl.lean_irfft_stats_transposed(base[:,:,i*300:(i+1)*300].contiguous(),npsi,decode=False,hyp_offset=i*300,outs=outs)
        assert r[0] is outs[0]
    va,ia=decode_argmax_packed(pka); vb,ib=decode_argmax_packed(outs[2])
    print(f"n_psi={npsi}: s1 {cerr(outs[0],s1a):.1e} s2 {cerr(outs[1],s2a):.1e} vmax {cerr(vb,va):.1e} idx mismatches {int((ia!=ib).sum())}/{P}")
