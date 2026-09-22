import statistics, sys, os, torch
sys.path.insert(0,"scratch/svd_search_opt/kernels"); import exp_loader
from torch.utils.cpp_extension import load
from panther_em.inference.search import fused_kernel_loader as fkl
from panther_em.inference.search.statistics import decode_argmax_packed
dev=torch.device("cuda:0"); torch.cuda.set_device(dev)
HERE="scratch/svd_search_opt/kernels"
incs=fkl._find_cufftdx_includes(); defines,gencode=fkl._arch_flags()
lean=load(name="panther_lean_irfft_stats", sources=[os.path.join(HERE,"lean_binding.cu")],
    extra_include_paths=[HERE], extra_cuda_cflags=["-O3","-std=c++17","-Xptxas","-v",*gencode], extra_cflags=["-O3","-std=c++17"],
    verbose=True)
m2=exp_loader.load(name="panther_exp2_irfft_stats", src="irfft_stats_exp2.cu")
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
print("\n=== correctness vs production cuFFTDx kernel (fp32 complex input) ===")
for (P,Q,F) in [(37,1000,64),(512,2048,64),(8,300,48),(8,300,32),(8,300,16),(3,5,64)]:
    c=torch.complex(torch.randn(F,P,Q,generator=g,device=dev),torch.randn(F,P,Q,generator=g,device=dev)); c[0].imag.zero_()
    c16=torch.view_as_real(c.permute(0,1,2)).reshape(F,P,2*Q).half().contiguous()
    for npsi in (128,256):
        s1r,s2r,vr,ar=fkl.fused_irfft_stats_transposed(c,npsi,decode=True)
        s1,s2,pk=lean.lean_transposed(c,npsi); v,a=decode_argmax_packed(pk)
        s1h,s2h,pkh=lean.lean_transposed(c16,npsi); vh,ah=decode_argmax_packed(pkh)
        print(f"P={P:4d} Q={Q:5d} F={F:2d} n_psi={npsi}: f32 lean vs ref  s1 {cerr(s1,s1r):.1e} s2 {cerr(s2,s2r):.1e} vmax {cerr(v,vr):.1e} amax mism {int((a!=ar).sum())}/{P} | f16-in: vmax {cerr(vh,vr):.1e} mism {int((ah!=ar).sum())}")
print("\n=== timing (fp16 input): lean vs cuFFTDx-best ===")
for npsi,cfg in ((128,(16,16)),(256,(16,32))):
    for (P,Q) in [(512,2048),(256,2048),(128,1024),(512,512),(1024,2048)]:
        c16=torch.randn(64,P,2*Q,device=dev).half()
        tl=bench(lambda: lean.lean_transposed(c16,npsi)); te=bench(lambda: m2.transposed_f16(c16,npsi,*cfg))
        corr=P*Q*npsi
        print(f"n_psi={npsi} P={P:4d} Q={Q:5d}: lean {tl:7.3f} ms {corr/tl/1e6:7.1f} Gcorr/s | cufftdx{cfg} {te:7.3f} ms {corr/te/1e6:7.1f} Gcorr/s | {te/tl:.2f}x")
