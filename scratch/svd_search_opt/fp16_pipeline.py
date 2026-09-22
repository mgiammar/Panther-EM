"""fp16 end-to-end inner step: 4M fp16 GEMM (fp16 out) -> fp16-input fused kernel.
Accuracy vs the fp32 pipeline, per-shape timing, and CUDA-graph replay of the step."""
import statistics, sys, torch
sys.path.insert(0, "scratch/svd_search_opt/kernels")
import exp_loader
from panther_em.inference.search import fused_kernel_loader as fkl
from panther_em.inference.search.statistics import decode_argmax_packed
dev=torch.device("cuda:0"); torch.cuda.set_device(dev)
torch.backends.cuda.matmul.allow_fp16_reduced_precision_reduction=False
torch.backends.cuda.matmul.allow_tf32=False
m2=exp_loader.load(name="panther_exp2_irfft_stats", src="irfft_stats_exp2.cu")
K,M=64,64
def bench(fn,reps=15,inner=5):
    for _ in range(4): fn()
    torch.cuda.synchronize()
    a,b=torch.cuda.Event(True),torch.cuda.Event(True); ts=[]
    for _ in range(reps):
        a.record()
        for _ in range(inner): fn()
        b.record(); torch.cuda.synchronize(); ts.append(a.elapsed_time(b)/inner)
    return statistics.median(ts)
def make(P,N,seed=0):
    g=torch.Generator(device=dev).manual_seed(seed)
    Y=torch.complex(torch.randn(K,P,M,generator=g,device=dev),torch.randn(K,P,M,generator=g,device=dev))
    W=torch.complex(torch.randn(K,M,N,generator=g,device=dev),torch.randn(K,M,N,generator=g,device=dev))
    Y[0].imag.zero_(); W[0].imag.zero_()
    A16=torch.cat([Y.real,Y.imag],-1).half().contiguous()
    B=torch.empty(K,2*M,2*N,device=dev); Wr,Wi=W.real,W.imag
    B[:,:M,0::2]=Wr; B[:,M:,0::2]=-Wi; B[:,:M,1::2]=Wi; B[:,M:,1::2]=Wr
    return Y,W,A16,B.half().contiguous()
def cerr(a,b): return ((a-b).abs().max()/b.abs().max()).item()

# ---------------- accuracy at production shape ----------------
P,N=512,2048
Y,W,A16,B16=make(P,N)
C32=torch.bmm(Y,W)                                   # (k,P,N) complex64 reference
C16=torch.bmm(A16,B16)                               # (k,P,2N) fp16
C16c=torch.view_as_complex(C16.float().view(K,P,N,2))
print(f"contraction fp16-4M vs fp32 cgemm: rel err {cerr(C16c,C32):.2e}   |C| max {C32.abs().max().item():.1f} median {C32.abs().median().item():.1f}")
for npsi,cfgs in ((128,[(8,16),(4,16),(16,16),(8,32),(8,8)]),(256,[(4,32),(8,32),(16,32),(8,16),(8,8)])):
    s1r,s2r,vr,ar=fkl.fused_irfft_stats_transposed(C32,npsi,decode=True)
    for fpb,ept in cfgs:
        s1,s2,pk=m2.transposed_f16(C16,npsi,fpb,ept); v,a=decode_argmax_packed(pk)
        s1f,s2f,pkf=m2.transposed_f32(C32,npsi,fpb,ept); vf,af=decode_argmax_packed(pkf)
        assert int((af!=ar).sum())==0 and cerr(vf,vr)<1e-5, "f32 exp kernel disagrees with production"
        mism=int((a!=ar).sum())
        # value at the fp16-chosen index in the fp32 correlogram, to see if mismatches are near-ties
        print(f"n_psi={npsi} cfg{(fpb,ept)}: fp16 path vs fp32 path  s1 {cerr(s1,s1r):.1e}  s2 {cerr(s2,s2r):.1e}  vmax {cerr(v,vr):.1e}  amax mismatches {mism}/{P}")
        break  # accuracy is config-independent; timing below

# ---------------- timing across shapes ----------------
print("\n{:>5}{:>6}{:>7} | {:>9}{:>9}{:>8} | {:>9}{:>8} | {:>10}{:>8} | {:>10}{:>8}".format("P","N","C16 MB","gemm16","gemm32c","tf32c","red16","cfg","step16 ms","Gc/s","graph ms","Gc/s"))
shapes=[(512,2048),(1024,2048),(256,2048),(128,2048),(128,1024),(64,2048),(64,1024),(256,1024),(512,512),(128,512)]
for npsi in (128,256):
    print(f"--- n_psi={npsi} ---")
    cfgs=[(8,16),(4,16),(16,16),(8,32)] if npsi==128 else [(4,32),(8,32),(16,32),(8,16)]
    for P,N in shapes:
        Y,W,A16,B16=make(P,N)
        t_g16=bench(lambda: torch.bmm(A16,B16))
        t_g32=bench(lambda: torch.bmm(Y,W))
        torch.backends.cuda.matmul.allow_tf32=True; t_tf=bench(lambda: torch.bmm(Y,W)); torch.backends.cuda.matmul.allow_tf32=False
        C16=torch.bmm(A16,B16)
        best=None
        for fpb,ept in cfgs:
            t=bench(lambda: m2.transposed_f16(C16,npsi,fpb,ept))
            if best is None or t<best[0]: best=(t,(fpb,ept))
        t_red,cfg=best
        def step(): return m2.transposed_f16(torch.bmm(A16,B16),npsi,*cfg)
        t_step=bench(step)
        # CUDA graph of NB steps (static operands; outputs allocated from graph pool)
        NB=4
        s=torch.cuda.Stream(); s.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(s):
            for _ in range(3): step()
        torch.cuda.current_stream().wait_stream(s); torch.cuda.synchronize()
        gph=torch.cuda.CUDAGraph()
        with torch.cuda.graph(gph, stream=s):
            for _ in range(NB): step()
        t_graph=bench(lambda: gph.replay(), reps=15, inner=2)/NB
        corr=P*N*npsi
        print(f"{P:>5}{N:>6}{P*N*K*4/1e6:>7.0f} | {t_g16:9.3f}{t_g32:9.3f}{t_tf:8.3f} | {t_red:9.3f}{str(cfg):>8} | {t_step:10.3f}{corr/t_step/1e6:8.1f} | {t_graph:10.3f}{corr/t_graph/1e6:8.1f}")
        del gph
