"""Per-hyp-batch pipeline options at the production shape (run ALONE on the GPU).

4M real GEMM with interleaved output columns so out (k,P,2N) real == (k,P,N) complex.
For C = Y @ W (W already conjugated):  Cr = Yr Wr - Yi Wi,  Ci = Yr Wi + Yi Wr
  A = [Yr | Yi] (k,P,2M);  B[:, 2n] = [Wr[:,n]; -Wi[:,n]],  B[:, 2n+1] = [Wi[:,n]; Wr[:,n]]
"""
import statistics, sys, torch
sys.path.insert(0, "scratch/svd_search_opt/kernels")
import exp_loader
from panther_em.inference.search import fused_kernel_loader as fkl
from panther_em.inference.search.statistics import decode_argmax_packed
dev=torch.device("cuda:0"); torch.cuda.set_device(dev)
g=torch.Generator(device=dev).manual_seed(0)
torch.backends.cuda.matmul.allow_fp16_reduced_precision_reduction=False
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

P,N,K,M=int(sys.argv[1]),int(sys.argv[2]),64,64
BEST={128:(8,16),256:(4,32)}   # (fpb,ept) from the EPT sweep
print(f"P={P} N={N} K={K} M={M}")
Y=torch.complex(torch.randn(K,P,M,generator=g,device=dev),torch.randn(K,P,M,generator=g,device=dev))
W=torch.complex(torch.randn(K,M,N,generator=g,device=dev),torch.randn(K,M,N,generator=g,device=dev))
Y[0].imag.zero_(); W[0].imag.zero_()

def c_cgemm(): return torch.bmm(Y,W)
A16=torch.cat([Y.real,Y.imag],-1).half().contiguous()
Wr,Wi=W.real,W.imag
B=torch.empty(K,2*M,2*N,device=dev)
B[:,:M,0::2]=Wr; B[:,M:,0::2]=-Wi; B[:,:M,1::2]=Wi; B[:,M:,1::2]=Wr
B16=B.half().contiguous(); A32,B32=A16.float(),B16.float()
def c_4m_fp16_to_fp32(): return torch.view_as_complex(torch.bmm(A16,B16,out_dtype=torch.float32).view(K,P,N,2))
def c_4m_fp16_to_fp16(): return torch.bmm(A16,B16)
def c_4m_tf32(): return torch.view_as_complex(torch.bmm(A32,B32).view(K,P,N,2))

torch.backends.cuda.matmul.allow_tf32=False
ref=c_cgemm()
def relerr(c): return ((c.float()-ref).abs().max()/ref.abs().max()).item()
print("\n--- contraction (accuracy vs fp32 cgemm) ---")
t=bench(c_cgemm); print(f"cgemm fp32            {t:7.3f} ms")
torch.backends.cuda.matmul.allow_tf32=True
t=bench(c_cgemm); print(f"cgemm tf32            {t:7.3f} ms  relerr {relerr(c_cgemm()):.1e}")
t=bench(c_4m_tf32); print(f"4M real tf32          {t:7.3f} ms  relerr {relerr(c_4m_tf32()):.1e}")
torch.backends.cuda.matmul.allow_tf32=False
t=bench(c_4m_fp16_to_fp32); print(f"4M fp16 -> fp32 out   {t:7.3f} ms  relerr {relerr(c_4m_fp16_to_fp32()):.1e}")
t=bench(c_4m_fp16_to_fp16); c16=c_4m_fp16_to_fp16(); print(f"4M fp16 -> fp16 out   {t:7.3f} ms  relerr {relerr(torch.view_as_complex(c16.float().view(K,P,N,2))):.1e}")
print(f"   |C| range: max {ref.abs().max().item():.3g}  median {ref.abs().median().item():.3g}")

print("\n--- reduce kernel, fp32 complex input, best (fpb,ept) ---")
for npsi in (128,256):
    fpb,ept=BEST[npsi]
    t8=bench(lambda: fkl.fused_irfft_stats_transposed(ref,npsi,decode=False))
    tb=bench(lambda: mod.fused_irfft_stats_transposed(ref,npsi,fpb,ept))
    print(f"n_psi={npsi}: prod(8,8) {t8:6.3f} ms {P*N*npsi/t8/1e6:6.1f} Gcorr/s | best{(fpb,ept)} {tb:6.3f} ms {P*N*npsi/tb/1e6:6.1f} Gcorr/s  [{P*N*K*8/tb/1e6:5.0f} GB/s read]")

variants=[("cgemm fp32",c_cgemm,False),("cgemm tf32",c_cgemm,True),("4M fp16->fp32",c_4m_fp16_to_fp32,False)]
print("\n--- step = contraction + reduce(best cfg), single stream ---")
for npsi in (128,256):
    fpb,ept=BEST[npsi]
    for name,fn,tf32 in variants:
        torch.backends.cuda.matmul.allow_tf32=tf32
        def step(): return mod.fused_irfft_stats_transposed(fn(),npsi,fpb,ept)
        t=bench(step); print(f"n_psi={npsi} {name:14s}: {t:7.3f} ms   {P*N*npsi/t/1e6:8.1f} Gcorr/s")

print("\n--- two-stream pipelined (GEMM i+1 || reduce i), 8 hyp batches, best cfg ---")
s_gemm=torch.cuda.Stream(); s_red=torch.cuda.Stream()
for npsi in (128,256):
    fpb,ept=BEST[npsi]
    for name,fn,tf32 in variants:
        torch.backends.cuda.matmul.allow_tf32=tf32
        NB=8
        bufs=[None,None]
        def pipelined():
            cur=torch.cuda.current_stream()
            s_gemm.wait_stream(cur); s_red.wait_stream(cur)
            done=[torch.cuda.Event(),torch.cuda.Event()]; ready=[torch.cuda.Event(),torch.cuda.Event()]
            for i in range(NB):
                slot=i%2
                with torch.cuda.stream(s_gemm):
                    if i>=2: s_gemm.wait_event(done[slot])
                    bufs[slot]=fn(); ready[slot].record(s_gemm)
                with torch.cuda.stream(s_red):
                    s_red.wait_event(ready[slot])
                    mod.fused_irfft_stats_transposed(bufs[slot],npsi,fpb,ept)
                    done[slot].record(s_red)
            cur.wait_stream(s_gemm); cur.wait_stream(s_red)
        def serial():
            for i in range(NB): mod.fused_irfft_stats_transposed(fn(),npsi,fpb,ept)
        ts=bench(serial,reps=7,inner=1)/NB; tp=bench(pipelined,reps=7,inner=1)/NB
        print(f"n_psi={npsi} {name:14s}: serial {ts:7.3f} ms/step  pipelined {tp:7.3f} ms/step  ({ts/tp:.2f}x)  -> {P*N*npsi/tp/1e6:7.1f} Gcorr/s")
