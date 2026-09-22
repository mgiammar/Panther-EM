"""How fast can the contraction go? cuBLAS cgemm vs TF32/BF16 real-GEMM formulations."""
import statistics, torch
dev=torch.device("cuda:0"); torch.cuda.set_device(dev)
g=torch.Generator(device=dev).manual_seed(0)
def bench(fn,reps=15,inner=5):
    for _ in range(5): fn()
    torch.cuda.synchronize()
    a,b=torch.cuda.Event(True),torch.cuda.Event(True); ts=[]
    for _ in range(reps):
        a.record()
        for _ in range(inner): fn()
        b.record(); torch.cuda.synchronize(); ts.append(a.elapsed_time(b)/inner)
    return statistics.median(ts)

P,N,K,M=512,2048,64,64
flops=8*P*N*K*M   # complex MACs
print(f"P={P} N={N} K={K} M={M}: complex flops per contraction = {flops/1e9:.1f} GFLOP")
Y=torch.complex(torch.randn(K,P,M,generator=g,device=dev),torch.randn(K,P,M,generator=g,device=dev))
W=torch.complex(torch.randn(K,M,N,generator=g,device=dev),torch.randn(K,M,N,generator=g,device=dev))
for tf32 in (False,True):
    torch.backends.cuda.matmul.allow_tf32=tf32
    t=bench(lambda: torch.bmm(Y,W)); print(f"cgemm complex64  allow_tf32={tf32!s:5}: {t:7.3f} ms  {flops/t/1e9:8.1f} GFLOP/s")
torch.backends.cuda.matmul.allow_tf32=False
# 4M real formulation: [Yr Yi] @ [[Wr, Wi],[-Wi, Wr]] = [Cr Ci]   -> (K, P, 2M) @ (K, 2M, 2N)
Y4=torch.cat([Y.real,Y.imag],dim=-1).contiguous()                       # (K,P,2M)
W4=torch.cat([torch.cat([W.real,W.imag],-1), torch.cat([-W.imag,W.real],-1)],dim=1).contiguous()  # (K,2M,2N)
ref=torch.bmm(Y,W)
for name,dt,tf32 in (("fp32 real 4M",torch.float32,False),("tf32 real 4M",torch.float32,True),("bf16 real 4M",torch.bfloat16,False),("fp16 real 4M",torch.float16,False)):
    torch.backends.cuda.matmul.allow_tf32=tf32
    y,w=Y4.to(dt),W4.to(dt)
    t=bench(lambda: torch.bmm(y,w))
    out=torch.bmm(y,w).float(); C=torch.complex(out[...,:N],out[...,N:])
    err=((C-ref).abs().max()/ref.abs().max()).item()
    print(f"{name:14s} allow_tf32={tf32!s:5}: {t:7.3f} ms  {flops/t/1e9:8.1f} GFLOP/s   max rel err {err:.2e}")
# 3M (Gauss) trick on real GEMMs: 3 real GEMMs of (P,M)x(M,N)
torch.backends.cuda.matmul.allow_tf32=True
Yr,Yi,Wr,Wi=Y.real.contiguous(),Y.imag.contiguous(),W.real.contiguous(),W.imag.contiguous()
def gauss():
    k1=torch.bmm(Wr, ) if False else None
def three_m():
    t1=torch.bmm(Yr,Wr); t2=torch.bmm(Yi,Wi); t3=torch.bmm(Yr+Yi, Wr+Wi)
    return t1-t2, t3-t1-t2
t=bench(three_m); print(f"tf32 3M (Gauss) : {t:7.3f} ms  {flops/t/1e9:8.1f} GFLOP/s (incl. adds)")
# what does a plain big real TF32 GEMM achieve on this GPU? (peak check)
A=torch.randn(8192,8192,device=dev); B=torch.randn(8192,8192,device=dev)
for tf32 in (False,True):
    torch.backends.cuda.matmul.allow_tf32=tf32
    t=bench(lambda: A@B, reps=5, inner=2); print(f"8192^3 real sgemm allow_tf32={tf32!s:5}: {t:7.3f} ms  {2*8192**3/t/1e9:8.1f} GFLOP/s")
Ab,Bb=A.bfloat16(),B.bfloat16()
t=bench(lambda: Ab@Bb, reps=5, inner=2); print(f"8192^3 bf16 gemm               : {t:7.3f} ms  {2*8192**3/t/1e9:8.1f} GFLOP/s")
