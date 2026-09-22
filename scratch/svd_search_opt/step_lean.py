"""Full inner step with the lean kernel: fp16 4M GEMM (fp16 out) -> lean reduce. Serial and CUDA-graphed."""
import statistics, sys, os, torch
sys.path.insert(0,"scratch/svd_search_opt/kernels")
from torch.utils.cpp_extension import load
from panther_em.inference.search import fused_kernel_loader as fkl
dev=torch.device("cuda:0"); torch.cuda.set_device(dev)
torch.backends.cuda.matmul.allow_fp16_reduced_precision_reduction=False
HERE="scratch/svd_search_opt/kernels"; _,gencode=fkl._arch_flags()
lean=load(name="panther_lean_irfft_stats", sources=[os.path.join(HERE,"lean_binding.cu")], extra_include_paths=[HERE],
          extra_cuda_cflags=["-O3","-std=c++17",*gencode], extra_cflags=["-O3","-std=c++17"], verbose=False)
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
def make(P,N):
    g=torch.Generator(device=dev).manual_seed(0)
    A16=torch.randn(K,P,2*M,generator=g,device=dev).half(); B16=torch.randn(K,2*M,2*N,generator=g,device=dev).half()
    return A16,B16
print(f"{'P':>5}{'N':>6}{'C16MB':>7}{'B16MB':>7} | {'gemm':>7}{'reduce':>7}{'step':>7}{'Gc/s':>7} | {'graph':>7}{'Gc/s':>7} | {'graph2s':>8}{'Gc/s':>7}")
shapes=[(512,2048),(256,2048),(128,2048),(512,1024),(256,1024),(128,1024),(64,1024),(512,512),(256,512),(128,512),(1024,512),(1024,256),(2048,256),(512,256),(256,256),(4096,128),(2048,128)]
for npsi in (256,128):
    print(f"--- n_psi={npsi} ---")
    for P,N in shapes:
        A16,B16=make(P,N)
        C16=torch.bmm(A16,B16)
        tg=bench(lambda: torch.bmm(A16,B16)); tr=bench(lambda: lean.lean_transposed(C16,npsi))
        def step(): return lean.lean_transposed(torch.bmm(A16,B16),npsi)
        ts=bench(step)
        NB=8
        s=torch.cuda.Stream(); s.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(s):
            for _ in range(3): step()
        torch.cuda.current_stream().wait_stream(s); torch.cuda.synchronize()
        gph=torch.cuda.CUDAGraph()
        with torch.cuda.graph(gph, stream=s):
            for _ in range(NB): step()
        tgr=bench(lambda: gph.replay(), reps=11, inner=2)/NB
        # two-stream graph: alternate steps on two streams (GEMM of one overlaps reduce of other)
        s2=torch.cuda.Stream()
        gph2=torch.cuda.CUDAGraph()
        with torch.cuda.graph(gph2, stream=s):
            s2.wait_stream(s)
            for i in range(NB):
                with torch.cuda.stream(s if i%2==0 else s2): step()
            s.wait_stream(s2)
        tgr2=bench(lambda: gph2.replay(), reps=11, inner=2)/NB
        corr=P*N*npsi
        print(f"{P:>5}{N:>6}{P*N*K*4/1e6:>7.0f}{K*2*M*2*N*2/1e6:>7.0f} | {tg:7.3f}{tr:7.3f}{ts:7.3f}{corr/ts/1e6:7.0f} | {tgr:7.3f}{corr/tgr/1e6:7.0f} | {tgr2:8.3f}{corr/tgr2/1e6:7.0f}")
        del gph,gph2
