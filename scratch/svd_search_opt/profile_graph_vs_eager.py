"""Per-kernel CUDA time for one pixel batch's hypothesis loop: eager vs CUDA graph (1 and 2 streams)."""
import sys, statistics, torch
from torch.profiler import profile, ProfilerActivity
from panther_em.inference.search.tiling import FeatureTiling
from panther_em.inference.search.fused_statistics import FusedPixelStats
from panther_em.inference.search.compressed import _HypLoopGraph
dev=torch.device("cuda:0"); torch.cuda.set_device(dev)
torch.backends.cuda.matmul.allow_fp16_reduced_precision_reduction=False
print("torch", torch.__version__)
g=torch.Generator(device=dev).manual_seed(0)
P,NH,HB,n_psi=int(sys.argv[1]) if len(sys.argv)>1 else 512, 2048, int(sys.argv[2]) if len(sys.argv)>2 else 512, 256
tiling=FeatureTiling.from_extents([(0,0,64,64)],device=dev); r=tiling.num_features
Y=torch.complex(torch.randn(P,r,generator=g,device=dev),torch.randn(P,r,generator=g,device=dev))
W=torch.complex(torch.randn(NH,r,generator=g,device=dev),torch.randn(NH,r,generator=g,device=dev))
dc=(tiling.k_indices==0); Y[:,dc]=Y[:,dc].real.to(Y.dtype); W[:,dc]=W[:,dc].real.to(W.dtype)
hb=[(torch.arange(h0,min(h0+HB,NH),device=dev), tiling.prepare_weights(W[h0:h0+HB],precision="fp16"), h0) for h0 in range(0,NH,HB)]
st=FusedPixelStats(P,device=dev)
def eager():
    st.clear(); feats=tiling.prepare_features(Y,precision="fp16")
    for _,pw,h0 in hb:
        C=tiling.run(None,None,64,prepared_weights=pw,prepared_features=feats,precision="fp16")
        st.update_graphed(C,h0,num_psi=n_psi,reverse_psi_axis=True)
    st.finalize()
graphs={k:_HypLoopGraph.try_capture(tiling,hb,FusedPixelStats(P,device=dev),Y,n_freq=64,n_psi=n_psi,precision="fp16",num_streams=k) for k in (1,2)}
def timed(fn,reps=20):
    for _ in range(3): fn()
    torch.cuda.synchronize(); a,b=torch.cuda.Event(True),torch.cuda.Event(True); ts=[]
    for _ in range(reps):
        a.record(); fn(); b.record(); torch.cuda.synchronize(); ts.append(a.elapsed_time(b))
    return statistics.median(ts)
corr=P*NH*n_psi
for name,fn in (("eager",eager),("graph 1-stream",lambda: graphs[1].run(Y)),("graph 2-stream",lambda: graphs[2].run(Y))):
    t=timed(fn); print(f"\n=== {name}: {t:.3f} ms/pixel-batch  ({corr/t/1e6:.0f} Gcorr/s) ===")
    with profile(activities=[ProfilerActivity.CUDA]) as prof:
        for _ in range(5): fn()
        torch.cuda.synchronize()
    agg={}
    for e in prof.events():
        if e.device_type.name!="CUDA": continue
        k=e.name[:70]; d=agg.setdefault(k,[0.0,0]); d[0]+=e.device_time/1000.0/5; d[1]+=1
    for k,(ms,n) in sorted(agg.items(), key=lambda kv:-kv[1][0])[:8]:
        print(f"  {ms:8.3f} ms  x{n//5:<3d} {k}")
