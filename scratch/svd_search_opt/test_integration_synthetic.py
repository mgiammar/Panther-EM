"""Integrated search machinery on synthetic features: fp32 vs fp16 contraction, in-kernel
accumulate vs torch reference, CUDA-graph hypothesis loop vs eager. No reconstructor needed."""
import statistics, time, torch
from panther_em.inference.search.tiling import FeatureTiling
from panther_em.inference.search.statistics import PixelStats
from panther_em.inference.search.fused_statistics import FusedPixelStats
from panther_em.inference.search.compressed import _HypLoopGraph
dev=torch.device("cuda:0"); torch.cuda.set_device(dev)
torch.backends.cuda.matmul.allow_fp16_reduced_precision_reduction=False
g=torch.Generator(device=dev).manual_seed(0)

def make(P, NH, tiling, scale=1.0):
    r=tiling.num_features
    Y=torch.complex(torch.randn(P,r,generator=g,device=dev),torch.randn(P,r,generator=g,device=dev))*scale
    W=torch.complex(torch.randn(NH,r,generator=g,device=dev),torch.randn(NH,r,generator=g,device=dev))
    dc=(tiling.k_indices==0)
    Y[:,dc]=Y[:,dc].real.to(Y.dtype); W[:,dc]=W[:,dc].real.to(W.dtype)
    return Y,W

def run_eager(tiling, Y, W, n_psi, hyp_batch, precision, stats_cls):
    n_freq=tiling.k_stop; NH=W.shape[0]
    hb=[(h0, tiling.prepare_weights(W[h0:h0+hyp_batch], precision=precision)) for h0 in range(0,NH,hyp_batch)]
    st=stats_cls(Y.shape[0], device=dev)
    feats=tiling.prepare_features(Y, precision=precision)
    for h0,pw in hb:
        C=tiling.run(None,None,n_freq,prepared_weights=pw,prepared_features=feats,precision=precision)
        st.update_graphed(C,h0,num_psi=n_psi,reverse_psi_axis=True)
    return st.finalize()

def run_reference(tiling, Y, W, n_psi, hyp_batch):
    """Pure-torch: fp32 cgemm + torch irfft + compiled reduce + eager update()."""
    n_freq=tiling.k_stop; NH=W.shape[0]; st=PixelStats(Y.shape[0],device=dev)
    for h0 in range(0,NH,hyp_batch):
        pw=tiling.prepare_weights(W[h0:h0+hyp_batch])
        C=tiling.run(Y,None,n_freq,prepared_weights=pw)
        st.update(C, torch.arange(h0,h0+pw[0].shape[0],device=dev), n_psi, reverse_psi_axis=True)
    return st.finalize()

def compare(a, b, label):
    mip=((a["mip"]-b["mip"]).abs().max()/b["mip"].abs().max()).item()
    z=(a["zscore"]-b["zscore"]).abs().max().item()
    mean=((a["mean"]-b["mean"]).abs().max()/b["mean"].abs().max().clamp_min(1e-12)).item()
    var=((a["variance"]-b["variance"]).abs().max()/b["variance"].abs().max()).item()
    idx=(a["best_index"]==b["best_index"]).float().mean().item(); psi=(a["best_psi"]==b["best_psi"]).float().mean().item()
    print(f"  {label:34s} mip rel {mip:.1e} | zscore abs {z:.1e} | mean rel {mean:.1e} | var rel {var:.1e} | best_index agree {idx*100:.2f}% | best_psi agree {psi*100:.2f}%")

for rects in ([(0,0,64,64)], [(0,0,64,32),(0,32,32,32)]):
    tiling=FeatureTiling.from_extents(rects, device=dev)
    P,NH,hyp_batch=512,2048,512
    Y,W=make(P,NH,tiling)
    for n_psi in (256,128):
        print(f"\n=== rects={rects} n_freq={tiling.k_stop} r={tiling.num_features} P={P} NH={NH} hyp_batch={hyp_batch} n_psi={n_psi} ===")
        ref=run_reference(tiling,Y,W,n_psi,hyp_batch)
        for precision in ("fp32","tf32","fp16"):
            out=run_eager(tiling,Y,W,n_psi,hyp_batch,precision,FusedPixelStats)
            compare(out,ref,f"FusedPixelStats eager {precision}")
        # graph path (fp16 and fp32)
        for precision in ("fp32","fp16"):
            n_freq=tiling.k_stop
            hb=[(torch.arange(h0,min(h0+hyp_batch,NH),device=dev), tiling.prepare_weights(W[h0:h0+hyp_batch],precision=precision), h0) for h0 in range(0,NH,hyp_batch)]
            st=FusedPixelStats(P,device=dev)
            gr=_HypLoopGraph.try_capture(tiling,hb,st,Y,n_freq=n_freq,n_psi=n_psi,precision=precision)
            if gr is None: print(f"  graph capture {precision}: NOT captured"); continue
            gr.run(Y); out=st.finalize()
            compare(out,ref,f"CUDA graph {precision}")
            # replay again with a different Y to make sure state resets correctly
            Y2,_=make(P,NH,tiling); ref2=run_reference(tiling,Y2,W,n_psi,hyp_batch); gr.run(Y2); out2=st.finalize()
            compare(out2,ref2,f"CUDA graph {precision} (2nd replay)")
        # timing
        corr=P*NH*n_psi
        def t(fn,reps=5):
            fn(); torch.cuda.synchronize(); ts=[]
            for _ in range(reps):
                t0=time.perf_counter(); fn(); torch.cuda.synchronize(); ts.append(time.perf_counter()-t0)
            return statistics.median(ts)
        tref=t(lambda: run_reference(tiling,Y,W,n_psi,hyp_batch),reps=2)
        print(f"  timing: torch reference (cgemm+irfft+compiled reduce) {tref*1e3:8.2f} ms {corr/tref/1e9:7.1f} Gcorr/s")
        for precision in ("fp32","fp16"):
            te=t(lambda: run_eager(tiling,Y,W,n_psi,hyp_batch,precision,FusedPixelStats))
            hb=[(torch.arange(h0,min(h0+hyp_batch,NH),device=dev), tiling.prepare_weights(W[h0:h0+hyp_batch],precision=precision), h0) for h0 in range(0,NH,hyp_batch)]
            st=FusedPixelStats(P,device=dev); gr=_HypLoopGraph.try_capture(tiling,hb,st,Y,n_freq=tiling.k_stop,n_psi=n_psi,precision=precision)
            tg=t(lambda: gr.run(Y)) if gr else float('nan')
            print(f"  timing: {precision} eager(incl. weight prep) {te*1e3:8.2f} ms {corr/te/1e9:7.1f} Gcorr/s | graph replay {tg*1e3:8.2f} ms {corr/tg/1e9:7.1f} Gcorr/s")
