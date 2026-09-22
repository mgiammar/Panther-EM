"""Real-data validation + end-to-end throughput of compressed_search configurations.

Reuses profile_svd_search.py's builders (60S map decomposition cached under scratch/).
Compares maps between the fp32 baseline and the optimized configs, and times each.
"""
from __future__ import annotations
import sys, time, statistics, gc, os
sys.path.insert(0, "experiments/profiling_svd_2dtm_search")
import click, torch
from profile_svd_search import SEARCH_RECTANGLES, build_reconstructor, build_search_images
from panther_em.inference.search import compressed_search
from panther_em.inference.search.compressed import resolve_search_args
from panther_em.inference.search.tiling import FeaturizedImageStore

@click.command()
@click.option("--num-particles", default=2)
@click.option("--num-hyps", default=4096, help="Restrict to the first N hypotheses (0 = all).")
@click.option("--n-psis", default="256,128")
@click.option("--pixel-batch", default=512)
@click.option("--hyp-batch", default=512)
@click.option("--repeats", default=2)
@click.option("--svd-cache", default="scratch/svd_search_opt/svd_60S_k160_e128.h5")
@click.option("--rectangles", default="0,0,64,64")
def main(num_particles, num_hyps, n_psis, pixel_batch, hyp_batch, repeats, svd_cache, rectangles):
    dev=torch.device("cuda:0"); torch.cuda.set_device(dev)
    torch.backends.cuda.matmul.allow_fp16_reduced_precision_reduction=False
    rects=[tuple(int(v) for v in r.split(",")) for r in rectangles.split(";")]
    t0=time.perf_counter(); recon=build_reconstructor(dev, svd_cache); print(f"reconstructor ready ({time.perf_counter()-t0:.0f}s)")
    imgs=build_search_images(num_particles, -1, dev); torch.cuda.synchronize()
    n_px,hyp_all,_=resolve_search_args(imgs, recon, None, None)
    hyps=hyp_all[:num_hyps] if num_hyps>0 else hyp_all
    print(f"images {tuple(imgs.shape)} -> n_px={n_px}, hypotheses {int(hyps.numel())} of {int(hyp_all.numel())}, rects={rects}")
    store=FeaturizedImageStore(n_px, device=dev)
    gc.collect(); torch.cuda.empty_cache()
    configs=[("fp32 baseline (lean kernel)", dict(precision="fp32", use_cuda_graph=False)),
             ("fp32 + cuda graph",            dict(precision="fp32", use_cuda_graph=True)),
             ("fp16",                         dict(precision="fp16", use_cuda_graph=False)),
             ("fp16 + cuda graph",            dict(precision="fp16", use_cuda_graph=True))]
    for n_psi in [int(v) for v in n_psis.split(",")]:
        print(f"\n===== n_psi={n_psi} pixel_batch={pixel_batch} hyp_batch={hyp_batch} =====")
        def run(**kw):
            return compressed_search(imgs, recon, rects, pixel_batch=pixel_batch, hyp_batch=hyp_batch, n_psi=n_psi,
                                     feature_chunk=32, hypothesis_indexes=hyps, store=store, show_progress=False, **kw)
        run(precision="fp32")  # featurize the persistent store once (excluded from timing)
        torch.cuda.synchronize()
        ref=None; corr=n_px*int(hyps.numel())*n_psi
        for name,kw in configs:
            maps=run(**kw); torch.cuda.synchronize()
            ts=[]
            for _ in range(repeats):
                t0=time.perf_counter(); maps=run(**kw); torch.cuda.synchronize(); ts.append(time.perf_counter()-t0)
            t=statistics.median(ts)
            line=f"{name:30s} {t*1e3:9.1f} ms  {corr/t/1e9:7.1f} Gcorr/s"
            if ref is None:
                ref=maps
                line+=f"   |mip| max {ref['mip'].abs().max().item():.3g}, zscore max {ref['zscore'].max().item():.2f}"
            else:
                mip=((maps["mip"]-ref["mip"]).abs().max()/ref["mip"].abs().max()).item()
                z=(maps["zscore"]-ref["zscore"]).abs().max().item()
                zrel=z/ref["zscore"].abs().max().item()
                bi=(maps["best_index"]==ref["best_index"]).float().mean().item(); bp=(maps["best_psi"]==ref["best_psi"]).float().mean().item()
                # do mismatched pixels have (near-)tied alternatives? look at their mip difference
                mism=maps["best_index"]!=ref["best_index"]
                worst=(maps["mip"][mism]-ref["mip"][mism]).abs().max().item() if mism.any() else 0.0
                line+=f" | vs baseline: mip rel {mip:.1e}, zscore abs {z:.2e} (rel {zrel:.1e}), best_index agree {bi*100:.3f}%, best_psi agree {bp*100:.3f}%, max |dmip| at mismatches {worst:.2e}"
            print(line, flush=True)
if __name__=="__main__":
    main()
