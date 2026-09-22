"""Real-data validation + end-to-end throughput of compressed_search configurations.

Reuses profile_svd_search.py's builders (60S map decomposition cached under scratch/).
Compares maps between the fp32 baseline and the optimized configs, and times each.
"""
from __future__ import annotations
import sys, time, statistics, gc, os
sys.path.insert(0, "experiments/profiling_svd_2dtm_search")
import click, torch
from profile_svd_search import (SEARCH_RECTANGLES, HDF5_PARTICLE_STACK, build_reconstructor,
                                build_search_image)
import mrcfile
from leopard_em.pydantic_models.config import PreprocessingFilters
from leopard_em.pydantic_models.data_structures import ParticleStackHDF5
from leopard_em.utils.ctf_utils import (_setup_ctf_kwargs_from_particle_stack,
                                        calculate_ctf_filter_stack_full_args)


def build_search_images(num_particles: int, crop_size: int, device: torch.device) -> torch.Tensor:
    """profile_svd_search.build_search_images with the particle DataFrame re-indexed 0..N-1
    (this Leopard-EM checkout's CTF helper indexes rows by label)."""
    particle_stack = ParticleStackHDF5.from_hdf5(HDF5_PARTICLE_STACK)
    print("particle df index head:", list(particle_stack._df.index[:4]), "rows:", len(particle_stack._df))
    particle_stack._df = particle_stack._df.reset_index(drop=True)
    micrograph_path = particle_stack._df.micrograph_path.iloc[0]
    micrograph_rfft = torch.fft.rfft2(torch.from_numpy(mrcfile.read(micrograph_path))).unsqueeze(0)
    image_stack_rfft = torch.fft.rfft2(particle_stack.image_stack).to(device)
    fourier_shape_particle = (particle_stack.extracted_box_size[0], particle_stack.extracted_box_size[1] // 2 + 1)
    particle_filter = particle_stack.construct_image_filters(PreprocessingFilters(), fourier_shape_particle, micrograph_rfft).to(device)
    particle_box = (int(particle_stack.extracted_box_size[0]), int(particle_stack.extracted_box_size[1]))
    ctf_kwargs_particle = _setup_ctf_kwargs_from_particle_stack(particle_stack, particle_box)
    ctf_kwargs_particle["astigmatism_angle"] = torch.from_numpy(particle_stack["astigmatism_angle"].values.copy())
    ctf_filters_particle = calculate_ctf_filter_stack_full_args(
        defocus_u=torch.from_numpy(particle_stack["defocus_u"].values.copy()),
        defocus_v=torch.from_numpy(particle_stack["defocus_v"].values.copy()),
        defocus_offsets=torch.from_numpy(particle_stack["relative_defocus"].values.copy()),
        pixel_size_offsets=torch.tensor([0.0]), **ctf_kwargs_particle).to(device)
    return torch.stack([build_search_image(i, image_stack_rfft, particle_filter, ctf_filters_particle, crop_size, device)
                        for i in range(num_particles)])
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
@click.option("--sweep", default="", help="Comma list of PxH batch shapes to sweep for fp16+graph, e.g. 512x512,1024x256")
@click.option("--skip-compare", is_flag=True)
def main(num_particles, num_hyps, n_psis, pixel_batch, hyp_batch, repeats, svd_cache, rectangles, sweep, skip_compare):
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
             ("pure torch (irfft + compiled reduce)", dict(precision="fp32", use_fused_kernel=False)),
             ("fp32, cuFFTDx kernel (retuned EPT)", dict(precision="fp32", _cufftdx=True)),
             ("fp32 + cuda graph",            dict(precision="fp32", use_cuda_graph=True)),
             ("fp16",                         dict(precision="fp16", use_cuda_graph=False)),
             ("fp16 + cuda graph",            dict(precision="fp16", use_cuda_graph=True)),
             ("fp16 + cuda graph (2 streams)", dict(precision="fp16", use_cuda_graph=2))]
    def timed(fn, reps):
        fn(); torch.cuda.synchronize(); ts=[]
        for _ in range(reps):
            t0=time.perf_counter(); out=fn(); torch.cuda.synchronize(); ts.append(time.perf_counter()-t0)
        return statistics.median(ts), out
    from panther_em.inference.search.utils import build_layout_weights
    from panther_em.inference.search.tiling import FeatureTiling
    tiling=FeatureTiling.from_extents(rects, device=dev)
    tw,_=timed(lambda: build_layout_weights(recon, tiling), 3)
    print(f"per-call fixed cost: build_layout_weights (U*S for all {int(hyp_all.numel())} hypotheses) = {tw*1e3:.1f} ms")
    n_img_px=n_px//imgs.shape[0]
    small_px=torch.arange(min(pixel_batch, n_img_px), device=dev)   # one pixel batch per image
    n_small=int(small_px.numel())*imgs.shape[0]
    for n_psi in [int(v) for v in n_psis.split(",")]:
        print(f"\n===== n_psi={n_psi} pixel_batch={pixel_batch} hyp_batch={hyp_batch} =====")
        from panther_em.inference.search import fused_kernel_loader as _fkl
        _lean_supported=_fkl.lean_supported
        def run(pb=pixel_batch, hb=hyp_batch, px=None, _cufftdx=False, **kw):
            # _cufftdx: force the cuFFTDx block-FFT kernel path (pre-lean fused kernel, retuned EPT)
            _fkl.lean_supported=(lambda *a, **k: False) if _cufftdx else _lean_supported
            try:
                return compressed_search(imgs, recon, rects, pixel_batch=pb, hyp_batch=hb, n_psi=n_psi,
                                         feature_chunk=32, hypothesis_indexes=hyps, pixel_index=px, store=store,
                                         show_progress=False, **kw)
            finally:
                _fkl.lean_supported=_lean_supported
        run(precision="fp32")  # featurize the persistent store once (excluded from timing)
        torch.cuda.synchronize()
        ref=None; corr=n_px*int(hyps.numel())*n_psi; corr_small=n_small*int(hyps.numel())*n_psi
        def report(label, kw, pb=pixel_batch, hb=hyp_batch):
            """End-to-end time on all pixels, and the marginal (steady-state) rate from a
            second run on one pixel batch per image: (T_all - T_small) / (corr_all - corr_small)."""
            t_all,maps=timed(lambda: run(pb=pb, hb=hb, **kw), repeats)
            t_small,_=timed(lambda: run(pb=pb, hb=hb, px=small_px, **kw), repeats)
            marg=(corr-corr_small)/max(t_all-t_small,1e-9)/1e9
            return maps, f"{label:34s} total {t_all*1e3:8.1f} ms ({corr/t_all/1e9:6.1f} Gcorr/s e2e) | fixed ~{t_small*1e3:6.1f} ms | steady-state {marg:7.1f} Gcorr/s"
        if sweep:
            print("--- batch-shape sweep, fp16 + cuda graph ---")
            for shape in sweep.split(","):
                pb,hb=(int(v) for v in shape.lower().split("x"))
                for streams,label in ((1,"1-stream"),(2,"2-stream")):
                    try:
                        _,line=report(f"  {pb:5d}x{hb:<5d} {label}", dict(precision="fp16", use_cuda_graph=streams), pb, hb)
                        print(line, flush=True)
                    except Exception as e:
                        print(f"  {pb:5d}x{hb:<5d} {label}: FAILED {str(e)[:100]}", flush=True)
                gc.collect(); torch.cuda.empty_cache()
            if skip_compare: continue
        for name,kw in configs:
            maps,line=report(name, kw)
            if ref is None:
                ref=maps
                line+=f"\n{'':34s} |mip| max {ref['mip'].abs().max().item():.3g}, zscore max {ref['zscore'].max().item():.2f}"
            else:
                mip=((maps["mip"]-ref["mip"]).abs().max()/ref["mip"].abs().max()).item()
                z=(maps["zscore"]-ref["zscore"]).abs().max().item()
                zrel=z/ref["zscore"].abs().max().item()
                bi=(maps["best_index"]==ref["best_index"]).float().mean().item(); bp=(maps["best_psi"]==ref["best_psi"]).float().mean().item()
                # do mismatched pixels have (near-)tied alternatives? look at their mip difference
                mism=maps["best_index"]!=ref["best_index"]
                worst=(maps["mip"][mism]-ref["mip"][mism]).abs().max().item() if mism.any() else 0.0
                line+=f"\n{'':34s} vs baseline: mip rel {mip:.1e}, zscore abs {z:.2e} (rel {zrel:.1e}), best_index agree {bi*100:.3f}%, best_psi agree {bp*100:.3f}%, max |dmip| at mismatches {worst:.2e}"
            print(line, flush=True)
if __name__=="__main__":
    main()
