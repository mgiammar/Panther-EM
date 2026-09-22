import sys, time, torch
sys.path.insert(0, "experiments/profiling_svd_2dtm_search")
from profile_svd_search import build_reconstructor
t0=time.perf_counter(); r=build_reconstructor(torch.device("cuda:0"), "scratch/svd_search_opt/svd_60S_k160_e128.h5")
print(f"done in {time.perf_counter()-t0:.0f}s; image_shape={r.image_shape} num_orientations={r.result.num_orientations} num_ff={r.result.num_fourier_filters}")
