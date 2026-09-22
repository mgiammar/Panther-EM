import sys; sys.path.insert(0,"scratch/svd_search_opt/kernels"); import exp_loader
m=exp_loader.load(name="panther_exp2_irfft_stats", src="irfft_stats_exp2.cu"); print("exp2 built:", m.get_supported_configs()[:3], "...")
