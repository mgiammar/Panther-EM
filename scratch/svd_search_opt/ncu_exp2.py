import sys, torch
sys.path.insert(0,"scratch/svd_search_opt/kernels"); import exp_loader
m2=exp_loader.load(name="panther_exp2_irfft_stats", src="irfft_stats_exp2.cu")
dev=torch.device("cuda:0"); torch.cuda.set_device(dev)
P,N,K=512,2048,64
c16=torch.randn(K,P,2*N,device=dev).half()
for _ in range(3): m2.transposed_f16(c16,256,16,32)
torch.cuda.synchronize(); torch.cuda.profiler.start()
for _ in range(2): m2.transposed_f16(c16,256,16,32)
torch.cuda.synchronize(); torch.cuda.profiler.stop()
