#!/bin/bash
cd /home/mgiammar/git_repositories/Panther-EM
L=scratch/svd_search_opt/validate_real3.log
: > $L
# synthetic regression first (test env)
timeout 1800 /home/mgiammar/miniconda3/envs/panther-em-cuda-test/bin/python scratch/svd_search_opt/test_integration_synthetic.py 2>&1 | grep -vE 'Warn|warn|Deprecat|AnnAssign|init_node|^\s*$' | grep -E '===|2nd replay|2-stream|timing' >> $L
echo "synthetic_done" >> $L
export TORCH_EXTENSIONS_DIR=$PWD/scratch/svd_search_opt/torch_ext_dev
PY=/home/mgiammar/miniconda3/envs/panther-em-dev/bin/python
timeout 3000 $PY scratch/svd_search_opt/validate_real.py --num-particles 4 --num-hyps 2048 --n-psis 256 --pixel-batch 512 --hyp-batch 512 --repeats 2 >> $L 2>&1
echo "exit256=$?" >> $L
timeout 3000 $PY scratch/svd_search_opt/validate_real.py --num-particles 4 --num-hyps 2048 --n-psis 256 --pixel-batch 1024 --hyp-batch 256 --repeats 2 >> $L 2>&1
echo "exit256b=$?" >> $L
timeout 3000 $PY scratch/svd_search_opt/validate_real.py --num-particles 4 --num-hyps 2048 --n-psis 128 --pixel-batch 512 --hyp-batch 512 --repeats 2 >> $L 2>&1
echo "exit128=$?" >> $L
