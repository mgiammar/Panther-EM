#!/bin/bash
cd /home/mgiammar/git_repositories/Panther-EM
export TORCH_EXTENSIONS_DIR=$PWD/scratch/svd_search_opt/torch_ext_dev
PY=/home/mgiammar/miniconda3/envs/panther-em-dev/bin/python
L=scratch/svd_search_opt/validate_real.log
: > $L
timeout 6000 $PY scratch/svd_search_opt/validate_real.py --num-particles 1 --num-hyps 2048 --n-psis 256 --pixel-batch 512 --hyp-batch 512 --repeats 1 --sweep 512x512,1024x256,256x1024,512x1024,512x2048 >> $L 2>&1
echo "exit256=$?" >> $L
timeout 3000 $PY scratch/svd_search_opt/validate_real.py --num-particles 1 --num-hyps 2048 --n-psis 128 --pixel-batch 512 --hyp-batch 512 --repeats 1 >> $L 2>&1
echo "exit128=$?" >> $L
