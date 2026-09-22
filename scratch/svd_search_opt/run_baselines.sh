#!/bin/bash
cd /home/mgiammar/git_repositories/Panther-EM
until grep -q 'exit128=' scratch/svd_search_opt/validate_real3.log; do sleep 2; done
export TORCH_EXTENSIONS_DIR=$PWD/scratch/svd_search_opt/torch_ext_dev
L=scratch/svd_search_opt/validate_real4.log
: > $L
timeout 3000 /home/mgiammar/miniconda3/envs/panther-em-dev/bin/python scratch/svd_search_opt/validate_real.py --num-particles 4 --num-hyps 2048 --n-psis 256 --pixel-batch 512 --hyp-batch 512 --repeats 2 >> $L 2>&1
echo "exit_base=$?" >> $L
