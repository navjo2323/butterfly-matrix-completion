#!/bin/bash
#SBATCH -A m2957
#SBATCH -C cpu
#SBATCH -q regular
#SBATCH -t 23:00:00
#SBATCH -N 1
#SBATCH --ntasks-per-node=1
#SBATCH --mail-user=liuyangzhuan@lbl.gov
#SBATCH --mail-type=BEGIN
#SBATCH -e ./tmp.er
module load python
python -u compare_test_train.py | tee a.out_L10_c4_rTT50_6rnlogn


