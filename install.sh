#!/bin/bash
#SBATCH -J test
#SBATCH -p wzidnormal
#SBATCH -N 1
#SBATCH -n 8
#SBATCH --gres=dcu:1
#SBATCH -t 1-0

module unload compiler/dtk/21.10
module load compiler/dtk/24.04
source activate py310_dtk2404_code

#python3 setup.py install
python setup.py bdist_wheel