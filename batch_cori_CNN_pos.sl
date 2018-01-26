#!/bin/bash -l
#SBATCH -p regular
#SBATCH -t 36:00:00
#SBATCH -C haswell
#SBATCH -N 1
#SBATCH -L SCRATCH

cd /global/u2/m/maweber/EXODeep/projects/PosFromAPD/
/global/u2/m/maweber/EXODeep/projects/PosFromAPD/submitTrainAPDCNNPos.sh

