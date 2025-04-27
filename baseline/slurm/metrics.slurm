#!/bin/bash --login
#SBATCH --nodes=1
#SBATCH --gres=gpu:v100:1
#SBATCH --output=%x-%j-metrics.out
#SBATCH --error=%x-%j-metrics.err
#SBATCH --mem=350G
#SBATCH --time=24:00:00

conda activate open_r1
cd ..
python metrices.py