#!/bin/bash --login
#SBATCH --nodes=1
#SBATCH --output=%x-%j-baseline.out
#SBATCH --error=%x-%j-baseline.err
#SBATCH --mem=350G
#SBATCH --time=24:00:00

conda activate open_r1
cd ..
python pure_gpt_4o.py