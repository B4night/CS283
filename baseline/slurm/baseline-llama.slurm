#!/bin/bash --login
#SBATCH --nodes=1
#SBATCH --gres=gpu:v100:2
#SBATCH --output=%x-%j-baseline_llama.out
#SBATCH --error=%x-%j-baseline_llama.err
#SBATCH --mem=350G
#SBATCH --time=24:00:00

conda activate open_r1
cd ..
python open_source_model.py --model=llama-3.1-8b
