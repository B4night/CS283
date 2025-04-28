#!/bin/bash --login
#SBATCH --nodes=1
#SBATCH --gres=gpu:v100:2
#SBATCH --output=%x-%j-method_qwen.out
#SBATCH --error=%x-%j-method_qwen.err
#SBATCH --mem=350G
#SBATCH --time=24:00:00

conda activate open_r1
cd ../../src
python main_with_opensource_model.py --model=qwen2.5-7b