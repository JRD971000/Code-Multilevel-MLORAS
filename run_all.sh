#!/bin/bash
#SBATCH --account=def-scottmac
#SBATCH --gpus-per-node=1
#SBATCH --mem=64G               # memory per node
#SBATCH --time=0-05:00

source ../mlvl-mloras/bin/activate
python train.py > output_one.txt                         # you can use 'nvidia-smi' for a test
