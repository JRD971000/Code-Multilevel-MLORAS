#!/bin/bash
#SBATCH --account=def-scottmac
#SBATCH --gpus-per-node=1
#SBATCH --mem=100000               # memory per node
#SBATCH --time=0-5:00

source ../mlvl-mloras/bin/activate
python train.py > output.txt                      # you can use 'nvidia-smi' for a test
