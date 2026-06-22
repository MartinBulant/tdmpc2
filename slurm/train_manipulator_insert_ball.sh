#!/bin/bash

#SBATCH --job-name=tdmpc2_manipulator_insert_ball
#SBATCH --output=logs/log_files/%j_%x.out
#SBATCH --error=logs/log_files/%j_%x.err
#SBATCH --partition=gpulong
#SBATCH --gres=gpu:1
#SBATCH --mem=20G
#SBATCH --mail-user=bulanma2@fel.cvut.cz
#SBATCH --mail-type=END,FAIL

ml virtualenv/20.26.2
source .venv_test2/bin/activate
cd ./tdmpc2/
python train.py \
    task=manipulator-insert-ball \
    wandb_project=dm_control_manipulator_insert_ball \
    steps=5000000 
    

