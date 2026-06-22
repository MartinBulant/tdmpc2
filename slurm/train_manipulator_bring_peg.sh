#!/bin/bash

#SBATCH --job-name=tdmpc2_manipulator_bring_peg
#SBATCH --output=logs/log_files/%j_%x.out
#SBATCH --error=logs/log_files/%j_%x.err
#SBATCH --partition=gpulong
#SBATCH --gres=gpu:1
#SBATCH --mem=20G
#SBATCH --array=0-2
#SBATCH --mail-user=bulanma2@fel.cvut.cz
#SBATCH --mail-type=END,FAIL

SEEDS=(0 1 2 3)

echo "Task ID: $SLURM_ARRAY_TASK_ID"
echo "Seed: $SLURM_ARRAY_TASK_ID"
echo "Seed offset: ${SEED_OFFSETS[$SLURM_ARRAY_TASK_ID]}"

ml virtualenv/20.26.2
source .venv_test2/bin/activate
cd ./tdmpc2/
python train.py \
    task=manipulator-bring-peg \
    wandb_project=dm_control_manipulator_bring_peg \
    steps=5000000 \
    seed=${SEEDS[$SLURM_ARRAY_TASK_ID]} \
    num_envs=1 \
    steps_per_update=1 \
    exp_name=one_env
    

