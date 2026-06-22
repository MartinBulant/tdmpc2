#!/bin/bash

srun \
  --job-name=interactive_tdmpc2 \
  --partition=gpufast \
  --gres=gpu:1 \
  --cpus-per-task=25 \
  --mem 45G \
  --pty bash -ic "
  exec bash"