#!/bin/bash
GPUS=$1
SEED=$2
CACHE=$3

array=( $@ )
len=${#array[@]}
EXTRA_ARGS=${array[@]:3:$len}
EXTRA_ARGS_SLUG=${EXTRA_ARGS// /_}

python train.py task=XHandHoraScrewDriver headless=False seed=${SEED} \
sim_device=cuda:${GPUS} rl_device=cuda:${GPUS} graphics_device_id=7 \
task.env.numEnvs=6 test=True \
train.algo=PPO \
task.env.reset_dist_threshold=0.12 \
wandb_activate=False \
checkpoint=last.pth \
${EXTRA_ARGS}