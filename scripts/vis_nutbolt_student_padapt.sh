#!/bin/bash
GPUS=$1
SEED=$2
CACHE=$3

array=( $@ )
len=${#array[@]}
EXTRA_ARGS=${array[@]:3:$len}
EXTRA_ARGS_SLUG=${EXTRA_ARGS// /_}

CUDA_VISIBLE_DEVICES=${GPUS} \
python train.py task=XHandHoraNutBolt headless=False seed=${SEED} \
task.env.numEnvs=10 test=True \
task.env.object.type=screw_trinut \
train.algo=ProprioAdapt \
train.ppo.proprio_adapt=True \
wandb_activate=False \
task.env.reset_dist_threshold=0.1 \
train.load_path=last.pth \
${EXTRA_ARGS}