#!/bin/bash
# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under Creative Commons-Non Commercial 4.0 found in the
# LICENSE file in the root directory of this source tree.

export MASTER_PORT=88888

echo $CUDA_VISIBLE_DEVICES
nvidia-smi
# End visible GPUs.

# setup conda environment
source setup.sh

which python

python src/nli/training.py \
  --model_class_name "roberta-large" \
  -n 1 \
  -g 8 \
  -nr 0 \
  --fp16 \
  --fp16_opt_level O2 \
  --max_length 156 \
  --gradient_accumulation_steps 1 \
  --per_gpu_train_batch_size 16 \
  --per_gpu_eval_batch_size 32 \
  --save_prediction \
  --train_data \
snli_train:none,mnli_train:none,fever_train:none,anli_r1_train:none,anli_r2_train:none,anli_r3_train:none \
  --train_weights \
1,1,1,10,20,10 \
  --eval_data \
snli_dev:none,mnli_m_dev:none,mnli_mm_dev:none,anli_r1_dev:none,anli_r2_dev:none,anli_r3_dev:none \
  --eval_frequency 2000 \
  --experiment_name "roberta-large|snli+mnli+fnli+r1*10+r2*20+r3*10|nli"