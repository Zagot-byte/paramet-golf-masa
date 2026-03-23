#!/bin/bash
cd ~/parameter-golf

killall hypridle 2>/dev/null
echo "hypridle killed"

TORCHDYNAMO_DISABLE=1 \
RUN_ID=masa_overnight \
TRAIN_BATCH_TOKENS=131072 \
TRAIN_SEQ_LEN=512 \
NUM_KV_HEADS=8 \
MASA_NUM_BASES=6 \
ITERATIONS=20000 \
VAL_LOSS_EVERY=2000 \
TRAIN_LOG_EVERY=500 \
VAL_BATCH_SIZE=32768 \
MAX_WALLCLOCK_SECONDS=0 \
torchrun --standalone --nproc_per_node=1 train_gpt_masa.py
