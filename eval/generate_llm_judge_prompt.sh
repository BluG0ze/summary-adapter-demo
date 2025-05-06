#!/bin/bash

BASE_MODEL=Qwen/Qwen2.5-3B-Instruct
ADAPTER=chenguang-wang/Qwen2.5-3B-Instruct-summary-sft-adapter
#ADAPTER=chenguang-wang/Qwen2.5-3B-Instruct-summary-dpo-adapter
WORK_DIR=$(dirname $(realpath $0))

accelerate launch -m lm_eval --model hf \
    --model_args pretrained=$BASE_MODEL,peft=$ADAPTER \
    --include_path $WORK_DIR/tasks \
    --tasks resp_gen \
    --apply_chat_template \
    --fewshot_as_multiturn \
    --batch_size 16 \
    --seed 42
