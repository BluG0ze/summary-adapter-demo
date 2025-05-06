#!/bin/bash
# This script is used to evaluate the summarize adapter capability by using lm-eval-harness.

BASE_MODEL=Qwen/Qwen2.5-3B-Instruct
ADAPTER=chenguang-wang/Qwen2.5-3B-Instruct-summary-sft-adapter
WORK_DIR=$(dirname $(realpath $0))

accelerate launch -m lm_eval --model hf \
    --model_args pretrained=$BASE_MODEL,peft=$ADAPTER \
    --include_path $WORK_DIR/tasks \
    --tasks en_xlsum,cht_xlsum,ja_xlsum,kr_xlsum \
    --apply_chat_template \
    --fewshot_as_multiturn \
    --batch_size 32 \
    --log_samples \
    --output_path ./eval_results 
