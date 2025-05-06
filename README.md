# Summary Adapter Demo
## Model
Base model: [Qwen/Qwen2.5-3B-Instruct](https://huggingface.co/Qwen/Qwen2.5-3B-Instruct)

SFT Adapter: [chenguang-wang/Qwen2.5-3B-Instruct-summary-sft-adapter](https://huggingface.co/chenguang-wang/Qwen2.5-3B-Instruct-summary-sft-adapter)

DPO Adapter: [chenguang-wang/Qwen2.5-3B-Instruct-summary-dpo-adapter](https://huggingface.co/chenguang-wang/Qwen2.5-3B-Instruct-summary-dpo-adapter)

### LoRA Setting
Adapter is using the following LoRA settings:
```yaml
lora:
  r: 16
  target_modules:
    - "q_proj"
    - "k_proj" 
    - "v_proj"
  lora_alpha: 32
  use_rslora: True
  lora_dropout: 0.1
  bias: "none"
  task_type: "CAUSAL_LM"
```

| Trainable Params | All Params     | Trainable Percent |
|------------------|----------------|--------------------|
| 5,013,504        | 3,090,952,192  | 0.1622%            |


## Requirements
This projects is based on [TRL](https://huggingface.co/docs/trl/en/index), [PEFT](https://huggingface.co/docs/peft/index) and [Accelerate](https://huggingface.co/docs/accelerate/index). To speed up the training, multi-gpu env is recommanded.


### Install the dependencies
```bash
# to prevent errors, install pytorch separately first
pip install torch==2.6.0
# install other dependencies
pip install -r requirements.txt
# install flash-attn in this way will reduce tons of lib errors
pip install flash-attn --no-build-isolation
```
### Setting the necessary ENVs
```bash
export SUM_ADPT_ROOT=$(pwd)
export PYTHONPATH="${PYTHONPATH}:$(SUM_ADPT_ROOT)"
```

## SFT Training
### 1. Prepare SFT datasets
The format of original XLSum datasets is not suitable for TRL SFT, transforming format is necessary.
```bash
cd $SUM_ADPT_ROOT/datasets
python download_datasets.py
python trans_format.py
```

### 2. Modify the config
You can use my training setting or modify to your prefer settings
```bash
cd $SUM_ADPT_ROOT/train/sft-lora
vim sft_lora_config.yaml
```

### 3. Start training by accelerate
```bash
cd $SUM_ADPT_ROOT/train/sft-lora
accelerate launch train_sft_trl.py
```

## DPO Training
### 1. Modify the config
*About the trainset: Making preference datasets(DPO trainingset) is complex, the trainingset I used is uploaded to [chenguang-wang/xlsum_pref_5k](https://huggingface.co/datasets/chenguang-wang/xlsum_pref_5k). The DPO training script will use it by default.*
```bash
cd $SUM_ADPT_ROOT/train/dpo-lora
vim dpo_lora_config.yaml
```
### 2. Start training by accelerate
```bash
cd $SUM_ADPT_ROOT/train/dpo-lora
accelerate launch train_dpo_trl.py
```

## Eval the models
Testsets are using [csebuetnlp/xlsum](https://huggingface.co/datasets/csebuetnlp/xlsum) CHT, JA, KO and EN test subsets.

### Eval with Rouge2 and Avg length
```bash
cd $SUM_ADPT_ROOT/eval
# Make sure all the setting are correctly
vim eval_lm_harness.sh
# start eval
bash eval_lm_harness.sh
```

#### Results
| Model                        | CHT(Rouge2)↑ | CHT(avg_len)↓ | EN(Rouge2)↑ | EN(avg_len)↓ | JA(Rouge2)↑ | JA(avg_len)↓ | KO(Rouge2)↑ | KO(avg_len)↓ |
|-----------------------------|------------------|----------------|------------------|---------------|------------------|---------------|------------------|---------------|
| QWen2                       | 0.0632           | 189.5081       | 0.0295           | 633.1491      | 0.0990           | 227.6738      | 0.0198           | 287.5564      |
| QWen2 + SFT(LoRA)           | 0.1648           | 47.1420        | 0.1204           | 112.1280      | 0.1915           | 68.2846       | 0.0365           | 52.9145       |
| QWen2 + SFT(LoRA) + DPO(LoRA)| 0.1667           | 44.2096        | 0.1211           | 110.4704      | 0.1860           | 67.3071       | 0.0340           | 52.0964       |


### Eval with win rate by LLM-as-a-Judge
Win-rate is using the completion by the adatper comparing with the golden anwer(summary in the original datasets)

Evaling win-rate need 4 main steps:
1. Setting your llm-judge API/KEY, and setting the candidate model
```bash
vim $SUM_ADPT_ROOT/utils/llm_judge_utils.py
vim $SUM_ADPT_ROOT/eval/generate_llm_judge_prompt.sh
```
2. Specifying the testset

*Due to the limited time, evaling with all languages are not supported now. Need to eval one by one.*
```bash
vim $SUM_ADPT_ROOT/eval/tasks/resp_gen/resp_gen.yaml
```

3. Generating completion from candidate model and transfer into llm-judge prompts

```bash
cd $SUM_ADPT_ROOT/eval
bash generate_llm_judge_prompt.sh
```
4. Calling llm-judge API get the win-rate results
```bash
cd $SUM_ADPT_ROOT/eval
python check_win_rate.py
```

#### Results
*The following results are judged by `Mixtral-8x22B-Instruct-v0.1`*
| Model                        | CHT(winrate) | EN(winrate) | JA(winrate) | KO(winrate) |
|-----------------------------|-------------------|-------------------|-------------------|-------------------|
| QWen2                       | 0.00 (16/4670)     | 0.00 (54/11535)    | 0.03 (29/889)      | 0.01 (6/550)       |
| QWen2 + SFT(LoRA)           | 0.42 (1961/4670)   | 0.53 (6083/11535)  | 0.59 (524/889)     | 0.44 (240/550)     |
| QWen2 + SFT(LoRA) + DPO(LoRA)| 0.36 (1660/4670)   | 0.49 (5700/11535)  | 0.56 (499/889)     | 0.41 (223/550)     |


## Inference
One simple example and one naive summerization assistant is provided. Play with them and have fun ;)
```bash
cd $SUM_ADPT_ROOT/inference
python example.py
python summerization_assistant.py
```
