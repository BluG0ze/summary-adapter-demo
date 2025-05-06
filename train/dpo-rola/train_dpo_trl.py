import os
import logging
import hydra
from omegaconf import DictConfig, OmegaConf
from datasets import load_dataset
from trl import DPOConfig, DPOTrainer
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    set_seed,
)
from peft import PeftModel
from hydra.utils import get_original_cwd

logging.basicConfig(
    format="%(asctime)s - %(levelname)s - %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
    level=logging.INFO,
)


@hydra.main(config_path="./", config_name="dpo_lora_config")
def main(cfg: DictConfig):
    """
    Main function to train the model using DPO.
    Args:
        cfg (DictConfig): Configuration object containing training parameters.
    """
    set_seed(cfg.seed)

    # hydra will change the working directory
    # so we need to change the relative path in the config to the absolute path
    current_path = get_original_cwd() + "/"
    training_cfg = OmegaConf.to_container(cfg.train, resolve=True)
    training_cfg["output_dir"] = current_path + training_cfg["output_dir"]

    # Load the model
    model = AutoModelForCausalLM.from_pretrained(
        cfg.model.base_model,
        trust_remote_code=True,
        attn_implementation="flash_attention_2",
        use_cache=False,
    )
    model = PeftModel.from_pretrained(
        model,
        cfg.model.sft_lora,
        is_trainable=True,
    )
    if model.supports_gradient_checkpointing:
        model.gradient_checkpointing_enable()
        model.enable_input_require_grads()

    # Get the number of trainable parameters
    trainable_params, all_param = model.get_nb_trainable_parameters()
    logging.info(
        f"trainable params: {trainable_params:,d} || " f"all params: {all_param:,d} || " f"trainable%: {100 * trainable_params / all_param:.4f}"
    )

    # Load the tokenizer
    tokenizer = AutoTokenizer.from_pretrained(
        cfg.model.base_model,
        trust_remote_code=True,
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"

    # Load the training arguments
    training_args = DPOConfig(**training_cfg)

    # Load the dataset
    
    dataset = load_dataset("chenguang-wang/xlsum_pref_5k", split="train")
    split_dataset = dataset.train_test_split(test_size=0.1, seed=42)

    # Set up the DPO trainer
    trainer = DPOTrainer(
        model,
        train_dataset=split_dataset["train"],
        eval_dataset=split_dataset["test"],
        args=training_args,
        processing_class=tokenizer,
    )
    trainer.train()


if __name__ == "__main__":
    main()
