import hydra
import logging
from datasets import load_dataset
from trl import SFTConfig, SFTTrainer, setup_chat_format
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    set_seed,
)
from peft import LoraConfig, TaskType, get_peft_model
from omegaconf import DictConfig, OmegaConf
from hydra.utils import get_original_cwd

logging.basicConfig(
    format="%(asctime)s - %(levelname)s - %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
    level=logging.INFO,
)


@hydra.main(config_path="./", config_name="sft_lora_config")
def main(cfg: DictConfig):
    """
    Main function to train the model using SFT.
    Args:
        cfg (DictConfig): Configuration object containing training parameters.
    """
    set_seed(cfg.seed)

    # hydra will change the working directory
    # so we need to change the relative path in the config to the absolute path
    current_path = get_original_cwd() + "/"
    train_files = [current_path + dataset for dataset in cfg.data.train]
    val_files = [current_path + dataset for dataset in cfg.data.val]
    lora_cfg = OmegaConf.to_container(cfg.lora, resolve=True)
    training_cfg = OmegaConf.to_container(cfg.train, resolve=True)
    training_cfg["output_dir"] = current_path + training_cfg["output_dir"]

    # Load the model
    model = AutoModelForCausalLM.from_pretrained(
        cfg.model.base_model,
        #torch_dtype=torch.float16,
        #device_map="auto", #{'':device_string}
        trust_remote_code=True,
        attn_implementation="flash_attention_2",
        use_cache=False,
    )

    if model.supports_gradient_checkpointing:
        model.gradient_checkpointing_enable()
        model.enable_input_require_grads()

    lora_config = LoraConfig(**lora_cfg)
    model = get_peft_model(model, lora_config)

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

    # Setup the training configuration
    training_cfg["eos_token"] = tokenizer.eos_token
    training_args = SFTConfig(**training_cfg)

    # Setup the dataset
    train_dataset = load_dataset("json", data_files=train_files)["train"]
    val_dataset = load_dataset("json", data_files=val_files)["train"]

    # Set up the SFT trainer
    trainer = SFTTrainer(
        model,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        args=training_args,
    )
    trainer.train()


if __name__ == "__main__":
    main()