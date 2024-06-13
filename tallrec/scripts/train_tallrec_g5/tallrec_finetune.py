import os
os.environ['LD_LIBRARY_PATH'] = '/data/baokq/miniconda3/envs/alpaca_lora/lib/'
import sys
from typing import List
import argparse
import numpy as np
import torch
from datetime import datetime
import subprocess
import wandb
import transformers
from datasets import load_dataset
from transformers import EarlyStoppingCallback
from peft import (  # noqa: E402
    LoraConfig,
    get_peft_model,
    get_peft_model_state_dict,
    prepare_model_for_int8_training,
    PeftModel,
    set_peft_model_state_dict,
)
from transformers import LlamaForCausalLM, LlamaTokenizer, AddedToken, BitsAndBytesConfig  # noqa: F402
from sklearn.metrics import roc_auc_score
from importlib.metadata import version
from torch.utils.tensorboard import SummaryWriter


def run_command(command):
    """Run a system command and return the output."""
    try:
        output = subprocess.check_output(command, stderr=subprocess.STDOUT).decode("utf-8")
        return True, output
    except subprocess.CalledProcessError as e:
        return False, e.output.decode("utf-8")

def check_cuda_availability():
    print(f"PyTorch version: {torch.__version__}")
    print(f"PyTorch built with CUDA: {torch.version.cuda}")
    
    cuda_available = torch.cuda.is_available()
    print(f"CUDA is available to PyTorch: {cuda_available}")
    if cuda_available:
        print(f"Number of CUDA devices: {torch.cuda.device_count()}")
    
    success, _ = run_command(["python", "-c", "import torch; torch.tensor([1.0], device='cuda:0')"])
    if success:
        print("Successfully created a tensor on CUDA.")
    else:
        print("Failed to create a tensor on CUDA. This might be due to no CUDA devices being available.")

def check_system_cuda():
    success, nvcc_version = run_command(["nvcc", "--version"])
    if success:
        print("System nvcc version:")
        print(nvcc_version)
    else:
        print("nvcc command failed or not found. Check if CUDA is installed correctly and added to PATH.")

    success, nvidia_smi_output = run_command(["nvidia-smi"])
    if success:
        print("nvidia-smi output:")
        print(nvidia_smi_output)
    else:
        print("nvidia-smi command failed or not found. Check if NVIDIA drivers are installed correctly.")


def check_env_variables():
    cuda_visible_devices = os.getenv("CUDA_VISIBLE_DEVICES", "Not set")
    print(f"CUDA_VISIBLE_DEVICES: {cuda_visible_devices}")
    
    ld_library_path = os.getenv("LD_LIBRARY_PATH", "Not set")
    print(f"LD_LIBRARY_PATH: {ld_library_path}")

def list_and_print_contents(dir_path):
    print(f"Contents of {dir_path}:")
    try:
        # List all files and directories in the specified path
        entries = os.listdir(dir_path)
        if entries:
            for entry in entries:
                print(entry)
        else:
            print("The directory is empty.")
    except FileNotFoundError:
        print("The directory does not exist.")


def generate_prompt(data_point):
    # sorry about the formatting disaster gotta move fast
    if data_point["input"]:
        return f"""Below is an instruction that describes a task, paired with an input that provides further context. Write a response that appropriately completes the request.  # noqa: E501

### Instruction:
{data_point["instruction"]}

### Input:
{data_point["input"]}

### Response:
{data_point["output"]}"""
    else:
        return f"""Below is an instruction that describes a task. Write a response that appropriately completes the request.  # noqa: E501

### Instruction:
{data_point["instruction"]}

### Response:
{data_point["output"]}"""


def train(args):
    """
    Finetunes a Language Model (LM) using Low-Rank Adaptation (LoRA) for recommendation tasks.

    Args (parsed):
        model_id (str): Pretrained model identifier for the LM to be finetuned.
        train_data_path (str): Path to the training dataset.
        val_data_path (str): Path to the validation dataset.
        output_dir (str): Path to save the finetuned model.
        sample (int): Number of samples to use for training; -1 for all data.
        seed (int): Random seed for reproducibility.
        batch_size (int): Batch size for the training.
        micro_batch_size (int): Micro batch size for gradient accumulation.
        num_epochs (int): Number of epochs to train the model.
        learning_rate (float): Learning rate for the optimizer.
        cutoff_len (int): Maximum sequence length for the model.
        lora_r (int): Rank for the LoRA adaptation.
        lora_alpha (int): Alpha parameter for LoRA.
        lora_dropout (float): Dropout rate for LoRA layers.
        lora_target_modules (List[str]): Model layers to which LoRA is applied.
        train_on_inputs (bool): If False, ignores inputs in the loss calculation.
        group_by_length (bool): Groups samples by length for efficient batching.
        wandb_project (str): Weights & Biases project name for logging.
        wandb_run_name (str): Weights & Biases run name.
        wandb_watch (str): Level of model watching in Weights & Biases.
        wandb_log_model (str): Flag to log the model in Weights & Biases.
        resume_from_checkpoint (str): Path to a training checkpoint to resume from.
    """
    print(f"Training with the following configuration: {args}")
    
    print(
        f"Training Alpaca-LoRA model with params:\n"
        f"model_id: {args.model_id}\n"
        f"train_data_path: {args.train_data_path}\n"
        f"val_data_path: {args.val_data_path}\n"
        f"sample: {args.sample}\n"
        f"seed: {args.seed}\n"
        f"output_dir: {args.output_dir}\n"
        f"batch_size: {args.batch_size}\n"
        f"micro_batch_size: {args.micro_batch_size}\n"
        f"num_epochs: {args.num_epochs}\n"
        f"learning_rate: {args.learning_rate}\n"
        f"cutoff_len: {args.cutoff_len}\n"
        f"lora_r: {args.lora_r}\n"
        f"lora_alpha: {args.lora_alpha}\n"
        f"lora_dropout: {args.lora_dropout}\n"
        f"lora_target_modules: {args.lora_target_modules}\n"
        f"train_on_inputs: {args.train_on_inputs}\n"
        f"group_by_length: {args.group_by_length}\n"
        f"wandb_project: {args.wandb_project}\n"
        f"wandb_run_name: {args.wandb_run_name}\n"
        f"wandb_watch: {args.wandb_watch}\n"
        f"wandb_log_model: {args.wandb_log_model}\n"
        f"resume_from_checkpoint: {args.resume_from_checkpoint}\n"
    )

    # Use args to access the command-line arguments
    model_id = args.model_id
    train_data_path = args.train_data_path
    val_data_path = args.val_data_path
    output_dir = args.output_dir
    sample = args.sample
    seed = args.seed
    batch_size = args.batch_size
    micro_batch_size = args.micro_batch_size
    num_epochs = args.num_epochs
    learning_rate = args.learning_rate
    cutoff_len = args.cutoff_len
    lora_r = args.lora_r
    lora_alpha = args.lora_alpha
    lora_dropout = args.lora_dropout
    lora_target_modules = args.lora_target_modules
    train_on_inputs = args.train_on_inputs
    group_by_length = args.group_by_length
    wandb_project = args.wandb_project
    wandb_run_name = args.wandb_run_name
    wandb_watch = args.wandb_watch
    wandb_log_model = args.wandb_log_model
    resume_from_checkpoint = args.resume_from_checkpoint

    assert (
        model_id
    ), "Please specify a --model_id, e.g. --model_id='decapoda-research/llama-7b-hf'"
    gradient_accumulation_steps = batch_size // micro_batch_size
    print(f"gradient_accumulation_steps: {gradient_accumulation_steps}")

    device_map = "auto"
    world_size = int(os.environ.get("WORLD_SIZE", 1))
    print(f"World size: {world_size}")

    ddp = world_size != 1
    if ddp:
        local_rank = int(os.environ.get("LOCAL_RANK") or 0)
        print(f"Local Rank: {local_rank}")
        device_map = {"": int(os.environ.get("LOCAL_RANK") or 0)}
        gradient_accumulation_steps = gradient_accumulation_steps // world_size

    # Check if parameter passed or if set within environ
    use_wandb = len(wandb_project) > 0 or (
        "WANDB_PROJECT" in os.environ and len(os.environ["WANDB_PROJECT"]) > 0
    )
    # Only overwrite environ if wandb param passed
    if len(wandb_project) > 0:
        os.environ["WANDB_PROJECT"] = wandb_project
    if len(wandb_watch) > 0:
        os.environ["WANDB_WATCH"] = wandb_watch
    if len(wandb_log_model) > 0:
        os.environ["WANDB_LOG_MODEL"] = wandb_log_model

    quantization_config = BitsAndBytesConfig(llm_int8_enable_fp32_cpu_offload=True)

    model = LlamaForCausalLM.from_pretrained(
        model_id,
        load_in_8bit=False,
        torch_dtype=torch.float16,
        device_map=device_map,
        quantization_config=quantization_config
    )

    tokenizer = LlamaTokenizer.from_pretrained(model_id)

    tokenizer.pad_token_id = (
        0  # unk. we want this to be different from the eos token
    )
    tokenizer.padding_side = "left"  # Allow batched inference

    def tokenize(prompt, add_eos_token=True):
        # there's probably a way to do this with the tokenizer settings
        # but again, gotta move fast
        result = tokenizer(
            prompt,
            truncation=True,
            max_length=cutoff_len,
            padding=False,
            return_tensors=None,
        )
        if (
            result["input_ids"][-1] != tokenizer.eos_token_id
            and len(result["input_ids"]) < cutoff_len
            and add_eos_token
        ):
            result["input_ids"].append(tokenizer.eos_token_id)
            result["attention_mask"].append(1)

        result["labels"] = result["input_ids"].copy()

        return result

    def generate_and_tokenize_prompt(data_point):
        full_prompt = generate_prompt(data_point)
        tokenized_full_prompt = tokenize(full_prompt)
        if not train_on_inputs:
            user_prompt = generate_prompt({**data_point, "output": ""})

            user_prompt_len = len(tokenized_user_prompt["input_ids"])

            tokenized_full_prompt["labels"] = [
                -100
            ] * user_prompt_len + tokenized_full_prompt["labels"][
                user_prompt_len:
            ]  # could be sped up, probably
        return tokenized_full_prompt

    # model = prepare_model_for_int8_training(model)

    config = LoraConfig(
        r=lora_r,
        lora_alpha=lora_alpha,
        target_modules=lora_target_modules,
        lora_dropout=lora_dropout,
        bias="none",
        task_type="CAUSAL_LM",
    )
    model = get_peft_model(model, config)

    train_data_path = train_data_path + "/" + "train.json"

    val_data_path = val_data_path + "/" + "valid.json"
    
    if train_data_path.endswith(".json"):  # todo: support jsonl
        train_data = load_dataset("json", data_files=train_data_path)
    else:
        train_data = load_dataset(train_data_path)
    
    if val_data_path.endswith(".json"):  # todo: support jsonl
        val_data = load_dataset("json", data_files=val_data_path)
    else:
        val_data = load_dataset(val_data_path)

    if resume_from_checkpoint:
        # Check the available weights and load them
        checkpoint_name = os.path.join(
            resume_from_checkpoint, "pytorch_model.bin"
        )  # Full checkpoint
        if not os.path.exists(checkpoint_name):
            checkpoint_name = os.path.join(
                resume_from_checkpoint, "adapter_model.bin"
            )  # only LoRA model - LoRA config above has to fit
            resume_from_checkpoint = (
                False  # So the trainer won't try loading its state
            )
        # The two files above have a different name depending on how they were saved, but are actually the same.
        if os.path.exists(checkpoint_name):
            print(f"Restarting from {checkpoint_name}")
            adapters_weights = torch.load(checkpoint_name)
            model = set_peft_model_state_dict(model, adapters_weights)
        else:
            print(f"Checkpoint {checkpoint_name} not found")

    model.print_trainable_parameters()  # Be more transparent about the % of trainable params.

    train_data["train"] = train_data["train"].shuffle(seed=seed).select(range(sample)) if sample > -1 else train_data["train"].shuffle(seed=seed)
    train_data = (train_data["train"].map(generate_and_tokenize_prompt))
    val_data = (val_data["train"].map(generate_and_tokenize_prompt))

    if not ddp and torch.cuda.device_count() > 1:
        print("Parallelizing model")
        model.is_parallelizable = True
        model.model_parallel = True

    def compute_metrics(eval_preds):
        pre, labels = eval_preds
        auc = roc_auc_score(pre[1], pre[0])
        return {'auc': auc}
    
    def preprocess_logits_for_metrics(logits, labels):
        """
        Original Trainer may have a memory leak. 
        This is a workaround to avoid storing too many tensors that are not needed.
        """
        labels_index = torch.argwhere(torch.bitwise_or(labels == 8241, labels == 3782))
        gold = torch.where(labels[labels_index[:, 0], labels_index[:, 1]] == 3782, 0, 1)
        labels_index[: , 1] = labels_index[: , 1] - 1
        logits = logits.softmax(dim=-1)
        logits = torch.softmax(logits[labels_index[:, 0], labels_index[:, 1]][:,[3782, 8241]], dim = -1)
        return logits[:, 1][2::3], gold[2::3]
    
    if sample > -1:
        if sample <= 128 :
            eval_step = 10
        else:
            eval_step = sample / 128 * 5
    else:
        eval_step = 100

    trainer = transformers.Trainer(
        model=model,
        train_dataset=train_data,
        eval_dataset=val_data,
        args=transformers.TrainingArguments(
            per_device_train_batch_size=micro_batch_size,
            gradient_accumulation_steps=gradient_accumulation_steps,
            warmup_steps=20,
            num_train_epochs=num_epochs,
            learning_rate=learning_rate,
            fp16=True, 
            logging_steps=8,
            optim="adamw_torch",
            evaluation_strategy="steps",
            save_strategy="steps",
            eval_steps=eval_step,
            save_steps=eval_step,
            output_dir=output_dir,
            save_total_limit=1,
            load_best_model_at_end=True,
            metric_for_best_model="eval_auc",
            ddp_find_unused_parameters=False if ddp else None,
            group_by_length=group_by_length,
            report_to=["wandb", "tensorboard"] if use_wandb else None,
            eval_accumulation_steps=10,
        ),
        data_collator=transformers.DataCollatorForSeq2Seq(
            tokenizer, pad_to_multiple_of=8, return_tensors="pt", padding=True
        ),
        compute_metrics=compute_metrics,
        preprocess_logits_for_metrics=preprocess_logits_for_metrics,
        callbacks=[EarlyStoppingCallback(early_stopping_patience=10)]
    )
    model.config.use_cache = False

    old_state_dict = model.state_dict
    model.state_dict = (
        lambda self, *_, **__: get_peft_model_state_dict(
            self, old_state_dict()
        )
    ).__get__(model, type(model))

    if torch.__version__ >= "2" and sys.platform != "win32":
        model = torch.compile(model)

    trainer.train(resume_from_checkpoint=resume_from_checkpoint)
    print(f"Saving adapted model to output_dir: {output_dir}")

    model.save_pretrained(output_dir)

    # Reload tokenizer to save it
    tokenizer = LlamaTokenizer.from_pretrained(model_id, trust_remote_code=True)

    # Set the padding side and pad token ID
    tokenizer.padding_side = "left"
    tokenizer.pad_token_id = 0

    # Define special tokens using AddedToken class
    special_tokens_dict = {
        "unk_token": AddedToken("<unk>", lstrip=False, rstrip=False),
        "bos_token": AddedToken("<s>", lstrip=False, rstrip=False),
        "eos_token": AddedToken("</s>", lstrip=False, rstrip=False)
    }

    # Add special tokens to the tokenizer
    tokenizer.add_special_tokens({
        "unk_token": special_tokens_dict["unk_token"],
        "bos_token": special_tokens_dict["bos_token"],
        "eos_token": special_tokens_dict["eos_token"]
    })

    # Ensure the special tokens are set correctly
    tokenizer.unk_token = special_tokens_dict["unk_token"]
    tokenizer.bos_token = special_tokens_dict["bos_token"]
    tokenizer.eos_token = special_tokens_dict["eos_token"]

    # Update model_max_length
    tokenizer.model_max_length = 2048
    tokenizer.save_pretrained(output_dir)

    new_model = output_dir

    del model
    del trainer

    # Reload model in FP16 and merge it with the LORA weights
    base_model = LlamaForCausalLM.from_pretrained(
        model_id,
        low_cpu_mem_usage=True,
        return_dict=True,
        torch_dtype=torch.float16,
        device_map=device_map,
    )

    merged_model = PeftModel.from_pretrained(base_model, new_model)
    merged_model = merged_model.merge_and_unload()

    merged_model_dir = output_dir + "/merged_model"

    merged_model.save_pretrained(merged_model_dir, safe_serialization=True, max_shard_size="2GB")

    torch.cuda.empty_cache()


if __name__ == "__main__":
    # Set up argument parsing
    parser = argparse.ArgumentParser(description="Finetune a LLM with LoRA.")
    
    parser.add_argument("--model_id", type=str, required=True, help="Pretrained model identifier for the LM to be finetuned.")
    parser.add_argument("--train_data_path", type=str, default=os.environ["SM_CHANNEL_TRAIN"], help="Path to the training dataset.")
    parser.add_argument("--val_data_path", type=str, default=os.environ.get("SM_CHANNEL_VALID"), help="Path to the validation dataset.")
    parser.add_argument("--output_dir", type=str, default=os.environ.get("SM_MODEL_DIR"), help="Path to save the finetuned model.")
    parser.add_argument("--sample", type=int, default=-1, help="Number of samples to use for training; -1 for all data.")
    parser.add_argument("--seed", type=int, default=0, help="Random seed for reproducibility.")
    parser.add_argument("--batch_size", type=int, default=16, help="Batch size")
    parser.add_argument("--micro_batch_size", type=int, default=4, help="Micro batch size for gradient accumulation.")
    parser.add_argument("--num_epochs", type=int, default=3, help="Number of epochs to train the model.")
    parser.add_argument("--learning_rate", type=float, default=3e-4, help="Learning rate for the optimizer.")
    parser.add_argument("--cutoff_len", type=int, default=256, help="Maximum sequence length for the model.")
    parser.add_argument("--lora_r", type=int, default=8, help="Rank for the LoRA adaptation.")
    parser.add_argument("--lora_alpha", type=int, default=16, help="Alpha parameter for LoRA.")
    parser.add_argument("--lora_dropout", type=float, default=0.05, help="Dropout rate for LoRA layers.")
    parser.add_argument("--lora_target_modules", type=str, nargs='+', default=["q_proj", "v_proj"], help="Model layers to which LoRA is applied.")
    parser.add_argument("--train_on_inputs", type=bool, default=True, help="If False, ignores inputs in the loss calculation.")
    parser.add_argument("--group_by_length", type=bool, default=False, help="Groups samples by length for efficient batching.")
    parser.add_argument("--wandb_project", type=str, default="", help="Weights & Biases project name for logging.")
    parser.add_argument("--wandb_run_name", type=str, default="trial1", help="Weights & Biases run name.")
    parser.add_argument("--wandb_watch", type=str, default="", help="Level of model watching in Weights & Biases.")
    parser.add_argument("--wandb_log_model", type=str, default="", help="Flag to log the model in Weights & Biases.")
    parser.add_argument("--resume_from_checkpoint", type=str, default=None, help="Path to a training checkpoint to resume from.")
    
    args = parser.parse_args()

    print('torch', version('torch'))
    print('transformers', version('transformers'))
    print('accelerate', version('accelerate'))
    print('# of gpus: ', torch.cuda.device_count())

    print("Checking PyTorch and CUDA...")
    check_cuda_availability()
    
    print("\nChecking environment variables...")
    check_env_variables()

    torch.cuda.memory_summary(device=None, abbreviated=False)

    torch.cuda.empty_cache()

    wandb.login(key="13dafa51b3dc2f45ebc7a6c217cc74c4b3974316")

    wandb_project_name = "nick-llm-project"
    wandb.init(
        project=wandb_project_name,

    )
    wandb.init(sync_tensorboard=True)

    args.wandb_project = wandb_project_name

    # Set PYTORCH_CUDA_ALLOC_CONF environment variable
    os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "caching_allocator"

    # Call the train function with the parsed arguments
    train(args)
    
    wandb.finish()