# IN PROGRESS
    
#!/usr/bin/env python3
"""
GRPO training script for multiturn conversation models.
Example usage:
python grpo_train.py
    --dataset_name bargain \
    --metric_names "gap_ratio" "consistency" "bargain_interactivity" 
    --metric_weights 0.4 0.3 0.3 \
    --evol_stage 1\
    --user_generation_kwargs '{"model": "gpt-4o-mini", "api_key": "xxx"}' \
    --assistant_generation_kwargs '{"model": "gpt-4o-mini", "api_key": "xxx", "temperature": 0.6}' \
    --reward_generation_kwargs '{"model": "claude-3-5-sonnet-latest", "api_key": "xxx"}' \
    --dataset_repo ./nego_datasets/bargain/ \
    --model_name outputs/sft/multiturn-bargain/Llama/checkpoint-epoch-2/ \
    --output_dir outputs/grpo/multiturn-bargain/epoch1 \
    --per_device_train_batch_size 1 \
    --gradient_accumulation_steps 4 \
    --num_train_epochs 1 \
    --learning_rate 5e-6 \
    --logging_steps 1 \
    --wandb_entity yue-team \
    --wandb_project nego-evolution \
    --num_samples 3 \
    --max_new_turns 13 \
    --max_metric_workers 2 \
    --use_4bit
    --use_lora
"""
from __future__ import annotations

import argparse, os, json
import torch.distributed as dist
import wandb
import hashlib
from typing import Tuple, Optional
import numpy as np
import copy
import logging

from trl import GRPOConfig, GRPOTrainer

import torch
from unsloth import FastLanguageModel
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
)
from peft import PeftConfig, PeftModel, LoraConfig, get_peft_model
from multiturn import MultiturnDataset
from reward import multiturn_aware_reward
#from simulation import ChatSessionSimulator

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)

logger = logging.getLogger(__name__)

# --------------------------------------------------------------------------- #
# CLI
# --------------------------------------------------------------------------- #
def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser("Parameter-free multiturn PPO trainer")

    # Data / paths
    p.add_argument("--dataset_repo", type=str, required=True)
    p.add_argument("--dataset_name", type=str, required=True)
    p.add_argument("--metric_names", nargs="+", required=True)
    p.add_argument("--evol_stage", type=int, default=1)
    p.add_argument("--user_generation_kwargs", type=json.loads, default="{}")
    p.add_argument("--assistant_generation_kwargs", type=json.loads, default="{}")
    p.add_argument("--reward_generation_kwargs", type=json.loads, default="{}")
    p.add_argument("--metric_weights", type=float, nargs="+", default=None)
    p.add_argument("--max_new_turns", type=int, default=8)
    p.add_argument("--num_samples", type=int, default=3)

    p.add_argument("--output_dir",   type=str, required=True)
    
    p.add_argument("--lower_bound_metric", type=str, default=None)
    p.add_argument("--lower_bound",        type=float, default=0.0)

    # Base / adapter models
    p.add_argument("--model_name", type=str, required=True)
    p.add_argument("--peft_r",     type=int,   default=32)
    p.add_argument("--peft_alpha", type=int,   default=16)
    p.add_argument("--peft_dropout", type=float, default=0.1)
    p.add_argument("--target_modules",
                   type=str, default="q_proj,k_proj,v_proj,o_proj,gate_proj,up_proj,down_proj")

    # Optim & schedule
    p.add_argument("--learning_rate", type=float, default=1e-5)
    p.add_argument("--lbeta", type=float, default=0.05)
    p.add_argument("--warmup_steps", type=int, default=50)
    p.add_argument("--temperature", type=float, default=0.8)
    p.add_argument("--num_generations", type=int, default=4)
    p.add_argument("--num_train_epochs", type=int, default=3)
    p.add_argument("--per_device_train_batch_size", type=int, default=4)
    p.add_argument("--gradient_accumulation_steps", type=int, default=4)
    p.add_argument("--save_steps", type=int, default=50)
    p.add_argument("--logging_steps", type=int, default=1)
    p.add_argument("--max_model_len", type=int, default=8196)
    p.add_argument("--max_new_tokens", type=int, default=512) 
    p.add_argument("--max_metric_workers", type=int, default=4)
    
    # Precision / hardware
    p.add_argument("--device", type=str, default="cuda")
    p.add_argument("--use_lora", action="store_true", default=False)
    p.add_argument("--use_4bit", action="store_true", default=False)

    # Tracking
    p.add_argument("--wandb_project", type=str)
    p.add_argument("--wandb_entity",  type=str)

    p.add_argument("--debug", action="store_true")

    # Optional JSON/YAML override
    p.add_argument("--config_file", type=str)

    args = p.parse_args()
    if args.config_file:
        with open(args.config_file) as f:
            override = json.load(f) if args.config_file.endswith(".json") else \
                       __import__("yaml").safe_load(f)
        for k, v in override.items():
            setattr(args, k, v)
    return args

# --------------------------------------------------------------------------- #
# Utilities
# --------------------------------------------------------------------------- #
def load_model_and_tokenizer(
    model_name: str,
    bnb_cfg: Optional[BitsAndBytesConfig],
    lora_cfg: Optional[LoraConfig],
    device: str = "cuda",
    is_eval: bool = False,
) -> Tuple[torch.nn.Module, AutoTokenizer]:
    try:
        pc = PeftConfig.from_pretrained(model_name)
        base = AutoModelForCausalLM.from_pretrained(
            pc.base_model_name_or_path,
            device_map={"": device},
            quantization_config=bnb_cfg,
            trust_remote_code=True,
        )
        model = PeftModel.from_pretrained(base, model_name, is_trainable=not is_eval)
        tok = AutoTokenizer.from_pretrained(pc.base_model_name_or_path, trust_remote_code=True)
    except Exception:
        logger.error(f"Failed to load PeftConfig for {model_name}, loading as base model.")
        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            device_map={"": device},
            quantization_config=bnb_cfg,
            trust_remote_code=True,
        )
        tok = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
        if lora_cfg:
            model = get_peft_model(model, lora_cfg)

    tok.padding_side, tok.pad_token = ("left" if is_eval else "right"), tok.eos_token
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total     = sum(p.numel() for p in model.parameters())
    print(f"Trainable params: {trainable:,}/{total:,} ({trainable/total:.2%})")
    return model, tok

def process_dataset(dataset, tokenizer, max_query_tokens, add_system_prompt):
    """Process dataset to create query tensors for GRPO training"""
    from collections import defaultdict
    from datasets import Dataset
    
    def tokenize(x, add_system_prompt):
        if add_system_prompt:
            even_indices = np.array(list(range(2, len(x["prompt"]), 2)))
        else:
            even_indices = np.array(list(range(1, len(x["prompt"]), 2)))
        
        new_samples = []
        for input_size in even_indices:
            new_x = copy.deepcopy(x)
            new_x["prompts"] = x["prompt"][:input_size]
            '''
            new_x["query"] = tokenizer.apply_chat_template(new_x["prompt"], 
                                                           tokenize=False, 
                                                           add_generation_prompt=True)
            new_x["input_ids"] = tokenizer.encode(new_x["query"],
                                                  max_length=max_query_tokens,
                                                  truncation=True)
            '''
            new_samples.append(new_x)
        
        return new_samples

    processed_slices = defaultdict(list)
    
    for item in dataset:
        tokenized_samples = tokenize(item, add_system_prompt)
        for sample in tokenized_samples:
            for key, value in sample.items():
                processed_slices[key].append(value)
    
    new_dataset = Dataset.from_dict(dict(processed_slices))
    #new_dataset.set_format(type="torch")
    return new_dataset

# --------------------------------------------------------------------------- #
# Main
# --------------------------------------------------------------------------- #
def main() -> None:
    args = parse_args()
    os.makedirs(args.output_dir, exist_ok=True)

    # Important for initializing vllm base model per GPU
    local_rank = int(os.environ.get('LOCAL_RANK', '0'))
    dist.init_process_group(backend='nccl', init_method=None)
    torch.cuda.set_device(local_rank)
    dist.barrier()
    
    datasets_info = {"bargain": "bargain", "persuasion": "persuasion"}

    # Dataset
    add_system_prompt = True
    ds = MultiturnDataset(args.dataset_repo, args.dataset_name, add_system_prompt = add_system_prompt).to_inputs_dataset( 
        args.evol_stage, lower_bound_metric = args.lower_bound_metric, lower_bound = args.lower_bound, use_train_thought = True)
    
    if "gpt" not in args.model_name:
        # Bits-and-bytes
        bnb_cfg = BitsAndBytesConfig(
            load_in_4bit=args.use_4bit,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_use_double_quant=False,
            bnb_4bit_compute_dtype=torch.bfloat16,
            ) if args.use_4bit else None

    # LoRA
        lora_cfg = LoraConfig(
            r=args.peft_r,
            lora_alpha=args.peft_alpha,
            lora_dropout=args.peft_dropout,
            bias="none",
            task_type="CAUSAL_LM",
            init_lora_weights="gaussian",
            target_modules=args.target_modules.split(","),
            )

    # Load model
        model, tok = load_model_and_tokenizer(
            args.model_name,
            bnb_cfg=bnb_cfg,
            lora_cfg=lora_cfg,
            device=args.device,
            is_eval=False,
            )
    else:
        model, tok = FastLanguageModel.from_pretrained(
            model_name = args.model_name,
            dtype = None, # None for auto detection
            max_seq_length = args.max_model_len, # Choose any for long context!
            load_in_4bit = args.use_4bit,  # 4 bit quantization to reduce memory
            full_finetuning = False, 
            )
        
        model = FastLanguageModel.get_peft_model(
            model,
            r = args.peft_r, # Choose any number > 0 ! Suggested 8, 16, 32, 64, 128
            target_modules = args.target_modules.split(","),
            lora_alpha = args.peft_alpha,
            lora_dropout = args.peft_dropout, # Supports any, but = 0 is optimized
            bias = "none",    # Supports any, but = "none" is optimized
            use_gradient_checkpointing = "unsloth", # True or "unsloth" for very long context
            random_state = 3407,
            use_rslora = False,  # We support rank stabilized LoRA
            loftq_config = None, # And LoftQ
            )

    # Process dataset
    train_dataset = process_dataset(
        ds["train"], 
        tok, 
        args.max_model_len - args.max_new_tokens,
        add_system_prompt
    )

    # W&B
    if args.wandb_project and os.environ.get("LOCAL_RANK", "0") == "0":
        wandb.init(
            project=args.wandb_project,
            entity=args.wandb_entity,
            name=args.output_dir.replace("/", "_"),
            config=vars(args),
            save_code=True,
            job_type="debug" if args.debug else "train",
        )

    # GRPO Config
    grpo_config = GRPOConfig(
        output_dir = args.output_dir,
        learning_rate = args.learning_rate,
        logging_steps = args.logging_steps,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        max_completion_length = args.max_new_tokens,
        max_prompt_length= args.max_model_len - args.max_new_tokens,
        num_train_epochs = args.num_train_epochs,
        per_device_train_batch_size = args.per_device_train_batch_size,
        save_steps = args.save_steps,
        warmup_steps = args.warmup_steps,
        weight_decay = 0.1,
        max_grad_norm = 0.1,
        beta=args.beta, 
        gradient_checkpointing = True,
        remove_unused_columns=False,
        num_generations=args.num_generations,
        lr_scheduler_type="cosine",
        temperature=args.temperature,
    )

    ######################## REWARD FUNCTION ########################
    def compute_hash(text: str) -> str:
        return hashlib.md5(text.encode("utf-8")).hexdigest()
    
    str_prompt_to_multiturn_data_map = {}
    
    def process_prompt_mapping(row):
        str_prompt = tok.apply_chat_template(row["prompt"], tokenize=False, add_generation_prompt=True)
        str_prompt_to_multiturn_data_map.setdefault(
            compute_hash(str_prompt), 
            {k: row[k] for k in ["prompt"]}
        )
        return row
    
    train_dataset.map(process_prompt_mapping, load_from_cache_file=False)
    
    negotiation_llm_model_kwargs = {
        "local_model": model.pretrained_model if hasattr(model, 'pretrained_model') else model,
        "local_tokenizer": tok
    }

    def compute_rewards(prompts, completions, Price_Buyer, Price_Seller, Title, Description,**kwargs):
        """Compute rewards for GRPO training"""
        rewards = []
        for prompt, response in zip(prompts, completions):
            multiturn_data = str_prompt_to_multiturn_data_map.get(compute_hash(tok.apply_chat_template(prompt, tokenize=False, add_generation_prompt=True)))
            if multiturn_data is None:
                rewards.append(0.0)
                continue
                
            chat_history = multiturn_data["prompt"] + [{"role": "assistant", "content": response}]
            if add_system_prompt:
                response_location = len(chat_history)-1 -1
            else:
                response_location = len(chat_history)-1
                
            reward_info = multiturn_aware_reward(
                chat_history=chat_history,
                response_location=response_location,
                task_desc=datasets_info[args.dataset_name],
                metric_names=args.metric_names,
                metric_weights=args.metric_weights,
                user_generation_kwargs=args.user_generation_kwargs,
                assistant_generation_kwargs=args.assistant_generation_kwargs,
                reward_generation_kwargs=args.reward_generation_kwargs,
                num_samples=args.num_samples,
                max_new_turns=args.max_new_turns,
                max_metric_workers=args.max_metric_workers,
                Price_Buyer= Price_Buyer, 
                Price_Seller= Price_Seller, 
                Title = Title, 
                Description = Description,
                **negotiation_llm_model_kwargs
            )
            rewards.append(np.mean(reward_info["MR"]))
        return rewards
    
    #GRPO Trainer
    trainer = GRPOTrainer(
        args=grpo_config,
        model = model, 
        reward_funcs = compute_rewards,
        train_dataset = train_dataset,
        processing_class = tok,
        peft_config=lora_cfg
        )
    trainer.train()
    # Final save
    trainer.save_model(args.output_dir)
    tok.save_pretrained(args.output_dir)

    if args.wandb_project:
        wandb.finish()

if __name__ == "__main__":
    main()
