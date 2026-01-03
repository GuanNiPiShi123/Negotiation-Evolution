# -*- coding: utf-8 -*-
"""
Fine-tune or SFT a causal-LM + LoRA adapter on a multi-turn dataset.
Examples:
w/ Lora (on 2 H100 GPUs):
-------
python3 sft_train.py \
    --dataset_repo ./nego-evolution/nego_datasets/bargain/ \
    --task bargain \
    --output_dir outputs/sft/multiturn-bargain/Llama \
    --model_name meta-llama/Llama-3.1-8B-Instruct \ # or unsloth/gpt-oss-120b-unsloth-bnb-4bit
    --lower_bound_metric outcome.gap_ratio \
    --lower_bound 0.2 \
    --per_device_train_batch_size 2 \
    --per_device_eval_batch_size 2 \
    --gradient_accumulation_steps 2 \
    --num_train_epochs 4 \
    --learning_rate 1e-5 \
    --eval_steps 1 \
    --logging_steps 1 \
    --wandb_entity yue-team \
    --wandb_project nego-evolution \
    --use_lora \
    --use_4bit
"""

from __future__ import annotations

import argparse, os, json
from typing import Tuple, Optional

import torch
from unsloth import FastLanguageModel
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
)
import torch.distributed as dist
from peft import PeftConfig, PeftModel, LoraConfig, get_peft_model
from multiturn import MultiturnDataset
from trl import SFTConfig, SFTTrainer
import wandb

# --------------------------------------------------------------------------- #
# CLI
# --------------------------------------------------------------------------- #
def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser("Parameter-free multiturn SFT trainer")

    # Data / paths
    p.add_argument("--dataset_repo",    type=str, required=True)
    p.add_argument("--task",    type=str, required=True)
    p.add_argument("--output_dir",      type=str, required=True)
    p.add_argument("--resume_ckpt_dir", type=str, default=None)
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
    p.add_argument("--num_train_epochs", type=int, default=4)
    p.add_argument("--per_device_train_batch_size", type=int, default=4)
    p.add_argument("--per_device_eval_batch_size", type=int, default=2)
    p.add_argument("--gradient_accumulation_steps", type=int, default=4)
    p.add_argument("--eval_steps", type=int, default=50)
    p.add_argument("--save_total_limit", type=int, default=2)
    p.add_argument("--max_seq_length", type=int, default=4096)
    p.add_argument("--warmup_ratio", type=float, default=0)          
    p.add_argument("--logging_steps", type=int, default=100)          

    # Precision / hardware
    p.add_argument("--device", type=str, default="cuda")
    p.add_argument("--use_lora", action="store_true", default=False)
    p.add_argument("--use_4bit", action="store_true", default=False)

    # Tracking
    p.add_argument("--wandb_project", type=str)
    p.add_argument("--wandb_entity",  type=str)

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

# --------------------------------------------------------------------------- #
# Main
# --------------------------------------------------------------------------- #
def main() -> None:
    args = parse_args()
    os.makedirs(args.output_dir, exist_ok=True)
    
    os.environ['RANK'] = os.environ.get('RANK', '0')
    os.environ['WORLD_SIZE'] = os.environ.get('WORLD_SIZE', '1')
    os.environ['MASTER_ADDR'] = os.environ.get('MASTER_ADDR', '127.0.0.1')
    os.environ['MASTER_PORT'] = os.environ.get('MASTER_PORT', '29500')

    local_rank = int(os.environ.get('LOCAL_RANK', '0'))
    os.environ['LOCAL_RANK'] = os.environ.get('LOCAL_RANK', '0')
    dist.init_process_group(backend='nccl', init_method='env://')
    torch.cuda.set_device(local_rank)
    dist.barrier()

    # Dataset
    ds = MultiturnDataset(args.dataset_repo, args.task).to_sft_dataset(
        lower_bound_metric = args.lower_bound_metric, lower_bound = args.lower_bound, 
        use_train_thought = True)
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
            ) if args.use_lora else None

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
            max_seq_length = args.max_seq_length, # Choose any for long context!
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

    # DeepSpeed zero
    ds_cfg = {
        "zero_optimization": {
            "stage": 2,
            "overlap_comm": False,
            "reduce_bucket_size": "auto",
            "contiguous_gradients": True,
            "offload_optimizer": {"device": "none"},
            "offload_param": {"device": "none"},
        },
        "gradient_clipping": "auto",
        "train_batch_size": "auto",
        "train_micro_batch_size_per_gpu": args.per_device_train_batch_size,
        "gradient_accumulation_steps": args.gradient_accumulation_steps,
        "steps_per_print": 200,
    }

    # Trainer config (uses newly exposed warmup_ratio & logging_steps)
    train_cfg = SFTConfig(
        output_dir=args.output_dir,
        logging_steps=args.logging_steps,
        warmup_ratio=args.warmup_ratio,
        optim="adamw_torch",
        report_to="wandb" if args.wandb_project else "none",
        do_eval=False,
        eval_steps=args.eval_steps,
        save_strategy="epoch",
        eval_strategy="steps",
        group_by_length=True,
        gradient_checkpointing=True,
        lr_scheduler_type="cosine",
        metric_for_best_model="eval_loss",
        learning_rate=args.learning_rate,
        num_train_epochs=args.num_train_epochs,
        save_total_limit=args.save_total_limit,
        max_seq_length=args.max_seq_length,
        gradient_checkpointing_kwargs={'use_reentrant': False},
        per_device_train_batch_size=args.per_device_train_batch_size,
        per_device_eval_batch_size=args.per_device_eval_batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        run_name=args.output_dir,
        deepspeed=ds_cfg,
        fp16=not torch.cuda.is_fp16_supported(), 
        bf16=torch.cuda.is_bf16_supported(),  
    )

    # W&B
    if args.wandb_project and os.environ.get("LOCAL_RANK", "0") == "0":
        wandb.init(
            project=args.wandb_project,
            entity=args.wandb_entity,
            name=args.output_dir.replace("/", "_"),
            config=train_cfg.to_dict(),
            save_code=True,
            job_type="train",
        )
    
    trainer = SFTTrainer(
        model=model,
        train_dataset=ds["train"],
        #eval_dataset=ds["eval"],
        processing_class=tok,
        peft_config=lora_cfg,
        args=train_cfg,
    )
    trainer.train(resume_from_checkpoint=args.resume_ckpt_dir)

    trainer.save_model(args.output_dir)
    tok.save_pretrained(args.output_dir)

    if args.wandb_project:
        wandb.finish()

if __name__ == "__main__":

    main()
