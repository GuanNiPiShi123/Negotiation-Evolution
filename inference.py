# -*- coding: utf-8 -*-
"""
Data Generation for Evolution Training.

Example run:
python inference.py
    --dataset_repo ./nego_datasets/bargain/ \
    --dataset_name bargain \
    --evol_stage sft-1\
    --model_name outputs/sft/multiturn-bargain/Llama/checkpoint-epoch-2/ \
    --max_new_turns 13 \
    --num_mcts_sims 6 \
    --max_realizations
    --evolution_ratio 0.0 \
    --num_samples 4 \
    --user_generation_kwargs '{"model": "gpt-4o-mini", "api_key": "xxx"}' \
    --assistant_generation_kwargs '{"model": "gpt"}' \
    --reward_generation_kwargs '{"model": "claude-3-5-sonnet-latest", "api_key": "xxx"}' \
    --metric_names "gap_ratio" 
    --metric_weights 1.0 \
    --use_4bit\
    --use_lora \
    --use_mcts \
    --add_system_prompt
"""
import os
import argparse
import json
import re
import random
import numpy as np
from tqdm import tqdm
import logging

from typing import Optional, Tuple
import torch
import torch.distributed as dist

from peft import PeftConfig, PeftModel, LoraConfig, get_peft_model
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
#from unsloth import FastLanguageModel

from mcts import OpenLoopMCTS, model_inference
from modules import UserSimulator, NEGOLLM_TERMINATION_SIGNAL
from reward import multiturn_aware_reward_inference
from utils.template import strip_system_prompt

logger = logging.getLogger(__name__)

def parse_args():
    p = argparse.ArgumentParser("Distributed multiturn inference")
    
    # LoRA config
    p.add_argument("--peft_r", type=int, default=32)
    p.add_argument("--peft_alpha", type=int, default=16)
    p.add_argument(
        "--target_modules",
        type=str,
        default="q_proj,k_proj,v_proj,o_proj,gate_proj,up_proj,down_proj"
    )
    
    # Evaluation + model
    p.add_argument("--dataset_repo", type=str, required=True)
    p.add_argument("--dataset_name", type=str, required=True)
    p.add_argument("--model_name", type=str, required=True)
    p.add_argument("--evol_stage", type=str, default="evol_1")
    p.add_argument("--max_new_turns", type=int, default=13)
    p.add_argument('--num_mcts_sims', type=int, default=6, help='number of mcts simulations')
    p.add_argument("--add_system_prompt", action="store_true", default=False)
    p.add_argument('--max_realizations', type=int, default=3, help='number of realizations per mcts state')
    p.add_argument("--cpuct", type=float, default=1.0)
    p.add_argument("--evolution_ratio", type=float, default=0.8)
    p.add_argument('--Q_0', type=float, default=0.25, help='initial Q value for unitialized states. to control exploration')
    p.add_argument("--num_samples", type=int, default=6)
    p.add_argument("--metric_names", nargs="+", required=True)
    p.add_argument("--metric_weights", type=float, nargs="+", default=None)
    
    p.add_argument("--user_generation_kwargs", type=json.loads, default="{}")
    p.add_argument("--assistant_generation_kwargs", type=json.loads, default={})
    p.add_argument("--reward_generation_kwargs", type=json.loads, default={})
    
    p.add_argument("--use_lora", action="store_true", default=False)
    p.add_argument("--use_4bit", action="store_true", default=False)
    p.add_argument("--use_mcts", action="store_true", default=False)
    
    return p.parse_args()

def load_model_and_tokenizer(
    model_name: str,
    bnb_cfg: Optional[BitsAndBytesConfig],
    lora_cfg: Optional[LoraConfig],
    device: str = "cuda",
    is_eval: bool = True,
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
            trust_remote_code=True
        )
        tok = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
        if lora_cfg:
            model = get_peft_model(model, lora_cfg)

    tok.padding_side, tok.pad_token = ("left" if is_eval else "right"), tok.eos_token
    #trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    #total     = sum(p.numel() for p in model.parameters())
    #print(f"Trainable params: {trainable:,}/{total:,} ({trainable/total:.2%})")
    return model.eval(), tok

def main():
    args = parse_args()
    
    # Distributed init                                                                                                      
    os.environ['RANK'] = os.environ.get('RANK', '0')                                                                        
    os.environ['WORLD_SIZE'] = os.environ.get('WORLD_SIZE', '1')                                                            
    os.environ['MASTER_ADDR'] = os.environ.get('MASTER_ADDR', '127.0.0.1')                                                  
    os.environ['MASTER_PORT'] = os.environ.get('MASTER_PORT', '29500') 
                                                                             
    local_rank = int(os.environ.get('LOCAL_RANK', '0'))                                                                     
    os.environ['LOCAL_RANK'] = os.environ.get('LOCAL_RANK', '0')                                                            
    dist.init_process_group(backend='nccl', init_method='env://')                                                           
    torch.cuda.set_device(local_rank)                                                                                       
    dist.barrier() 

    add_system_prompt = args.add_system_prompt
    datasets_info = {"bargain": "bargain", "persuasion": "persuasion"}
    task_desc = datasets_info[args.dataset_name]
    
    with open(os.path.join(args.dataset_repo, "sft_test.json"),'r', encoding='utf-8') as test_sft:
        testset = json.load(test_sft)
    
    if "gpt" not in args.model_name:
        lora_cfg = LoraConfig(
            r=args.peft_r,
            lora_alpha=args.peft_alpha,
            bias="none",
            task_type="CAUSAL_LM",
            init_lora_weights="gaussian",
            target_modules=args.target_modules.split(","),
        ) if args.use_lora else None

        # Bits-and-bytes
        bnb_cfg = BitsAndBytesConfig(
            load_in_4bit=args.use_4bit,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_use_double_quant=False,
            bnb_4bit_compute_dtype=torch.bfloat16,
            ) if args.use_4bit else None

        torch.cuda.empty_cache()

        # Load model
        model, tok = load_model_and_tokenizer(
            model_name=args.model_name,
            lora_cfg=lora_cfg,
            bnb_cfg=bnb_cfg, # No quantization for eval
            is_eval=True
            )
    else:
        '''
        model, tok = FastLanguageModel.from_pretrained(
            model_name = args.model_name,
            dtype = None, # None for auto detection
            #max_seq_length = args.max_model_len, # Choose any for long context!
            load_in_4bit = args.use_4bit,  # 4 bit quantization to reduce memory
            full_finetuning = False 
            )
        
        model = FastLanguageModel.get_peft_model(
            model,
            r = args.peft_r, # Choose any number > 0 ! Suggested 8, 16, 32, 64, 128
            target_modules = args.target_modules.split(","),
            lora_alpha = args.peft_alpha,
            bias = "none",    # Supports any, but = "none" is optimized
            use_gradient_checkpointing = "unsloth", # True or "unsloth" for very long context
            random_state = 3407,
            use_rslora = False,  # We support rank stabilized LoRA
            loftq_config = None, # And LoftQ
            )
        model.eval()
        '''
        pass
    # Run local inference
    test_data = []
    test_reward_data = []
    
    for ex in tqdm(testset, desc=f"Processing evolution examples"):
        #Simulation Started!
        ex["Price_Buyer"] = str(ex["buyer_target"])
        ex["Price_Seller"] = str(ex["seller_target"])
        
        if add_system_prompt: 
            if "bargain" in task_desc:
                with open(os.path.join('./prompts', 'bargain_system_prompt.txt'), 'r') as f:
                    SYSTEM_PROMPT = f.read()
                    SYSTEM_PROMPT.format(
                        Title = ex['Title'],
                        Description = ' '.join(ex['Description']),
                        Buyer_Price = str(ex['buyer_target'])
                    )
            else:
                with open(os.path.join('./prompts', 'persuasion_system_prompt.txt'), 'r') as f:
                    SYSTEM_PROMPT = f.read()
            ex["prompt"] = SYSTEM_PROMPT
            session = [{"role":"system","content":ex["prompt"]}]
        else:
            session = []

        if "bargain" in task_desc:
            if random.random() > 0.5:
                user_simulator = UserSimulator(task_desc = task_desc, **args.user_generation_kwargs)
                user_response = user_simulator(session, ex["Title"], " ".join(ex["Description"]), ex["Price_Seller"])
                session.append({"role":"user", "content":user_response})
            else:
                session.append({"role":"user", "content":""})
            session.append({"role":"assistant", "content":model_inference(args.model_name, model, tok, session, args.assistant_generation_kwargs)})
            
            user_simulator = UserSimulator(task_desc = task_desc, **args.user_generation_kwargs)
            user_response = user_simulator(session, ex["Title"], " ".join(ex["Description"]), ex["Price_Seller"])
            session.append({"role":"user", "content":user_response})
        else:
            if random.random() > 0.5:
                user_simulator = UserSimulator(task_desc = task_desc, **args.user_generation_kwargs)
                user_response = user_simulator(session, "", "", "")
                session.append({"role":"user", "content":user_response})
            else:
                session.append({"role":"user", "content":""})
            session.append({"role":"assistant", "content":model_inference(args.model_name, model, tok, session, args.assistant_generation_kwargs)})
            user_simulator = UserSimulator(task_desc = task_desc, **args.user_generation_kwargs)
            user_response = user_simulator(session, "", "", "")
            session.append({"role":"user", "content":user_response})

        msg_budget = args.max_new_turns-2
        while msg_budget>0:
            if args.use_mcts:
                dialog_planner = OpenLoopMCTS(args.max_realizations, 
                                          task_desc, 
                                          args.Q_0, 
                                          args.cpuct, 
                                          args.num_samples, 
                                          args.evolution_ratio, 
                                          ex["Title"], 
                                          " ".join(ex["Description"]), 
                                          ex["Price_Buyer"], 
                                          ex["Price_Seller"],
                                          args.assistant_generation_kwargs, 
                                          args.user_generation_kwargs, 
                                          args.reward_generation_kwargs,
                                          model,
                                          tok,
                                          args.model_name,
                                          test = True)
                for i in tqdm(range(args.num_MCTS_sims)):
                    dialog_planner.search(session, msg_budget)
                policy_next_da = dialog_planner.get_action_prob(session, msg_budget)
                response = dialog_planner.get_best_realization(session, policy_next_da)
            else:
                response = model_inference(args.model_name, model, tok, session, args.assistant_generation_kwargs)
            session.append({"role":"assistant", "content":response})
            msg_budget -= 1
            
            if msg_budget == 0:
                break
            
            if "bargain" in task_desc:
                user_simulator = UserSimulator(task_desc = task_desc, **args.user_generation_kwargs)
                user_response = user_simulator(session, ex["Title"], " ".join(ex["Description"]), ex["Price_Seller"])
            else:
                user_simulator = UserSimulator(task_desc = task_desc, **args.user_generation_kwargs)
                user_response = user_simulator(session, "", "", "")
            
            session.append({"role":"user", "content":user_response})
            msg_budget -= 1
            
            if NEGOLLM_TERMINATION_SIGNAL in user_response:
                break
        
        processed_session = process(strip_system_prompt(session))
        
        if processed_session !=[]:
            test_data.append({"generated_session": processed_session, "Title":ex["Title"], "Description": ex["Description"],"buyer_target":ex["Price_Buyer"],"seller_target":ex["Price_Seller"]})
        
        #Add reward
        test_reward_data.append(processed_session)


    with open(os.path.join(args.dataset_repo, "test_"+str(args.evol_stage)+".json"),'w') as f:
        json.dump(test_data, f, ensure_ascii=False)
        
    reward_dict = multiturn_aware_reward_inference(task_desc=task_desc, sessions=test_reward_data, metric_names= args.metric_names, metric_weights = args.metric_weights, reward_generation_kwargs = args.reward_generation_kwargs)
    print(reward_dict)
    print(np.mean(reward_dict["MR"]))

def process(sessions, use_thought =True):
    strategy_pattern = r'<intent>(.*?)</intent>'
    thought_pattern = r'<think>(.*?)</think>'
    response_pattern = r'<response>(.*?)</response>'
    new_sessions = []
    for session in sessions:
        if session["role"] == "assistant":
            strategy_match = re.search(strategy_pattern, session["content"])
            response_match = re.search(thought_pattern, session["content"])
            if use_thought:
                thought_match = re.search(thought_pattern, session["content"])
                if thought_match and strategy_match and response_match:
                    new_sessions.append({"role":"assistant", "thought":thought_match.group(1), "strategy":strategy_match.group(1), "content":response_match.group(1)})
                else:
                    new_sessions.append({"role":"assistant", "thought":"", "strategy":"Unknown", "content":session["content"]})
            else:
                if strategy_match and response_match:
                    new_sessions.append({"role":"assistant", "strategy":strategy_match.group(1), "content":response_match.group(1)})
                else:
                    new_sessions.append({"role":"assistant", "strategy":"Unknown", "content":session["content"]})
        else:
            new_sessions.append(session)
            
    return new_sessions
        
if __name__ == "__main__":
    main()

