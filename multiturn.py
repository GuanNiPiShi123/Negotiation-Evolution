"""
nego_datasets.multiturn
~~~~~~~~~~~~~~~~~~~~~~~~~~~~
Unified loader + wrapper for multi-turn chat data.

• `to_sft_dataset()`   → DatasetDict {text}
• `to_inputs_dataset()`→ DatasetDict {prompt, single_turn_completion}
"""

import os
import json
import random
import numpy as np
from typing import Any, Dict, List, Optional, Sequence, Union
from datasets import Dataset, DatasetDict

import logging
logger = logging.getLogger(__name__)

# --------------------------------------------------------------------------- #
# uniform splitter                                                            #
# --------------------------------------------------------------------------- #
def _uniform_split(
    full_ds: Dataset,
    *,
    eval_ratio: float,
    n_eval: Optional[int],
    seed: int,
) -> DatasetDict:
    k = n_eval if n_eval is not None else int(eval_ratio * len(full_ds))
    k = min(k, len(full_ds))

    random.seed(seed)
    eval_idx = set(random.sample(range(len(full_ds)), k=k))
    train_idx = [i for i in range(len(full_ds)) if i not in eval_idx]

    return DatasetDict(
        {
            "train": full_ds.select(train_idx),
            "eval": full_ds.select(sorted(eval_idx)),
        }
    )

# --------------------------------------------------------------------------- #
# main dataclass                                                              #
# --------------------------------------------------------------------------- #
class MultiturnDataset:
    def __init__(
        self,
        local_dir: str,
        task: str,
        *,
        seed: int = 42,
        add_system_prompt: bool = True,
    ):
        
        self.local_dir = local_dir
        self.seed = seed
        self.task = task
        self.add_system_prompt = add_system_prompt

    # ------------------------------------------------------------------ #
    # SFT                                                                #
    # ------------------------------------------------------------------ #
    def to_sft_dataset(
        self,
        *,
        lower_bound_metric: Optional[str] = None,
        lower_bound: Optional[float] = 0.0,
        use_train_thought = True
    ) -> DatasetDict:
        
        with open(os.path.join(self.local_dir, "sft_train.json"),'r', encoding='utf-8') as train_sft:
            train_lst = json.load(train_sft)
        
        if os.path.exists(os.path.join(self.local_dir, "sft_valid.json")):
            with open(os.path.join(self.local_dir, "sft_valid.json"),'r', encoding='utf-8') as valid_sft:
                valid_lst = json.load(valid_sft)
        else:
            valid_lst = []
        # Build SFT dialogues, filtering by optional metric threshold
        def serialize(data_lst, lower_bound_metric, use_train_thought =False):
            serialized_dialogues = []
            for row_id, row in enumerate(data_lst):
                if lower_bound_metric:
                    try:
                        metric = row
                        for key in lower_bound_metric.split("."):
                            metric = metric.get(key, {})
                        value = np.asarray(metric).mean().item()
                    except Exception as e:
                        logger.error(f"Failed to extract metric '{lower_bound_metric}' from row: {row} — {e}")
                        continue
                    
                    if value < lower_bound:
                        logger.warning(
                            f"Filtered out conv_id={row_id} "
                            f"due to {lower_bound_metric}={value:.3f} < {lower_bound:.3f}"
                            )
                        continue
                    
                if "bargain" in self.task:
                    if self.add_system_prompt:
                        with open(os.path.join('./prompts', 'bargain_system_prompt.txt'), 'r') as f:
                            SYSTEM_PROMPT = f.read()
                            SYSTEM_PROMPT.format(
                                Title = row['Title'],
                                Description = ' '.join(row['Description']),
                                Price_Buyer = row['buyer_target']
                                )
                            #self.sys_msg = [{"role": "system", "content": SYSTEM_PROMPT}] if self.add_system_prompt else []

                            if use_train_thought and "thought" in row["prompt"][0]:
                                messages = [{"role": "system", "content": SYSTEM_PROMPT}] + \
                                    [{"role":p["role"], "content":"<think>"+p["thought"] + "</think>"+"<intent>"+p["strategy"]+"</intent>"+"<response>"+p["content"]+"</response>"} if p["role"]=="assistant" 
                                    else {"role":p["role"], "content":p["content"]} for p in row["prompt"]] 
                            else:
                                messages = [{"role": "system", "content": SYSTEM_PROMPT}] + \
                                    [{"role":p["role"], "content":"<intent>"+p["strategy"]+"</intent>"+"<response>"+p["content"]+"</response>"} if p["role"]=="assistant" 
                                     else {"role":p["role"], "content":p["content"]} for p in row["prompt"] ] 
                    else:
                        if use_train_thought and "thought" in row["prompt"][0]:
                            messages = [{"role":p["role"], "content":"<think>"+p["thought"] + "</think>"+"<intent>"+p["strategy"]+"</intent>"+"<response>"+p["content"]+"</response>"} if p["role"]=="assistant" 
                                else {"role":p["role"], "content":p["content"]} for p in row["prompt"]] 
                        else:
                            messages = [{"role":p["role"], "content":"<intent>"+p["strategy"]+"</intent>"+"<response>"+p["content"]+"</response>"} if p["role"]=="assistant" 
                                else {"role":p["role"], "content":p["content"]} for p in row["prompt"]]
                        
                elif "persuasion" in self.task and self.add_system_prompt:
                    if self.add_system_prompt:
                        with open(os.path.join('./prompts', 'persuasion_system_prompt.txt'), 'r') as f:
                            SYSTEM_PROMPT = f.read()
                            #self.sys_msg = [{"role": "system", "content": SYSTEM_PROMPT}] if self.add_system_prompt else []
                            if use_train_thought and "thought" in row["prompt"][0]:
                                messages = [{"role": "system", "content": SYSTEM_PROMPT}] + \
                                    [{"role":p["role"], "content":"<think>"+p["thought"] + "</think>"+"<intent>"+p["strategy"]+"</intent>"+"<response>"+p["content"]+"</response>"} if p["role"]=="assistant" 
                                    else {"role":p["role"], "content":p["content"]} for p in row["prompt"]] 
                            else:
                                messages = [{"role": "system", "content": SYSTEM_PROMPT}] + \
                                    [{"role":p["role"], "content":"<intent>"+p["strategy"]+"</intent>"+"<response>"+p["content"]+"</response>"} if p["role"]=="assistant" 
                                     else {"role":p["role"], "content":p["content"]} for p in row["prompt"] ]
                    else:
                        if use_train_thought and "thought" in row["prompt"][0]:
                            messages = [{"role":p["role"], "content":"<think>"+p["thought"] + "</think>"+"<intent>"+p["strategy"]+"</intent>"+"<response>"+p["content"]+"</response>"} if p["role"]=="assistant" 
                                else {"role":p["role"], "content":p["content"]} for p in row["prompt"]]
                        else:
                            messages =  [{"role":p["role"], "content":"<intent>"+p["strategy"]+"</intent>"+"<response>"+p["content"]+"</response>"} if p["role"]=="assistant" 
                                else {"role":p["role"], "content":p["content"]} for p in row["prompt"]]
                            
                serialized_dialogues.append(messages)
            print(len(serialized_dialogues))
            logger.info(
                    f"Converted {len(serialized_dialogues)} dialogues "
                    f"(filter: {lower_bound_metric} ≥ {lower_bound}); "
                    f"retention ratio: {len(serialized_dialogues)/len(data_lst):.2f}"
                )
            return serialized_dialogues
        
        logger.info("Converting Training data ...")
        train_serialized_dialogues = serialize(train_lst, lower_bound_metric, use_train_thought = use_train_thought)
        logger.info("Converting Validation data ...")
        if len(valid_lst) != 0:
            valid_serialized_dialogues = serialize(valid_lst, lower_bound_metric)
        else:
            valid_serialized_dialogues = []

        train_dataset = Dataset.from_dict({"messages": train_serialized_dialogues})
        valid_dataset = Dataset.from_dict({"messages": valid_serialized_dialogues})
        
        sft_dataset = DatasetDict(
            {
                "train": train_dataset,
                "eval": valid_dataset,
            }
        )

        return sft_dataset
    
    def to_inputs_dataset(
        self, 
        evol_stage,
        lower_bound_metric: Optional[str] = None,
        lower_bound: Optional[float] = 0.0,
        use_train_thought = True
    ) -> DatasetDict:
        
        with open(os.path.join(self.local_dir, "grpo_train_"+str(evol_stage)+".json"),'r', encoding='utf-8') as train_sft:
            train_lst = json.load(train_sft)
        
        if os.path.exists(os.path.join(self.local_dir, "grpo_valid_"+str(evol_stage)+".json")):
            with open(os.path.join(self.local_dir, "grpo_valid_"+str(evol_stage)+".json"),'r', encoding='utf-8') as valid_sft:
                valid_lst = json.load(valid_sft)
        else:
            valid_lst = []
            
        # Build GRPO dialogues, filtering by optional metric threshold
        def serialize(data_lst, lower_bound_metric, use_train_thought =False):
            serialized_dialogues = []
            buyer_target_price, seller_target_price, description, item_name = [], [], [], []
            for row_id, row in enumerate(data_lst):
                if lower_bound_metric:
                    try:
                        metric = row
                        for key in lower_bound_metric.split("."):
                            metric = metric.get(key, {})
                        value = np.asarray(metric).mean().item()
                    except Exception as e:
                        logger.error(f"Failed to extract metric '{lower_bound_metric}' from row: {row} — {e}")
                        continue
                    
                    if value < lower_bound:
                        logger.warning(
                            f"Filtered out conv_id={row_id} "
                            f"due to {lower_bound_metric}={value:.3f} < {lower_bound:.3f}"
                            )
                        continue
                    
                if "bargain" in self.task:
                    if self.add_system_prompt:
                        with open(os.path.join('./prompts', 'bargain_system_prompt.txt'), 'r') as f:
                            SYSTEM_PROMPT = f.read()
                            SYSTEM_PROMPT.format(
                                Title = row['Title'],
                                Description = ' '.join(row['Description']),
                                Buyer_Price = str(row['buyer_target'])
                                )
                            #self.sys_msg = [{"role": "system", "content": SYSTEM_PROMPT}] if self.add_system_prompt else []
                            
                            if use_train_thought and "thought" in row["prompt"][0]:
                                messages = [{"role": "system", "content": SYSTEM_PROMPT}] + \
                                    [{"role":p["role"], "content":"<think>"+p["thought"] + "</think>"+"<intent>"+p["strategy"]+"</intent>"+"<response>"+p["content"]+"</response>"} if p["role"]=="assistant" 
                                    else {"role":p["role"], "content":p["content"]} for p in row["prompt"]] 
                            else:
                                messages = [{"role": "system", "content": SYSTEM_PROMPT}] + \
                                    [{"role":p["role"], "content":"<intent>"+p["strategy"]+"</intent>"+"<response>"+p["content"]+"</response>"} if p["role"]=="assistant" 
                                     else {"role":p["role"], "content":p["content"]} for p in row["prompt"] ] 
                    else:
                        if use_train_thought and "thought" in row["prompt"][0]:
                            messages = [{"role":p["role"], "content":"<think>"+p["thought"] + "</think>"+"<intent>"+p["strategy"]+"</intent>"+"<response>"+p["content"]+"</response>"} if p["role"]=="assistant" 
                                else {"role":p["role"], "content":p["content"]} for p in row["prompt"]] 
                        else:
                            messages = [{"role":p["role"], "content":"<intent>"+p["strategy"]+"</intent>"+"<response>"+p["content"]+"</response>"} if p["role"]=="assistant" 
                                else {"role":p["role"], "content":p["content"]} for p in row["prompt"]]
                    buyer_target_price.append(str(row['buyer_target'])) 
                    seller_target_price.append(str(row['seller_target'])) 
                    description.append(' '.join(row['Description'])) 
                    item_name.append(row['Title'])
                    
                elif "persuasion" in self.task and self.add_system_prompt:
                    if self.add_system_prompt:
                        with open(os.path.join('./prompts', 'persuasion_system_prompt.txt'), 'r') as f:
                            SYSTEM_PROMPT = f.read()
                            #self.sys_msg = [{"role": "system", "content": SYSTEM_PROMPT}] if self.add_system_prompt else []
                            if use_train_thought and "thought" in row["prompt"][0]:
                                messages = [{"role": "system", "content": SYSTEM_PROMPT}] + \
                                    [{"role":p["role"], "content":"<think>"+p["thought"] + "</think>"+"<intent>"+p["strategy"]+"</intent>"+"<response>"+p["content"]+"</response>"} if p["role"]=="assistant" 
                                    else {"role":p["role"], "content":p["content"]} for p in row["prompt"]] 
                            else:
                                messages = [{"role": "system", "content": SYSTEM_PROMPT}] + \
                                    [{"role":p["role"], "content":"<intent>"+p["strategy"]+"</intent>"+"<response>"+p["content"]+"</response>"} if p["role"]=="assistant" 
                                     else {"role":p["role"], "content":p["content"]} for p in row["prompt"] ]
                    else:
                        if use_train_thought and "thought" in row["prompt"][0]:
                            messages = [{"role":p["role"], "content":"<think>"+p["thought"] + "</think>"+"<intent>"+p["strategy"]+"</intent>"+"<response>"+p["content"]+"</response>"} if p["role"]=="assistant" 
                                else {"role":p["role"], "content":p["content"]} for p in row["prompt"]]
                        else:
                            messages =  [{"role":p["role"], "content":"<intent>"+p["strategy"]+"</intent>"+"<response>"+p["content"]+"</response>"} if p["role"]=="assistant" 
                                else {"role":p["role"], "content":p["content"]} for p in row["prompt"]]
                    buyer_target_price.append('') 
                    seller_target_price.append('') 
                    description.append('') 
                    item_name.append('')
                    
                serialized_dialogues.append(messages)
                
            logger.info(
                    f"Converted {len(serialized_dialogues)} dialogues "
                    f"(filter: {lower_bound_metric} ≥ {lower_bound}); "
                    f"retention ratio: {len(serialized_dialogues)/len(data_lst):.2f}"
                )
            return serialized_dialogues, buyer_target_price, seller_target_price, description, item_name
        
        logger.info("Converting GRPO Training data ...")
        train_serialized_dialogues, train_buyer_target_price, train_seller_target_price, train_description, train_item_name = serialize(train_lst, lower_bound_metric, use_train_thought = use_train_thought)
        logger.info("Converting GRPO Validation data ...")
        if len(valid_lst) != 0:
            valid_serialized_dialogues, valid_buyer_target_price, valid_seller_target_price, valid_description, valid_item_name = serialize(valid_lst, lower_bound_metric)
        else:
            valid_serialized_dialogues = []
            valid_buyer_target_price, valid_seller_target_price, valid_description, valid_item_name = [],[],[],[]
            
        train_dataset = Dataset.from_dict({"prompt": train_serialized_dialogues, "Price_Buyer":train_buyer_target_price, "Price_Seller":train_seller_target_price, "Title": train_item_name, "Description": train_description })
        valid_dataset = Dataset.from_dict({"prompt": valid_serialized_dialogues, "Price_Buyer":valid_buyer_target_price,"Price_Seller":valid_seller_target_price, "Title":valid_item_name, "Description":valid_description})
        
        grpo_dataset = DatasetDict(
            {
                "train": train_dataset,
                "eval": valid_dataset,
            }
        )
        return grpo_dataset

