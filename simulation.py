# -*- coding: utf-8 -*-
# chat_session_simulator.py
from __future__ import annotations

import os
import copy
import logging
from typing import Dict, List, Optional, Any
from concurrent.futures import ThreadPoolExecutor, as_completed

import torch
from transformers import pipeline, PreTrainedModel, PreTrainedTokenizerBase, TextStreamer
from tqdm import tqdm

from modules import LLMNegotiator, UserSimulator

logger = logging.getLogger(__name__)

NEGOLLM_TERMINATION_SIGNAL = "[TERMINATE CHAT]"

class ChatSessionSimulator:
    """Manages multiple simultaneous chat sessions."""

    # --------------------------------------------------------------------------- #
    # ChatSessionSimulator.run_chat_simulation                                    #
    # --------------------------------------------------------------------------- #
    def run_chat_simulation(
        self,
        *,
        task_desc: str,
        chat_history: List[Dict[str, str]],
        assistant_generation_kwargs: Dict[str, Any],
        user_generation_kwargs: Dict[str, Any],
        Price_Buyer: str, 
        Price_Seller: str, 
        Title: str, 
        Description: str ,
        num_samples: int = 1,                      
        max_new_turns: int = 0,
        local_model: Optional[PreTrainedModel] = None,
        local_tokenizer: Optional[PreTrainedTokenizerBase] = None,
        model_name = "Llama",
        max_workers: int = 8,
        verbose: bool = True,
    ) -> List[List[Dict[str, str]]]:
        """
        Simulate *num_samples* conversations in parallel (internally batched).

        Returns
        -------
        List[List[Dict[str, str]]]
            A list of `num_samples` full chat transcripts.
        """
        # ------------------------------------------------------------------ #
        # 0 · Validation / defaults                                          #
        # ------------------------------------------------------------------ 
        self._validate_session_inputs(
            task_desc,
            max_new_turns,
            local_model,
            local_tokenizer,
            assistant_generation_kwargs,
            user_generation_kwargs
        )

        # ------------------------------------------------------------------ #
        # 1 · Per-conversation state                                         #
        # ------------------------------------------------------------------ #
        sessions: List[List[Dict[str, str]]] = [
            copy.deepcopy(chat_history or []) for _ in range(num_samples)
        ]

        current_roles = [self._determine_starting_role(hist) for hist in sessions]

        user_sims = [
            UserSimulator(
                task_desc=task_desc,
                **user_generation_kwargs,
            )
            for _ in range(num_samples)
        ]

        # ------------------------------------------------------------------ #
        # 2 · Conversation loop (respects max_new_turns budget)              #
        # ------------------------------------------------------------------ #
        msg_budget = [max_new_turns for _ in range(num_samples)]  # ← NEW
        active: set[int] = {i for i, b in enumerate(msg_budget) if b > 0}

        pbar = tqdm(total=max_new_turns, desc="Simulating chat", disable=not verbose)

        while active:
            # ---------- USER TURNS ---------------------------------------- #
            user_idx = [i for i in active if current_roles[i] == "user"]
            if user_idx:
                with ThreadPoolExecutor(max_workers=max_workers) as pool:
                    fut_to_i = {pool.submit(user_sims[i], sessions[i], Title, Description, Price_Seller): i for i in user_idx}
                    for fut in as_completed(fut_to_i):
                        i = fut_to_i[fut]
                        resp = fut.result()
                        self._log_response(f"user (Turn {len(sessions[i])})", resp)
                        sessions[i].append({"role": "user", "content": resp})
                        
                        msg_budget[i] -= 1

                        # early exit checks
                        if msg_budget[i] == 0 or self._should_terminate_conversation(resp):
                            current_roles[i] = "terminated"
                            active.discard(i)
                        else:
                            current_roles[i] = "assistant"
                    pbar.update(1)

            if not active:  # all dialogues exhausted their budget / terminated
                break

            # ---------- ASSISTANT TURNS ----------------------------------- #
            asst_idx = [i for i in active if current_roles[i] == "assistant"]
            if not asst_idx:
                continue

            # --- generate assistant replies (batched or threaded) --- #
            if local_model is None:
                with ThreadPoolExecutor(max_workers=max_workers) as pool:
                    fut_to_i = {}
                    for i in asst_idx:
                        negoLLM_i = LLMNegotiator(task_desc = task_desc, **assistant_generation_kwargs)
                        fut = pool.submit(negoLLM_i, sessions[i], Title, Description, Price_Buyer)
                        fut_to_i[fut] = i

                    responses = {fut_to_i[f]: f.result() for f in fut_to_i}
            else:
                batch_sess = [sessions[i] for i in asst_idx]
                outs = self._batch_generate_with_huggingface(
                    batch_sess,
                    local_model,
                    local_tokenizer,
                    assistant_generation_kwargs,
                    model_name = model_name
                )
                responses = {g: r for g, r in zip(asst_idx, outs)}

            # --- post-process assistant replies --- #
            for i, resp in responses.items():
                self._log_response(f"assistant (Turn {len(sessions[i])})", resp)
                sessions[i].append({"role": "assistant", "content": resp})

                msg_budget[i] -= 1

                if msg_budget[i] == 0 or self._should_terminate_conversation(resp):
                    current_roles[i] = "terminated"
                    active.discard(i)
                else:
                    current_roles[i] = "user"
            pbar.update(1)

        pbar.close()
        return sessions

    # ------------------------------------------------------------------ #
    # Batch generators                                                   #
    # ------------------------------------------------------------------ #
    def _batch_generate_with_huggingface(
        self,
        batch_messages: List[List[Dict[str, str]]],
        local_model,
        local_tokenizer,
        generation_kwargs: Dict[str, Any],
        model_name
    ) -> List[str]:
        """Batched HF generation (one forward pass)."""
        torch.cuda.empty_cache()
        local_tokenizer.padding_side = "left"
        local_tokenizer.pad_token = local_tokenizer.eos_token

        generator = pipeline(
            "text-generation",
            model=local_model,
            tokenizer=local_tokenizer,
            model_kwargs={"torch_dtype": "auto"},
            device_map="auto",
        )

        generation_kwargs = copy.deepcopy(generation_kwargs)
        max_new = generation_kwargs.pop("max_tokens", 512)
        generation_kwargs.pop("model", None)  # not needed for HF pipeline
        results = []
        if "gpt" not in model_name:
            prompts = [msgs for msgs in batch_messages]  # HF pipeline accepts list
            outputs = generator(
                prompts,
                max_new_tokens=max_new,
                **generation_kwargs,
            )

            # Extract only the newly generated part for each item

            for prompt_msgs, out in zip(prompts, outputs):
                if isinstance(out, list):
                    out = out[0]  # HF pipeline returns list of dicts
                full_text = out["generated_text"]

                if isinstance(prompt_msgs, str):
                    results.append(full_text[len(prompt_msgs) :])
                else:
                    results.append(full_text[-1]["content"])
            torch.cuda.empty_cache()
        else:
            for message in batch_messages:
                inputs = local_tokenizer.apply_chat_template(
                    message,
                    add_generation_prompt = True,
                    return_tensors = "pt",
                    return_dict = True,
                    reasoning_effort = "medium",
                    ).to("cuda")

                result = local_model.generate(**inputs, max_new_tokens = 2048, streamer = TextStreamer(local_tokenizer))
            results.append(result)
        return results

    def _validate_session_inputs(
        self,
        task_desc: str,
        max_new_turns: int,
        local_model: Optional[PreTrainedModel] = None,
        local_tokenizer: Optional[PreTrainedTokenizerBase] = None,
        assistant_generation_kwargs: Optional[Dict[str, Any]] = None,
        user_generation_kwargs: Optional[Dict[str, Any]] = None
    ) -> None:
        """
        Sanity-check all arguments before starting a chat session.

        Raises
        ------
        ValueError
            If any invariant required by the session runner is violated.
        """
        if not isinstance(task_desc, str) or not task_desc.strip():
            raise ValueError("`task_desc` must be a non-empty string.")

        if not isinstance(max_new_turns, int) or max_new_turns < 0:
            raise ValueError("`max_new_turns` must be an integer ≥ 0.")

        if (local_model is None) ^ (local_tokenizer is None):
            raise ValueError(
                "Provide *both* `local_model` and `local_tokenizer`, or neither."
            )

        if assistant_generation_kwargs.get("model") is None:
            raise ValueError(
                "`assistant_generation_kwargs` must include a 'model' key."
            )
        if user_generation_kwargs.get("model") is None:
            raise ValueError(
                "`user_generation_kwargs` must include a 'model' key."
            )

    def _determine_starting_role(self, chat_history: List[Dict[str, str]]) -> str:
        """Determine which role should start the conversation."""
        if chat_history and chat_history[-1]['role'] == 'user':
            return 'assistant'
        return 'user'
    
    def _should_terminate_conversation(self, response: str) -> bool:
        """
        Check if the response contains a termination signal.
        
        Args:
            response: The response text to check
            
        Returns:
            True if conversation should terminate, False otherwise
        """
        try:
            return NEGOLLM_TERMINATION_SIGNAL in response
        except Exception as e:
            logger.error(f"Error checking for chat termination: {e}")
            return False
    
    def _log_response(self, role: str, response: str) -> None:
        """Log the response if verbose mode is enabled and on main process."""
        logger.info(f"[rank {os.environ.get('RANK', 0)}]{role.capitalize()}: {response}")