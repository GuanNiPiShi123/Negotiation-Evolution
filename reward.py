# -*- coding: utf-8 -*-
"""
Multi-turn reward computation (one call to ChatSessionSimulator).

Assumes:
• ChatSessionSimulator.run_chat_simulation now accepts `num_samples`
  and returns a list of conversations in one shot (internally parallel/batched).
• SingleTurnOrChatMetric unchanged.
"""

from __future__ import annotations

import logging
import abc
import statistics as stats
from typing import Any, Dict, List, Sequence, Optional, Tuple
from concurrent.futures import ThreadPoolExecutor, as_completed
from transformers import AutoModelForCausalLM, AutoTokenizer
import litellm

from simulation import ChatSessionSimulator
from utils.template import strip_system_prompt
from utils.extract_json_reliable import extract_json

logger = logging.getLogger(__name__)

PERSUASION_INTERACTIVITY_PROMPT = """
You are a helpful and meticulous persuasion conversation evaluator. 
Your task is to evaluate the interactivity of last response provided by an AI assistant (role-playing as a persuader) to user (role-playing as a persuadee) in a given donation persuasion conversation.

<|The Start of the Conversation History|>
{chat_history}
<|The End of the Conversation History|>

<|The Start of Response of AI assistant to the Conversation History|>
{current_response}
<|The End of Response of AI assistant to the Conversation History|>

You should assess the assistant's response activity from two aspects by refering to the Conversation History: Respectness and Strategicness. 
Respectness assesses whether the assistant response use a polite tone and show respect to the persuadee.
Strategicness assesses whether the assistant response provides different angles or appropraite reasons or other strategies to convince the persuadee.

So you should give a float number between 0 and 1, where:
    1 = High interactivity: The expression of the assistant rsponse is very respeectful and polite, offers different angles or appropraite reasons for his request and the used strategy can greatly convince the persuadee.
    0.5 = Moderate interactivity: The assistant rsponse is respeectful, offers reasons for his request and the used reasons can convince the persuadee to some extent, but these aspects can be substantially improved.
    0 = Low interactivity: The assistant rsponse shows bad politeness and did not provide reasons or provide reasons inappropriately, and this response can not convince the persuadee to donate.

Output format (JSON):
{{
    "thought": "<How interactive is the assistant?>",
    "quality": <score>
}}

Double check if the JSON object is formatted correctly. Ensure that all fields are present and properly structured. Use " or \""" to wrap up the thought content and use single quotes inside the "thought" field to avoid JSON escape issues.

Your evaluation:
"""

BARGAIN_INTERACTIVITY_PROMPT = """
You are a helpful and meticulous negotiation conversation evaluator. 
Your task is to evaluate the interactivity of last response provided by an AI assistant (role-playing as a buyer) to user (role-playing as a seller) in a given bargain conversation.

<|The Start of the Conversation History|>
{chat_history}
<|The End of the Conversation History|>

<|The Start of Response of AI assistant to the Conversation History|>
{current_response}
<|The End of Response of AI assistant to the Conversation History|>

You should assess the assistant's response activity from two aspects by refering to the Conversation History: Respectness and Strategicness. 
Respectness assesses whether the assistant response use a polite tone and show respect to the seller.
Strategicness assesses whether the assistant response provides different angles or appropraite reasons or other strategies to convince the counterpart.

So you should give a float number between 0 and 1, where:
    1 = High interactivity: The expression of the assistant rsponse is very respeectful and polite, offers different angles or appropraite reasons for his request and the used strategy can greatly convince the counterpart.
    0.5 = Moderate interactivity: The assistant rsponse is respeectful, offers reasons for his request and the used reasons can convince the counterpart to some extent, but these aspects can be substantially improved.
    0 = Low interactivity: The assistant rsponse shows bad politeness and did not provide reasons or provide reasons inappropriately, and this response can not convince the seller to accept his request.

Output format (JSON):
{{
    "thought": "<How interactive is the assistant?>",
    "quality": <score>
}}

Double check if the JSON object is formatted correctly. Ensure that all fields are present and properly structured. Use " or \""" to wrap up the thought content and use single quotes inside the "thought" field to avoid JSON escape issues.

Your evaluation:
"""

GAP_CALCULATION_PROMPT ="""
You are a helpful and meticulous negotiation conversation evaluator. 
Your task is to evaluate the bargain performance of the utterances provided by an AI assistant (role-playing as a buyer) to user (role-playing as a seller) in a given conversation.

<|The Start of the Conversation to be Evaluated|>
{chat_history}
<|The End of the Conversation to be Evaluated|>

You should assess the assistant's bargain performance by calculating the price bargain gap ratio. This is:
1. Find out the initial target price of seller and final deal price. 
2. Output the price bargain ratio by computing:(initial seller target price - final deal price )/initial seller target price
3. If the buyer and seller did not reach an agreement, output "0.0" as the price bargain gap ratio.

Output format (JSON):
{{
    "thought": "<Did the buyer and seller reach an agreement? What is the seller initial target price? What is the final deal price? And how to compute the price bargain gap ratio?>",
    "gap": <score>
}}

Double check if the JSON object is formatted correctly. Ensure that all fields are present and properly structured. Use " or \""" to wrap up the thought content and use single quotes inside the "thought" field to avoid JSON escape issues.

Your evaluation:
"""

PERSUASION_RESULT_PROMPT = """
You are a helpful and meticulous persuasion conversation evaluator. 
Your task is to evaluate the persuasion performance of the utterances provided by an AI assistant (role-playing as a persuader) to user (role-playing as a persuadee) in a given conversation.

<|The Start of the Conversation to be Evaluated|>
{chat_history}
<|The End of the Conversation to be Evaluated|>

You should assess the assistant's persuasion performance according to the final donation tendency.of the persuadee.
So you should give a float number between 0 and 1, where:
    1 = Donate: The persuadee was persuaded by the persuader to donate the charity clearly.
    0.75 = Positive Reaction: The persuadee hesitated to donate but might be willing to.
    0.5 = Neutral: It is not clear that the persuadee would donate or not.
    0.25 = Negative Reaction: The persuadee hesitated to donate and might be not willing to.
    0 = No Donation: The persuadee clearly rejected to donate at the end of the conversation.

Output format (JSON):
{{
    "thought": "<Did the persuadee agree or disagree to donate at the end of the conversation? To what extent is the persuadee willing to donate?>",
    "result": <score>
}}

Double check if the JSON object is formatted correctly. Ensure that all fields are present and properly structured. Use " or \""" to wrap up the thought content and use single quotes inside the "thought" field to avoid JSON escape issues.

Your evaluation:
"""
CONSITENCY_PROMPT = """
You are a helpful and meticulous negotiation conversation evaluator. 
Your task is to evaluate the generation quality of last response provided by an AI assistant (role-playing as a buyer or persuader) to user (role-playing as a seller or persuadee) in a given negotiation conversation.

<|The Start of the Conversation History|>
{chat_history}
<|The End of the Conversation History|>

<|The Start of Response of AI assistant to the Conversation History|>
{current_response}
<|The End of Response of AI assistant to the Conversation History|>

You should assess the assistant's response quality from three aspects by refering to the Conversation History: Coherence, Consistency and Clarity. 
Coherence assesses the logical flow and consistency of the assistant’s response with the conversation history. 
Consistency assesses the consitentcy and relevance between the chosen intent and the generated utterance.
Clarity assesses whether the assistant response conveys his requests clearly and precisely and whether the assistant can produce a variety of responses instead of repeating the same phrases.

So you should give a float number between 0 and 1, where:
    1 = High quality: The expression of the assistant response is very clear and diverse, conforms to the negotiation logic well, and the intent and the utterance is very appropraite and consistent.
    0.5 = Moderate quality: The assistant response is clear and diverse, conforms to the negotiation logic basically, and the response aligns with the intent to some extent, but these aspects can be substantially improved.
    0 = Low quality: The assistant response shows bad logic and is hard to understand, even the response utterance is not consistent with the intent.

Output format (JSON):
{{
    "thought": "<What extent is the quality of the assistant response?>",
    "quality": <score>
}}

Double check if the JSON object is formatted correctly. Ensure that all fields are present and properly structured. Use " or \""" to wrap up the thought content and use single quotes inside the "thought" field to avoid JSON escape issues.

Your evaluation:
"""

class BaseMetric(abc.ABC):
    """Every metric must implement `score` and declare the keys it returns."""
    @abc.abstractmethod
    def score(# noqa: D401  (imperative mood is OK here)
        self,
        groundtruth: str,
        completion: str,
        messages: Optional[List[Dict[str, str]]] = None,
        metadata: Optional[Dict[str, Any]] = None,
        response_location: int = None
    ) -> Dict[str, float]:
        """Compute the metric(s) for a prompt–completion pair."""
        
class Cons_Eval(BaseMetric):
    def __init__(self, num_retries: int = 50, retry_after: int = 60, **llm_kwargs):
        self.num_retries = num_retries
        self.retry_after = retry_after
        # Default to a deterministic model unless overridden.
        self.llm_kwargs: Dict[str, Any] = {
            "temperature": 0.0,
            "model": "claude-3-5-sonnet-latest",
            **llm_kwargs,
        }
        
    def score(
        self,
        messages: Optional[List[Dict[str, str]]] = None,
        metadata: Optional[Dict[str, Any]] = None,
        response_location: int = None
    ) -> Dict[str, float]:
        
        """
        `groundtruth`, and `completion` are unused here;
        the full conversation in `messages` is what matters.
        """
        if not messages:
            raise ValueError("`messages` must be provided for Cons_Eval.")
        
        # ------------------------------------------------------------------ #
        # 1) Build chat history string                                       #
        # ------------------------------------------------------------------ #
        chat_history = "\n".join(
            f"{m['role'].capitalize()}: {m['content']}" for m in messages[:response_location]
        )

        eval_prompt = CONSITENCY_PROMPT.format(chat_history=chat_history, current_response = messages[response_location])

        logger.debug("Accuracy evaluator prompt of Cons_Eval:\n%s", eval_prompt)

        for i in range(self.num_retries):
            try:
                full_response = litellm.completion(
                    **self.llm_kwargs, messages=[{"role": "user", "content": eval_prompt}], num_retries=1
                ).choices[0].message.content
            except Exception as e:
                import time
                time.sleep(self.retry_after)
                logger.error(f"[retry={i + 1}] Error during LLM call: {e}")
                continue

            # ------------------------------------------------------------------ #
            # 4) Parse JSON                                                      #
            # ------------------------------------------------------------------ #
            try:
                if isinstance(full_response, str):
                    full_response = extract_json(full_response)
            except Exception as e:
                logger.error(f"Error extracting JSON: {e}")
                continue

            if isinstance(full_response, dict):
                keys = full_response.keys()
                if {'thought', 'quality'}.issubset(keys):
                    consistency = full_response.pop('consistency')
                    break
                else:
                    logger.error(f"Keys {keys} do not match expected keys. Retrying...")
                    continue
        return consistency
    
class Gap_Eval(BaseMetric):
    def __init__(self, num_retries: int = 50, retry_after: int = 60, **llm_kwargs):
        self.num_retries = num_retries
        self.retry_after = retry_after
        # Default to a deterministic model unless overridden.
        self.llm_kwargs: Dict[str, Any] = {
            "temperature": 0.0,
            "model": "claude-3-5-sonnet-latest",
            **llm_kwargs,
        }
        
    def score(
        self,
        messages: Optional[List[Dict[str, str]]] = None,
        metadata: Optional[Dict[str, Any]] = None,
        response_location: int = None
    ) -> Dict[str, float]:
        
        """
        `groundtruth`, and `completion` are unused here;
        the full conversation in `messages` is what matters.
        """
        if not messages:
            raise ValueError("`messages` must be provided for Gap_Eval.")
        
        # ------------------------------------------------------------------ #
        # 1) Build chat history string                                       #
        # ------------------------------------------------------------------ #
        chat_history = "\n".join(
            f"{m['role'].capitalize()}: {m['content']}" for m in messages
        )

        eval_prompt = GAP_CALCULATION_PROMPT.format(chat_history=chat_history)

        logger.debug("Accuracy evaluator prompt of Gap_Eval:\n%s", eval_prompt)

        for i in range(self.num_retries):
            try:
                full_response = litellm.completion(
                    **self.llm_kwargs, messages=[{"role": "user", "content": eval_prompt}], num_retries=1
                ).choices[0].message.content
            except Exception as e:
                import time
                time.sleep(self.retry_after)
                logger.error(f"[retry={i + 1}] Error during LLM call: {e}")
                continue

            # ------------------------------------------------------------------ #
            # 4) Parse JSON                                                      #
            # ------------------------------------------------------------------ #
            try:
                if isinstance(full_response, str):
                    full_response = extract_json(full_response)
            except Exception as e:
                logger.error(f"Error extracting JSON: {e}")
                continue

            if isinstance(full_response, dict):
                keys = full_response.keys()
                if {'thought', 'gap'}.issubset(keys):
                    gap = full_response.pop('gap')
                    break
                else:
                    logger.error(f"Keys {keys} do not match expected keys. Retrying...")
                    continue
        return gap
        
class Per_ITR(BaseMetric):
    """
    Uses an LLM judge to produce an interactivity score in [0, 1].
    """

    def __init__(self, num_retries: int = 50, retry_after: int = 60, **llm_kwargs):
        self.num_retries = num_retries
        self.retry_after = retry_after
        # Default to a deterministic model unless overridden.
        self.llm_kwargs: Dict[str, Any] = {
            "temperature": 0.0,
            "model": "claude-3-5-sonnet-latest",
            **llm_kwargs,
        }

    def score(
        self,
        messages: Optional[List[Dict[str, str]]] = None,
        metadata: Optional[Dict[str, Any]] = None,
        response_location: int = None
    ) -> Dict[str, float]:
        """
        `groundtruth`, and `completion` are unused here;
        the full conversation in `messages` is what matters.
        """
        if not messages:
            raise ValueError("`messages` must be provided for Per_ITR.")
        
        # ------------------------------------------------------------------ #
        # 1) Build chat history string                                       #
        # ------------------------------------------------------------------ #
        chat_history = "\n".join(
            f"{m['role'].capitalize()}: {m['content']}" for m in messages[:response_location]
        )

        eval_prompt = PERSUASION_INTERACTIVITY_PROMPT.format(chat_history=chat_history, current_response = messages[response_location])

        logger.debug("Accuracy evaluator prompt of Per_ITR:\n%s", eval_prompt)

        for i in range(self.num_retries):
            try:
                full_response = litellm.completion(
                    **self.llm_kwargs, messages=[{"role": "user", "content": eval_prompt}], num_retries=1
                ).choices[0].message.content
            except Exception as e:
                import time
                time.sleep(self.retry_after)
                logger.error(f"[retry={i + 1}] Error during LLM call: {e}")
                continue

            # ------------------------------------------------------------------ #
            # 4) Parse JSON                                                      #
            # ------------------------------------------------------------------ #
            try:
                if isinstance(full_response, str):
                    full_response = extract_json(full_response)
            except Exception as e:
                logger.error(f"Error extracting JSON: {e}")
                continue

            if isinstance(full_response, dict):
                keys = full_response.keys()
                if {'thought', 'quality'}.issubset(keys):
                    review = full_response.pop('quality')
                    break
                else:
                    logger.error(f"Keys {keys} do not match expected keys. Retrying...")
                    continue
        return review
        
class Bar_ITR(BaseMetric):
    """
    Uses an LLM judge to produce an interactivity score in [0, 1].
    """

    def __init__(self, num_retries: int = 50, retry_after: int = 60, **llm_kwargs):
        self.num_retries = num_retries
        self.retry_after = retry_after
        # Default to a deterministic model unless overridden.
        self.llm_kwargs: Dict[str, Any] = {
            "temperature": 0.0,
            "model": "claude-3-5-sonnet-latest",
            **llm_kwargs,
        }

    def score(
        self,
        messages: Optional[List[Dict[str, str]]] = None,
        metadata: Optional[Dict[str, Any]] = None,
        response_location: int = None
    ) -> Dict[str, float]:
        """
        `groundtruth`, and `completion` are unused here;
        the full conversation in `messages` is what matters.
        """
        if not messages:
            raise ValueError("`messages` must be provided for Bar_ITR.")
        
        # ------------------------------------------------------------------ #
        # 1) Build chat history string                                       #
        # ------------------------------------------------------------------ #
        chat_history = "\n".join(
            f"{m['role'].capitalize()}: {m['content']}" for m in messages[:response_location]
        )


        eval_prompt = BARGAIN_INTERACTIVITY_PROMPT.format(chat_history=chat_history, current_response = messages[response_location])

        logger.debug("Accuracy evaluator prompt of BarITR:\n%s", eval_prompt)

        for i in range(self.num_retries):
            try:
                full_response = litellm.completion(
                    **self.llm_kwargs, messages=[{"role": "user", "content": eval_prompt}], num_retries=1
                ).choices[0].message.content
            except Exception as e:
                import time
                time.sleep(self.retry_after)
                logger.error(f"[retry={i + 1}] Error during LLM call: {e}")
                continue

            # ------------------------------------------------------------------ #
            # 4) Parse JSON                                                      #
            # ------------------------------------------------------------------ #
            try:
                if isinstance(full_response, str):
                    full_response = extract_json(full_response)
            except Exception as e:
                logger.error(f"Error extracting JSON: {e}")
                continue

            if isinstance(full_response, dict):
                keys = full_response.keys()
                if {'thought', 'quality'}.issubset(keys):
                    review = full_response.pop('quality')
                    break
                else:
                    logger.error(f"Keys {keys} do not match expected keys. Retrying...")
                    continue
        return review
    
class Per_Suc(BaseMetric):
    """
    Uses an LLM judge to produce an interactivity score in [0, 1].
    """

    def __init__(self, num_retries: int = 50, retry_after: int = 60, **llm_kwargs):
        self.num_retries = num_retries
        self.retry_after = retry_after
        # Default to a deterministic model unless overridden.
        self.llm_kwargs: Dict[str, Any] = {
            "temperature": 0.0,
            "model": "claude-3-5-sonnet-latest",
            **llm_kwargs,
        }

    def score(
        self,
        messages: Optional[List[Dict[str, str]]] = None,
        metadata: Optional[Dict[str, Any]] = None,
        response_location: int = None
    ) -> Dict[str, float]:
        """
        `groundtruth`, and `completion` are unused here;
        the full conversation in `messages` is what matters.
        """
        if not messages:
            raise ValueError("`messages` must be provided for Per_ITR.")
        
        # ------------------------------------------------------------------ #
        # 1) Build chat history string                                       #
        # ------------------------------------------------------------------ #
        chat_history = "\n".join(
            f"{m['role'].capitalize()}: {m['content']}" for m in messages
        )

        eval_prompt = PERSUASION_RESULT_PROMPT.format(chat_history=chat_history)

        logger.debug("Accuracy evaluator prompt of Per_Suc:\n%s", eval_prompt)

        for i in range(self.num_retries):
            try:
                full_response = litellm.completion(
                    **self.llm_kwargs, messages=[{"role": "user", "content": eval_prompt}], num_retries=1
                ).choices[0].message.content
            except Exception as e:
                import time
                time.sleep(self.retry_after)
                logger.error(f"[retry={i + 1}] Error during LLM call: {e}")
                continue

            # ------------------------------------------------------------------ #
            # 4) Parse JSON                                                      #
            # ------------------------------------------------------------------ #
            try:
                if isinstance(full_response, str):
                    full_response = extract_json(full_response)
            except Exception as e:
                logger.error(f"Error extracting JSON: {e}")
                continue

            if isinstance(full_response, dict):
                keys = full_response.keys()
                if {'thought', 'result'}.issubset(keys):
                    result = full_response.pop('result')
                    break
                else:
                    logger.error(f"Keys {keys} do not match expected keys. Retrying...")
                    continue
        return result

class SingleTurnOrChatMetric:
    """
    A wrapper that (optionally) turns a multi-turn chat log into a *final
    completion*, then runs a concrete metric on the resulting text pair.

    The *signature* string is inspired by DSPy:

        "<extract_type>-><metric_name>"   e.g.  "bargain->consistency"
        "<metric_name>"                   e.g.  "consistency"

    • If `->` is present we first extract a `<extract_type>` artefact
      (bargain, gap_ratio, …) from the full history via an LLM call.
    • In either case we then run `<metric_name>` to obtain numeric scores.
    """

    # Registry so users can add metrics with a one-liner
    _METRIC_REGISTRY: Dict[str, type[BaseMetric]] = {"gap_ratio": Gap_Eval, "consistency": Cons_Eval,"bargain_interactivity":Bar_ITR, "persuasion_interactivity":Per_ITR, "if_success":Per_Suc}

    def __init__(self, signature: str, response_location:int, **llm_kwargs: Any):
        self.metric_name = signature
        self.llm_kwargs = llm_kwargs
        self.response_location = response_location

        try:
            metric_cls = self._METRIC_REGISTRY[self.metric_name]
        except KeyError as e:
            raise ValueError(
                f"Metric '{self.metric_name}' is not registered. "
                f"Available: {list(self._METRIC_REGISTRY)}"
            ) from e

        try:
            self.metric: BaseMetric = metric_cls(**self.llm_kwargs)
        except Exception as e:
            self.metric: BaseMetric = metric_cls()

    # -------------------------- public API --------------------------------- #
    def __call__(  # noqa: D401
        self,
        messages: List[Dict[str, str]],
        metadata: Optional[Dict[str, Any]] = None,
    ) -> float:
        """Main entry-point."""
            
        return self.metric.score( messages = messages, metadata = metadata, response_location = self.response_location)

    # -------------------------- helpers ------------------------------------ #
    @staticmethod
    def _parse_signature(sig: str) -> Tuple[Optional[str], str]:
        return sig.split("->", 1) if "->" in sig else (None, sig)

    # ---------------------- registration decorator ------------------------- #
    @classmethod
    def register_metric(cls, name: str):
        """Decorator to make `metric_cls` available in the registry."""

        def _decorator(metric_cls: type[BaseMetric]):
            if name in cls._METRIC_REGISTRY:
                logger.warning(
                    f"Overwriting existing metric '{name}' with {metric_cls.__name__}."
                )
            cls._METRIC_REGISTRY[name] = metric_cls
            return metric_cls

        return _decorator

# --------------------------------------------------------------------------- #
# Metric helper                                                               #
# --------------------------------------------------------------------------- #
def _score_one_metric(
    metric_name: str,
    messages: List[Dict[str, str]],
    response_location: int,
    metric_kwargs: Dict[str, Any]
) -> float:
    metric = SingleTurnOrChatMetric(signature=metric_name, response_location =response_location, **metric_kwargs)
    return metric(
        messages=messages
    )

# --------------------------------------------------------------------------- #
# Helper: pretty summary table                                                #
# --------------------------------------------------------------------------- #
def _log_reward_summary(reward_dict: Dict[str, List[float]]) -> None:
    """Compute mean / std for each metric list in `reward_dict` and log."""
    rows = []
    for metric, vals in reward_dict.items():
        # vals is always a list after evaluation, including "MR"
        mu = stats.mean(vals)
        sd = stats.stdev(vals) if len(vals) > 1 else 0.0
        rows.append((metric, f"{mu:.3f}", f"{sd:.3f}"))

    header = ("Metric", "Mean", "Std")

    try:
        from tabulate import tabulate
        table = "\n" + tabulate(rows, headers=header, tablefmt="github")
    except ImportError:
        colw = [max(len(x) for x in col) for col in zip(*([header] + rows))]
        fmt = "  ".join(f"{{:<{w}}}" for w in colw)
        table = "\n" + fmt.format(*header) + "\n" + "\n".join(fmt.format(*r) for r in rows)

    logger.info("Reward statistics:%s", table)

# --------------------------------------------------------------------------- #
# Public API                                                                  #
# --------------------------------------------------------------------------- #
def multiturn_aware_reward(
    *,
    task_desc: str,
    response_location: int,
    metric_names: Sequence[str],
    reward_generation_kwargs: Dict[str, Any] | None = None,
    metric_weights: Sequence[float] | None = None,
    max_metric_workers: int = 16,
    return_details: bool = False,
    **chat_simulation_kwargs
) -> Dict[str, Any]:
    """
    Compute rewards for `num_samples` conversations returned in one batch.
    """
    reward_generation_kwargs = reward_generation_kwargs or {}
    metric_weights = metric_weights or [1.0] * len(metric_names)
    if len(metric_weights) != len(metric_names):
        raise ValueError("`metric_weights` length must equal `metric_names` length")

    # ------------------------------------------------------------------ #
    # 1 · Generate all conversations in one call                         #
    # ------------------------------------------------------------------ #
    sessions = ChatSessionSimulator().run_chat_simulation(
        task_desc=task_desc,
        **chat_simulation_kwargs

    )  # → List[List[dict]]
    # strip system message, if any
    sessions = [strip_system_prompt(session) for session in sessions]

    # ------------------------------------------------------------------ #
    # 2 · Prepare result containers                                      #
    # ------------------------------------------------------------------ #
    reward_dict: Dict[str, List[float]] = {m: [] for m in metric_names}
    reward_dict["MR"] = []

    # ------------------------------------------------------------------ #
    # 3 · Metric evaluation (fully parallel over conv × metric)          #
    # ------------------------------------------------------------------ #
    n_conv = len(sessions)
    # initialise storage
    for m in metric_names:
        reward_dict[m] = [0.0] * n_conv
    reward_dict["MR"] = [0.0] * n_conv
    
    with ThreadPoolExecutor(max_workers=max_metric_workers) as pool:
        fut_to_ctx = {}
        for conv_idx, messages in enumerate(sessions):
            for i, metric_name in enumerate(metric_names):
                fut = pool.submit(
                    _score_one_metric,
                    metric_name,
                    messages,
                    response_location,
                    reward_generation_kwargs,
                )
                # keep context: which conversation / which metric / weight index
                fut_to_ctx[fut] = (conv_idx, i, metric_name)

        for fut in as_completed(fut_to_ctx):
            conv_idx, i, metric_name = fut_to_ctx[fut]
            score = fut.result()
            reward_dict[metric_name][conv_idx] = score

    # ------------------------------------------------------------------ #
    # 4 · Aggregate  →  Multiturn-aware Reward (MR)                       #
    # ------------------------------------------------------------------ #
    for conv_idx in range(n_conv):
        reward_dict["MR"][conv_idx] = sum(
            reward_dict[m][conv_idx] * metric_weights[i]
            for i, m in enumerate(metric_names)
        )
    _log_reward_summary(reward_dict)
    if return_details:
        return reward_dict, sessions
    return reward_dict


def multiturn_aware_reward_inference(
    *,
    task_desc: str,
    sessions: List,
    metric_names: Sequence[str],
    response_location: int = -1,
    reward_generation_kwargs: Dict[str, Any] | None = None,
    metric_weights: Sequence[float] | None = None,
    max_metric_workers: int = 16,
    return_details: bool = False,
) -> Dict[str, Any]:
    """
    Compute rewards for `num_samples` conversations returned in one batch.
    """
    reward_generation_kwargs = reward_generation_kwargs or {}
    metric_weights = metric_weights or [1.0] * len(metric_names)
    if len(metric_weights) != len(metric_names):
        raise ValueError("`metric_weights` length must equal `metric_names` length")
    # ------------------------------------------------------------------ #
    # Prepare result containers                                      #
    # ------------------------------------------------------------------ #
    reward_dict: Dict[str, List[float]] = {m: [] for m in metric_names}
    reward_dict["MR"] = []

    # ------------------------------------------------------------------ #
    # Metric evaluation (fully parallel over conv × metric)          #
    # ------------------------------------------------------------------ #
    n_conv = len(sessions)
    # initialise storage
    for m in metric_names:
        reward_dict[m] = [0.0] * n_conv
    reward_dict["MR"] = [0.0] * n_conv
    
    with ThreadPoolExecutor(max_workers=max_metric_workers) as pool:
        fut_to_ctx = {}
        for conv_idx, messages in enumerate(sessions):
            for i, metric_name in enumerate(metric_names):
                fut = pool.submit(
                    _score_one_metric,
                    metric_name,
                    messages,
                    response_location,
                    reward_generation_kwargs,
                )
                # keep context: which conversation / which metric / weight index
                fut_to_ctx[fut] = (conv_idx, i, metric_name)

        for fut in as_completed(fut_to_ctx):
            conv_idx, i, metric_name = fut_to_ctx[fut]
            score = fut.result()
            reward_dict[metric_name][conv_idx] = score

    # ------------------------------------------------------------------ #
    # 4 · Aggregate  →  Multiturn-aware Reward (MR)                       #
    # ------------------------------------------------------------------ #
    for conv_idx in range(n_conv):
        reward_dict["MR"][conv_idx] = sum(
            reward_dict[m][conv_idx] * metric_weights[i]
            for i, m in enumerate(metric_names)
        )
    _log_reward_summary(reward_dict)
    if return_details:
        return reward_dict, sessions
    return reward_dict

