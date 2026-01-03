# -*- coding: utf-8 -*-
import numpy as np
import copy
import math
import re
import random
import logging
import torch

from transformers import pipeline, PreTrainedModel, PreTrainedTokenizerBase, TextStreamer

from modules import LLMNegotiator, UserSimulator, NEGOLLM_TERMINATION_SIGNAL
from reward import multiturn_aware_reward, Gap_Eval, Per_Suc
from simulation import ChatSessionSimulator

from utils.template import strip_system_prompt

logger = logging.getLogger(__name__)

def model_inference(model_name, model, tok, state, assistant_generation_kwargs):
    if "gpt" not in model_name:
        generator = pipeline(
            "text-generation",
            model = model,
            tokenizer = tok,
            model_kwargs={"torch_dtype": "auto"},
            device_map="auto",
        )
    
        generation_kwargs = copy.deepcopy(assistant_generation_kwargs)
        max_new = generation_kwargs.pop("max_tokens", 512)
        generation_kwargs.pop("model", None)  # not needed for HF pipeline
        generation_kwargs.pop("api_key", None)
        prompts = [state]  # HF pipeline accepts list
        outputs = generator(
            prompts,
            max_new_tokens=max_new,
            **generation_kwargs,
        )
    
        results = []
        for prompt_msgs, out in zip(prompts, outputs):
            if isinstance(out, list):
                out = out[0]  # HF pipeline returns list of dicts
            full_text = out["generated_text"]

            if isinstance(prompt_msgs, str):
                results.append(full_text[len(prompt_msgs) :])
            else:
                results.append(full_text[-1]["content"])
        torch.cuda.empty_cache()
        result = results[0]
        
    else:
        inputs = tok.apply_chat_template(
            state,
            add_generation_prompt = True,
            return_tensors = "pt",
            return_dict = True,
            reasoning_effort = "medium",
        ).to("cuda")
        result = model.generate(**inputs, max_new_tokens = 2048, streamer = TextStreamer(tok))
        torch.cuda.empty_cache()
    return result.strip()

class OpenLoopMCTS():
    def __init__(self, 
                 max_realizations, 
                 task_desc, 
                 Q_0, cpuct, 
                 num_samples, 
                 evolution_ratio, 
                 Title, 
                 Description, 
                 Price_Buyer,
                 Price_Seller,
                 assistant_generation_kwargs, 
                 user_generation_kwargs, 
                 reward_generation_kwargs,
                 inference_model,
                 tok,
                 model_name,
                 test=False) -> None:
        self.P: dict = {}
        self.Ns: dict = {}
        self.Nsa: dict = {}
        self.Q: dict = {}
        self.realizations: dict = {}
        self.realizations_Vs: dict = {}
        self.realizations_Ns: dict = {}
        self.valid_moves: dict = {}
        
        self.evolution_ratio = evolution_ratio
        self.Q_0 = Q_0
        self.cpuct = cpuct
        self.num_samples = num_samples #8
        
        self.title = Title
        self.description = Description
        self.price_buyer = Price_Buyer
        self.price_seller = Price_Seller
        self.assistant_generation_kwargs = assistant_generation_kwargs
        self.reward_generation_kwargs = reward_generation_kwargs
        self.user_generation_kwargs = user_generation_kwargs
        self.inference_model = inference_model
        self.tok = tok
        self.model_name = model_name
        self.test = test
        
        self.smoothing = 1.0
        self.task_desc = task_desc
        if "bargain" in task_desc:
            self.dialog_acts = ["Greetings", "Ask a question", "Answer a question", "Propose the initial price",
                                "Propose a counter price", "Use comparatives", "Confirm information", "Affirm confirmation", 
                                "Deny confirmation", "Agree with the proposal", "Disagree with a proposal"]
        else:
            self.dialog_acts = ["Greetings", "Logical Appeal", "Emotion Appeal", "Credibility Appeal", "Foot in the Door", "Self-Modeling",
                                "Personal Story", "Donation Information", "Source-related Inquiry", "Task-related Inquiry", "Personal-related Inquiry"]
        return

    def _to_string_rep(self, state):
        strategy_pattern = r'<intent>(.*?)</intent>'
        das = []
        for conv in state:
            if conv["role"] == "assistant":
                match = re.search(strategy_pattern, conv["content"])
                if match:
                    das.append(match.group(1))
                else:
                    das.append("Unknown")
        return "__".join(das)

    def _init_node(self, state, msg_budget):
        hashable_state = self._to_string_rep(state)
        prior, v, act_utter_pairs = self.predict(state, msg_budget)
        
        allowed_actions = {}
        dialog_acts = []
        for key, values in prior.items():
            dialog_acts.append(key)
            if key not in self.dialog_acts:
                allowed_actions[key]={"allow":1, "explore":1, "utterance": act_utter_pairs[key]} if key != "Greetings" else {"allow":0, "explore":1, "utterance": act_utter_pairs[key]}
            else:
                allowed_actions[key]={"allow":1, "explore":0, "utterance": act_utter_pairs[key]} if key != "Greetings" else {"allow":0, "explore":0, "utterance": act_utter_pairs[key]}
        
        self.valid_moves[hashable_state] = allowed_actions

        self.Ns[hashable_state] = 0
        self.Nsa[hashable_state] = {action: 0 for action in self.valid_moves[hashable_state].keys()}
        self.Q[hashable_state] = {action: self.Q_0 for action in self.valid_moves[hashable_state].keys()}
        self.realizations[hashable_state] = [copy.deepcopy(state)]
            
        self.P[hashable_state]["acts"] = dialog_acts
        self.P[hashable_state]["prob"] = np.array([prior[act]*allowed_actions[act]["allow"] for act in dialog_acts])
        
        # renormalize
        if np.sum(self.P[hashable_state]["prob"]) == 0:
            digital_allowed_actions = np.array([allowed_actions[act]["allow"] for act in dialog_acts])
            self.P[hashable_state]["prob"] = digital_allowed_actions / np.sum(digital_allowed_actions)
            logger.warning("This should never happen")
        else:
            self.P[hashable_state]["prob"] /= np.sum(self.P[hashable_state]["prob"])
        return v

    def _sample_realization(self, hashable_state):
        rand_i = np.random.randint(len(self.realizations[hashable_state]))
        return self.realizations[hashable_state][rand_i]

    def _add_new_realizations(self, state):
        hashable_state = self._to_string_rep(state)
        if hashable_state not in self.realizations:
            self.realizations[hashable_state] = []
        if state in self.realizations[hashable_state]:
            return
        
        self.realizations[hashable_state].append(copy.deepcopy(state))
        if len(self.realizations[hashable_state]) > self.max_realizations:
            # should never happen
            logger.warning(f"len(self.realizations[hashable_state])={len(self.realizations[hashable_state])}")
            self.realizations[hashable_state].pop(0)
        return

    def _get_next_state(self, state, best_action, action_meta):
        prefetch_state = self._to_string_rep(state) + "__" + best_action
        if prefetch_state in self.realizations and len(self.realizations[prefetch_state]) == self.max_realizations:
            # use the cached realization
            return self._sample_realization(prefetch_state)
        
        intent_pattern = r'<intent>(.*?)</intent>'
        if action_meta["explore"] ==1:
            next_state = copy.deepcopy(state)+[{"role":"assistant", "content":action_meta["utterance"]}]
        else:
            best_flag = False
            for i in range(3):
                resp = model_inference(self.model_name, self.inference_model, self.tok, state, self.assistant_generation_kwargs)
                intent_match = re.search(intent_pattern, resp, re.DOTALL)
                if intent_match and intent_match.group(1).strip() in best_action:
                    best_flag=True
                    break
            if best_flag:
                next_state = copy.deepcopy(state) + [{"role":"assistant", "content":resp}]
            else:
                next_state = copy.deepcopy(state)+[{"role":"assistant", "content":action_meta["utterance"]}]

        user_simulator = UserSimulator(task_desc=self.task_desc, **self.user_generation_kwargs)
        user_response = user_simulator(next_state, self.title, self.description, self.price_seller)
        next_state = next_state + [{"role":"user", "content":user_response}]
        return next_state
    
    def _update_realizations_Vs(self, state, v: float):
        hashable_state = self._to_string_rep(state)
        if hashable_state not in self.realizations_Vs:
            self.realizations_Vs[hashable_state] = {}
            self.realizations_Ns[hashable_state] = {}
        sys_utt = state[-2]["content"]

        if sys_utt not in self.realizations_Vs[hashable_state]:
            self.realizations_Vs[hashable_state][sys_utt] = 0
            self.realizations_Ns[hashable_state][sys_utt] = 0
        # update
        self.realizations_Ns[hashable_state][sys_utt] += 1
        self.realizations_Vs[hashable_state][sys_utt] += (v - self.realizations_Vs[hashable_state][sys_utt]) / self.realizations_Ns[hashable_state][sys_utt]
        return

    def search(self, state, msg_budget):
        hashable_state = self._to_string_rep(state)
        
        assert state[-1]["role"] == "user"
        
        if NEGOLLM_TERMINATION_SIGNAL in state[-1]["content"]:
            if "bargain" in self.task_desc:
                metric = Gap_Eval(**self.reward_generation_kwargs)
                result = metric.score(messages = strip_system_prompt(state), metadata = None, response_location = 0)
                return -1.0+ 2* float(result)
            else:
                metric = Per_Suc(**self.reward_generation_kwargs)
                result = metric.score(messages = strip_system_prompt(state), metadata = None, response_location = 0)
                return -1.0+ 2* float(result)
            
        if hashable_state not in self.P:
            # selected leaf node, expand it
            # first visit V because v is only evaluated once for a hashable_state
            v = self._init_node(state, msg_budget)
            return v
        else:
            # add only when it is new
            self._add_new_realizations(state)

        # existing, continue selection
        # go next state by picking best according to U(s,a)
        best_uct = -float('inf')
        best_action = "Unknown"
        for a in self.valid_moves[hashable_state].keys():
            if self.valid_moves[hashable_state][a]["allow"] == 1:
                Ns = self.Ns[hashable_state]
                if Ns == 0:
                    Ns = 1e-8
                # a variant of PUCT
            uct = self.Q[hashable_state][a] + self.cpuct * self.P[hashable_state][self.P[hashable_state]["acts"].index(a)] * math.sqrt(Ns) / (1 + self.Nsa[hashable_state][a])
            
            if uct > best_uct:
                best_uct = uct
                best_action = a
                
        # transition. For open loop, first sample from an existing realization
        state = self._sample_realization(hashable_state)
        next_state = self._get_next_state(state, best_action, self.valid_moves[hashable_state][a])
        
        # 1. if not leaf, continue traversing, and state=s will get the value from the leaf node
        # 2. if leaf, we will expand it and return the value for backpropagation
        v = self.search(next_state, msg_budget-2)

        # update stats
        # add in new estimate and average
        self.Q[hashable_state][best_action] = (self.Nsa[hashable_state][best_action] * self.Q[hashable_state][best_action] + v) / (self.Nsa[hashable_state][best_action] + 1)
        
        self.Ns[hashable_state] += 1
        self.Nsa[hashable_state][best_action] += 1

        # update v to realizations for NLG at inference
        self._update_realizations_Vs(next_state, v)
        # now we are single player, hence just v instead of -v
        return v
    
    def get_best_realization(self, state, action):
        prefetch_state = self._to_string_rep(state) + "__" + action
        if prefetch_state not in self.realizations_Vs:
            #raise Exception("querying a state that has no realizations sampled before")
            hashable_state = self._to_string_rep(state)
            valid_moves = [act for act, act_meta in self.valid_moves[hashable_state].items() if act_meta["allow"] == 1]
            for act in valid_moves:
                prefetch_state = hashable_state + "__" + act
                if prefetch_state in self.realizations_Vs:
                    break
        # get the counts for all moves
        # convert to prob
        curr_best_v = -float('inf')
        curr_best_realization = None
        for sys_utt, v in self.realizations_Vs[prefetch_state].items():
            if v > curr_best_v:
                curr_best_v = v
                curr_best_realization = sys_utt
        return curr_best_realization
    
    def get_action_prob(self, state, msg_budget):
        hashable_state = self._to_string_rep(state)
        if hashable_state not in self.Ns:
            # selected leaf node, expand
            logging.warn("querying a state that has not been visited")
            self._init_node(state, msg_budget)
        # get the counts for all moves
        # convert to prob
        valid_moves = [act for act, act_meta in self.valid_moves[hashable_state].items() if act_meta["allow"] == 1]
        prob = np.zeros(len(valid_moves))
        for i, a in enumerate(valid_moves):
            prob[i] = self.Nsa[hashable_state][a]
        prob /= prob.sum()
        policy_next_da = valid_moves[np.argmax(prob)]
        return policy_next_da
    
    def predict(self, state, msg_budget = 0):
        # test k times and compute prob. See num_return_sequences in the API
        # the value would be our objective function
        sampled_das, act_utter_pairs = [], {}
        intent_pattern = r'<intent>(.*?)</intent>'
        
        if_evolution = random.random()
        if if_evolution < self.evolution_ratio:
            negotiator = LLMNegotiator(task_desc = self.task_desc, **self.assistant_generation_kwargs)
            for i in range(self.num_samples):
                response = negotiator(messages = state, Title = self.title, Description = self.description, Price_Buyer = self.price_buyer)
                #think_pattern = r'<think>(.*?)</think>'
                
                #response_pattern = r'<response>(.*?)</response>'
                #think_match = re.search(response_pattern, response, re.DOTALL)
                intent_match = re.search(intent_pattern, response, re.DOTALL)
                #response_match = re.search(response_pattern, response, re.DOTALL)
                if intent_match:
                    sampled_das.append(intent_match.group(1).strip())
                    if intent_match.group(1).strip() not in act_utter_pairs:
                        act_utter_pairs[intent_match.group(1).strip()] = response 
        else:
            for i in range(self.num_samples):
                response = model_inference(self.model_name, self.inference_model, self.tok, state, self.assistant_generation_kwargs)
                intent_match = re.search(intent_pattern, response, re.DOTALL)
                if intent_match:
                    sampled_das.append(intent_match.group(1).strip())
                    if intent_match.group(1).strip() not in act_utter_pairs:
                        act_utter_pairs[intent_match.group(1).strip()] = response
        print("_________________")
        print(if_evolution)
        print(intent_match)
        print("_________________")
        prob = {act:self.smoothing/(len(self.dialog_acts)+len(sampled_das)) for act in self.dialog_acts}
        logger.debug(f"sampled das: {sampled_das}")

        for da in sampled_das:
            if da not in prob:
                prob[da] = self.smoothing/(len(self.dialog_acts)+len(sampled_das))+1.0/(len(self.dialog_acts)+len(sampled_das))
            else:
                prob[da] += 1.0/(len(self.dialog_acts)+len(sampled_das))

        if "bargain" in self.task_desc:
            if msg_budget >= 2:
                if self.test:
                    local_model = self.inference_model
                    local_tokenizer = self.tok
                else:
                    local_model = None
                    local_tokenizer = None
                reward_info = multiturn_aware_reward(task_desc = self.task_desc, 
                                                     response_location = -1, 
                                                     metric_names = ["gap_ratio"],
                                                     reward_generation_kwargs = self.reward_generation_kwargs,
                                                     metric_weights = [1.0],
                                                     chat_history = state,
                                                     user_generation_kwargs = self.user_generation_kwargs,
                                                     assistant_generation_kwargs = self.assistant_generation_kwargs,
                                                     num_samples = 2,
                                                     max_new_turns = msg_budget,
                                                     Price_Buyer= self.price_buyer, 
                                                     Price_Seller= self.price_seller, 
                                                     Title = self.title, 
                                                     Description = self.description,
                                                     local_model = local_model,
                                                     local_tokenizer = local_tokenizer
                                                     )
                v = -1.0 + 2* np.mean(reward_info["MR"])
            else:
                v= -0.5
        else:
            if msg_budget >= 1 and not self.test:
                if self.test:
                    local_model = self.inference_model
                    local_local_tokenizer = self.tok
                else:
                    local_model = None
                    local_tokenizer = None
                reward_info = multiturn_aware_reward(task_desc = self.task_desc, 
                                                     response_location = -1, 
                                                     metric_names = ["if_success"],
                                                     reward_generation_kwargs= self.reward_generation_kwargs,
                                                     metric_weights = [1.0],
                                                     chat_history = state + [{"role":"assistant","content":"Would you be interested in donating to Save the Children?"}],
                                                     user_generation_kwargs = self.user_generation_kwargs,
                                                     assistant_generation_kwargs = self.assistant_generation_kwargs,
                                                     num_samples = 2,
                                                     max_new_turns = 1,
                                                     Price_Buyer= "", 
                                                     Price_Seller= "", 
                                                     Title = "", 
                                                     Description = "",
                                                     local_model = local_model,
                                                     local_tokenizer = local_tokenizer
                                                     )
                v = -1.0 + 2* np.mean(reward_info["MR"])
            else:
                v= -0.5
        return prob, v, act_utter_pairs

