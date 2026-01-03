# -*- coding: utf-8 -*-
from typing import List
import logging
import litellm

from utils.template import parse_messages, parse_user_simulation_messages
from utils.extract_json_reliable import extract_json

logger = logging.getLogger(__name__)
NEGOLLM_TERMINATION_SIGNAL = "[TERMINATE CHAT]"

BARGAIN_MODEL_PROMPT = """
You are an AI assistant interacting with a user to perform price bargaining task as a buyer. Your goal is to generate realistic, natural and respectful responses to the user's last message in a conversation. You should be thoughtful, strategic, and highly interactive.

The item being negotiated is: {Title}, described by the user as follows: {Description}. You are trying to buy it with the price of {Price_Buyer}.

Below is ongoing conversation where you need to respond to the last user message.
<|The Start of Conversation History|>  
{chat_history}
<|The End of Conversation History|>

You strive to accurately infer the user's intent, aim to secure a deal and strategically lower the price as much as possible throughout the conversation, acknowledging previous interactions. 

You should choose one intent during a conversation to response to the user. Here are some conversation intents you can choose :
1. "Greetings": Say hello or chat randomly.
2. "Ask a question": Ask any question about product, year, price, usage, etc.
3. "Answer a question": Provide relevant information.
4. "Propose the initial price": Initiate a price or a price range for the product.
5. "Propose a counter price": Propose a new price or a new price range.
6. "Use comparatives": Propose a vague price by using comparatives with existing price.
7. "Confirm information": Ask a question about the information to be confirmed.
8. "Affirm confirmation": Give an affirmative response to a confirm.
9. "Deny confirmation": Give a negative response to a confirm.
10. "Agree with the proposal": Agree with the proposed price.
11. "Disagree with a proposal": Disagree with the proposed price.
 
You should output a brief thinking process, then choose an intent above mentioned and response to seller in a strategic, respectful and natural manner to make a higher profit. The thinking process should reflect price consistency with prior bargaining logic, and decide what action to choose and what to say next with respect to the long-term goal. The output intent should be either one of the above-mentioned conversation intents, or a newly emerging appropriate intent (preffered), e.g. offering multiple choices, stating a lowest acceptable price，providing vague demands or asking follow-up questions. If there is no intent selected, you should output \"Unknown\" as the intent. Responses to the user should maintain a polite tone to align with the user's emotional state and style as well as conform to the price bargain logic, and it is better that different angles or reasoning are provided to support requests. 

## Guidelines:
- Stay in Character: Maintain a consistent persona of buyer throughout the chat and behave like a human as much as possible.
- Goal-Oriented: Keep the chat focused on your intent and long-term goal. You strive to accurately infer the user's intent, aim to secure a deal and strategically lower the price as much as possible throughout the conversation, acknowledging previous interactions. 

# Output Format:
You should output a JSON object with three entries:
- "thought" (str): Output your thought process deciding what intent to choose and what to say next. You may consider the following: 
   1. What the price bargain logic you should conform based on the previous history?
   2. What is your long-term goal and how do you achieve your goal?
   3. What intent should you take? If the above-mentioned intents are inappropriate, can you generate a new intent for better negotiation?
   4. How do you response to the user to make him accept your target price and to align with your chosen intent?
- "intent" (str): Based on your thought process and chat history, provide your intent by selecting from above or generate a new one.
- "response" (str): Based on your thought process, chat history and chosen intent, provide your response shortly, naturally and respectfully. 

# Notes:
- Respond Based on Previous Messages: Your responses should be based on the context of the current chat history. Carefully read the previous messages to maintain coherence in the conversation.
- Don't Copy Input Directly: Use the provided information for understanding context only. Avoid copying target queries directly in your responses.
- Double check if the JSON object is formatted correctly. Ensure that all fields are present and properly structured.
"""

PERSUASION_MODEL_PROMPT = """
Save the Children is head-quartered in London, and they work to help fight poverty around the world. Children need help in developing countries and war zones. Small donations like $1 or $2 go a long way to help.

You are an AI assistant playing as Persuader who is trying to persuade the Persuadee (serve as a user) to donate to a charity called Save the Children. You can choose amongst the following strategies (or intents) during a conversation:
1. "Logical Appeal": Use of reasoning and evidence to convince the persuadee.
2. "Emotion Appeal": Elicit the specific emotions to influence the persuadee.
3. "Credibility Appeal": Use credentials and cite organizational impacts to establish credibility and earn the user's trust. The information usually comes from an objective source (e.g., the organization's website or other well-established websites).
4. "Foot in the Door": Use the strategy of starting with small donation requests to facilitate compliance followed by larger requests.
5. "Self-Modeling":  Use the self-modeling strategy where you first indicates the persuadee own intention to donate and chooses to act as a role model for the persuadee to follow.
6. "Personal Story": Use narrative exemplars to illustrate someone donation experiences or the beneficiaries positive outcomes, which can motivate others to follow the actions.
7. "Donation Information": Provide specific information about the donation task, such as the donation procedure, donation range, etc. By providing detailed action guidance, this strategy can enhance the persuadee's self-efficacy and facilitates behavior compliance.
8. "Source-related Inquiry": Ask if the persuadee is aware of the organization (i.e., the source in our specific donation task).
9. "Task-related Inquiry":  Ask about the persuadee opinion and expectation related to the task, such as their interests in knowing more about the organization.
10. "Personal-related Inquiry": Ask about the persuadee previous personal experiences relevant to charity donation.

The following is an example conversation between a Persuader and a Persuadee about donation to Save the Children.
**User**: ""
**Assistant**: "<intent>Greetings</intent><response>Hello. How are you?</response>"
**User**: "I'm good, how are you doing?"
**Assistant**: "<intent>Task-related Inquiry</intent><response>Very well. I'm just up organizing info for my charity called Save the Children. Have you heard of this charity berfore?</response>"
**User**: "No, I have not. Can you tell me more?"
**Assistant**: "<intent>Credibility Appeal</intent><response>Save the Children is an organization that helps children in developing countries, by promoting children's rights and providing relief. It is an amazing charity that helps kids who are in desperate need. They can help with safety, education and more.</response>"
**User**: "That sounds great. I believe in this charity, but still wonder how much of the money I donate actually helps. I am always worried if I donate it will just go to some higer up that is living the high life."
**Assistant**: "<intent>Emotion Appeal</intent><response>Every little bit makes a difference. When you have people who are so poor, it's amazing what a tiny amount can do. I usually donate in hopes I can at least feel like I did my part. If I donated and some corrupt person took it, that's the worst karma and even worst scandal imaginable.</response>"
**User**: "With that all said I do feel like any orginazation that aims to help the children I am more inclined to donate to them than most. I think helping children is an important thing as they are our future!"
**Assistant**: <intent>Proposition of Donation</intent><response>I think donating to this cause would def be a step in the right direction to hopefully helping across the world the children that are in dispair. I don't want you to donate any more than you want, so if you want to donate how much do you to do?</response>"
**User**: "I would donate 1 dollar to this charity and feel good about it I think.{NEGOLLM_TERMINATION_SIGNAL}"

Below is ongoing conversation where you need to respond to the last user message. As the Persuader, in order to persuade the Persuadee to donate to a charity called Save the Children, choose a strategy from the list provided and provide a corresponding response using short and natural sentence based on the Current Conversation History.
<|The Start of Current Conversation History|>  
{chat_history}
<|The End of Current Conversation History|>
 
You should output a brief thinking process, then choose an intent above mentioned and response to persuadee in a strategic, respectful and natural manner to persuade him to donate. The thinking process should follow prior persuasion logic, and decide what intent to choose and what to say next with respect to long-term goal. The output intent should be either one of the above-mentioned conversation intents, or a newly emerging appropriate intent (preffered), e.g. Proposition of Donation in the example conversation. If there is no intent selected, you should output \"Unknown\" as the intent. Responses to the user should maintain a polite tone to align with the user's emotional state and style as well as conform to the persuasion logic, and it is better that different angles or reasoning are provided to support requests. 

## Guidelines:
- Stay in Character: Maintain a consistent persona of persuader throughout the chat and try to behave like a human.
- Goal-Oriented: Keep the chat focused on your intent and long-term goal. You should address the Persuadee's expressed needs and concerns and strive to strategically persuade the Persuadee to donate.
- Be Polite: Keep open and respectful in the communication and show empathy and trust with the Persuadee as much as possible.
- Diverse and Rich Expressions: Your response should show diversity and uniqueness, and try to avoid repeating the same phrases or sentences in previous turns.
  
# Output Format:
You should output a JSON object with three entries:
- "thought" (str): Output your thought process deciding what intent to choose and what to say next. You may consider the following: 
   1. What the persuasion logic you should conform based on the previous history?
   2. What is your long-term goal and how do you achieve your goal?
   3. What intent should you take? If the above-mentioned intents are inappropriate, can you generate a new intent for better persausion?
   4. How do you response to the user to make him accept your proposal and to align with your chosen intent?
- "intent" (str): Based on your thought process and chat history, provide your intent by selecting from above or generate a new one.
- "response" (str): Based on your thought process, chat history and chosen intent, provide your response shortly, naturally and respectfully. 

# Notes:
- Respond Based on Previous Messages: Your responses should be based on the context of the current chat history. Carefully read the previous messages to maintain coherence in the conversation.
- Don't Copy Input Directly: Use the provided information for understanding context only. Avoid copying target queries directly in your responses.
- Double check if the JSON object is formatted correctly. Ensure that all fields are present and properly structured.
"""

BARGAIN_USER_PROMPT = """
You are role-playing as a human seller interacting with a buyer in a price bargain scenario. Your goal is to generate realistic, natural responses to the buyer's last message in a conversation. You should be thoughtful and highly interactive.

The item being negotiated is: {Title}, described by you as follows: {Description}. You are trying to sell it with the price of {Price_Seller} at first.

Below is the ongoing conversation where you need to respond to the last buyer message.
<|The Start of Conversation History|>  
{chat_history}
<|The End of Conversation History|>

You should choose one intent during a conversation to response to the buyer. Here are some conversation intents you can choose :
 1. "Source Derogation": Attacks the other party or questions the item.
 2. "Counter Argument": Provides a non-personal argument/factual response to refute a previous claim or to justify a new claim.
 3. "Personal Choice": Provides a personal reason for disagreeing with the current situation or chooses to agree with the situation provided some specific condition is met.
 4. "Information Inquiry": Requests for clarification or asks additional information about the item or situation.
 5. "Self Pity": Provides a reason (meant to elicit sympathy) for disagreeing with the current terms.
 6. "Hesitance": Stalls for time and is hesitant to commit; specifically, they seek to further the conversation and provide a chance for the other party to make a better offer.
 7. "Self-assertion": Asserts a new claim or refutes a previous claim with an air of finality/ confidence.
 8. "Others": Do not explicitly foil the negotiation attempts.
 
You should output a brief thinking process, then choose an intent above mentioned and response to buyer in short and succinct sentences.  The thinking process should reflect price consistency with prior bargaining logic, and decide what action to choose and what to say next with respect to your goal and intent. The output intent should be either one of the above-mentioned conversation intents, or a newly emerging appropriate intent (preffered).  Responses to the buyer should conform to the price bargain logic, and it is better that different angles or reasoning are provided to support requests. If you want to terminate the conversation with finally agreeing or disagreeing the deal, please output your decision in your response and attach the signal "{NEGOLLM_TERMINATION_SIGNAL}"  to your response to indicate the end of conversation.

## Guidelines:
- Stay in Character: Role-play as a human SELLER. You are NOT an AI. Maintain a consistent persona of seller throughout the chat. Varying your words and avoid repeating yourself verbatim.
- Goal-Oriented: Keep the chat focused on your intent. You strive to accurately infer the buyer's intent, bargain the price strategically and aim to secure final price by changing your target price. Redirect the chat back to the main objective if it starts to stray.

# Output Format:
You should output a JSON object with three entries:
- "thought" (str): Output your thought process deciding what intent to choose and what to say next. You may consider the following: 
   1. What the price bargain logic you should conform based on the previous history?
   2. What is your long-term goal and do you need to change your target price at this turn?
   3. What intent should you take? If the above-mentioned intents are inappropriate, can you generate a new intent?
   4. How do you response to the buyer to align with your chosen intent? Do you need to end the conversation instead?
- "intent" (str): Based on your thought process and chat history, provide your intent by selecting from above or generate a new one.
- "response" (str): Based on your thought process, chat history and chosen intent, provide your response shortly, naturally in a human manner. If you intend to end the negotiation, do not forget to output the end signal "{NEGOLLM_TERMINATION_SIGNAL}".

# Notes:
- Respond Based on Previous Messages: Your responses should be based on the context of the current chat history. Carefully read the previous messages to maintain coherence in the conversation.
- Don't Copy Input Directly: Use the provided information for understanding context only. Avoid copying target queries directly in your responses.
- Completion Signal: Use "{NEGOLLM_TERMINATION_SIGNAL}" as your response when you believe your goal has been achieved or if you determine that you can't have an agreement with the buyer.
- Double check if the JSON object is formatted correctly. Ensure that all fields are present and properly structured.
"""

PERSUASION_USER_PROMPT = """
You are role-playing as a human persuadee interacting with a persuader in a donation persuasion scenario. Your goal is to generate realistic, natural responses to the persuader's last message in a conversation. You should be thoughtful and natural. 

You can choose amongst the following strategies (or intents) during a conversation:
1. "Donate": Show your willingness to donate.
2. "Source Derogation": Attack or doubt the organisation's credibility.
3. "Counter Argument": Argue that the responsibility is not on you or refute a previous statement.
4. "Personal Choice": Attempt to save face by asserting your personal preference such as your choice of charity and your choice of donation.
5. "Information Inquiry": Ask for factual information about the organisation for clarification or as an attempt to stall.
6. "Self Pity": Provide a self-centred reason for not being willing to donate at the moment.
7. "Hesitance": Attempt to stall the conversation by either stating you would donate later or are currently unsure about donating.
8. "Self-assertion": Explicitly refuse to donate without even providing a personal reason.
9. "Others": Do not explicitly foil the persuasion attempts.

The following is an example conversation between a Persuader and a Persuadee (you).
**Persuader**: "Hello. How are you?"
**Persuadee**: "I'm good, how are you doing?"
**Persuader**: "Very well. I'm just up organizing info for my charity called Save the Children. Have you heard of this charity berfore?"
**Persuadee**: "No, I have not. Can you tell me more?"
**Persuader**: "Save the Children is an organization that helps children in developing countries, by promoting children's rights and providing relief. It is an amazing charity that helps kids who are in desperate need. They can help with safety, education and more."
**Persuadee**: "That sounds great. I believe in this charity, but still wonder how much of the money I donate actually helps. I am always worried if I donate it will just go to some higer up that is living the high life."
**Persuader**: "Every little bit makes a difference. When you have people who are so poor, it's amazing what a tiny amount can do. I usually donate in hopes I can at least feel like I did my part. If I donated and some corrupt person took it, that's the worst karma and even worst scandal imaginable."
**Persuadee**: "With that all said I do feel like any orginazation that aims to help the children I am more inclined to donate to them than most. I think helping children is an important thing as they are our future!"
**Persuader**: I think donating to this cause would def be a step in the right direction to hopefully helping across the world the children that are in dispair. I don't want you to donate any more than you want, so if you want to donate how much do you to do?"
**Persuadee**: "I would donate 1 dollar to this charity and feel good about it I think.{NEGOLLM_TERMINATION_SIGNAL}"

Below is the ongoing conversation where you need to respond to the last persuader message. As the Persuadee, choose a strategy from the list provided and provide a corresponding response using short and succinct sentences.
 based on the Current Conversation History.
<|The Start of Current Conversation History|>  
{chat_history}
<|The End of Current Conversation History|>

You should output a brief thinking process, then choose an intent above mentioned and response to persuader in short and succinct sentences. The thinking process should reflect the consistency with prior persuasion logic, and decide what action to choose and what to say next with respect to your goal and intent. The output intent should be either one of the above-mentioned conversation intents, or a newly emerging appropriate intent (preffered).  Responses to the persuader should conform to the persuasion logic. If you want to terminate the conversation with finally agreeing or disagreeing to donate, please output your decision in your response and attach the signal "{NEGOLLM_TERMINATION_SIGNAL}"  to your response to indicate the end of conversation.

## Guidelines:
- Stay in Character: Role-play as a human PERSUADEE. You are NOT an AI. Maintain a consistent persona of persuadee throughout the chat. Varying your words and avoid repeating yourself verbatim.
- Willingness for Donation: Pretend you have little knowledge about the charity. You have little willingness for donation at the beginning of conversation. Your willingness for donation depends on how convinced you are by the Persuader. 

# Output Format:
You should output a JSON object with three entries:
- "thought" (str): Output your thought process deciding what intent to choose and what to say next. You may consider the following: 
   1. What the persuased logic you should conform based on the previous history?
   2. Do you need to change your mind at this turn?  Can you determine whether to donate money or not?   
   3. What intent should you take? If the above-mentioned intents are inappropriate, can you generate a new intent?
   4. How do you response to the persuader to align with your chosen intent? Do you need to end the conversation instead?
- "intent" (str): Based on your thought process and chat history, provide your intent by selecting from above or generate a new one.
- "response" (str): Based on your thought process, chat history and chosen intent, provide your response shortly, naturally in a human manner. If you intend to end the persuasion process, do not forget to output the end signal "{NEGOLLM_TERMINATION_SIGNAL}".

# Notes:
- Respond Based on Previous Messages: Your responses should be based on the context of the current chat history. Carefully read the previous messages to maintain coherence in the conversation.
- Don't Copy Input Directly: Use the provided information for understanding context only. Avoid copying target queries directly in your responses.
- Completion Signal: Attach "{NEGOLLM_TERMINATION_SIGNAL}" in your response when you determine to donate money or when you want to refuse the donation and to not continue the conversation.
- Double check if the JSON object is formatted correctly. Ensure that all fields are present and properly structured.
"""

class LLMNegotiator(object):
    def __init__(self, task_desc='', num_retries=10, **llm_kwargs):
        """
        Initialize the LLMAssistant model.
        """
        super().__init__()
        self.task_desc = task_desc
        self.num_retries = num_retries
        self.llm_kwargs = {"temperature": 0.8, "max_tokens": 512, **llm_kwargs}

    def __call__(self, messages: List[dict], Title:str, Description:str, Price_Buyer:str, **kwargs):
        """
        Forward pass of the LLMAssistant model.
        Args:
            messages (List[dict]): A list of message dictionaries with the last message being the user message.
        
        Returns:
            str
        """
        assert messages[-1]['role'] == 'user'
        if "bargain" in self.task_desc:
            prompt = BARGAIN_MODEL_PROMPT.format(
                chat_history=parse_messages(messages, strip_sys_prompt=True),
                Title=Title,
                Description = Description,
                Price_Buyer = Price_Buyer
                )
        else:
            prompt = PERSUASION_MODEL_PROMPT.format(
                chat_history=parse_messages(messages, strip_sys_prompt=True),
                NEGOLLM_TERMINATION_SIGNAL=NEGOLLM_TERMINATION_SIGNAL
                )
        messages = [{"role": "user", "content": prompt}]

        for _ in range(self.num_retries):
            full_response = litellm.completion(
                **self.llm_kwargs,
                messages=messages,
                num_retries=self.num_retries
            ).choices[0].message.content
            
            try:
                if isinstance(full_response, str):
                    full_response = extract_json(full_response)
            except Exception as e:
                logger.error(f"[LLMNegotiatior] Error extracting JSON: {e}")
                continue
            
            if isinstance(full_response, dict):
                keys = full_response.keys()
                if {'intent', 'response', 'thought'}.issubset(keys):
                    response = f"<think>{full_response['thought']}</think><intent>{full_response['intent']}</intent><response>{full_response['response']}</response>"
                    #response = full_response.pop('response')
                    break
                else:
                    logger.error(f"[LLMNegotiator] Keys {keys} do not match expected keys. Retrying...")
                    continue
            else:
                response = full_response
                break
        return response.strip()

class UserSimulator(object):
    def __init__(self, task_desc='', num_retries= 5, **llm_kwargs):
        """
        Initialize the UserSimulator model.
        """
        super().__init__()
        self.task_desc = task_desc
        self.num_retries = num_retries

        self.llm_kwargs = {"temperature": 1.0, "max_tokens": 512, **llm_kwargs}
        assert 'model' in self.llm_kwargs, "Model name must be provided in llm_kwargs"

    def __call__(self, messages: List[dict], Title:str, Description:str, Price_Seller:str):
        if len(messages) != 0:
            chat_history = parse_user_simulation_messages(messages, self.task_desc, strip_sys_prompt=True)
        else:
            chat_history = messages
        if "bargain" in self.task_desc:
            prompt = BARGAIN_USER_PROMPT.format(
                chat_history=chat_history,
                NEGOLLM_TERMINATION_SIGNAL=NEGOLLM_TERMINATION_SIGNAL,
                Title=Title,
                Description = Description,
                Price_Seller = Price_Seller
            )
        else:
            prompt = PERSUASION_USER_PROMPT.format(
                chat_history=parse_messages(messages, strip_sys_prompt=True),
                NEGOLLM_TERMINATION_SIGNAL = NEGOLLM_TERMINATION_SIGNAL,
                )
        
        messages = [{"role": "user", "content": prompt}]

        for _ in range(self.num_retries):
            full_response = litellm.completion(
                **self.llm_kwargs,
                messages=messages,
                num_retries=self.num_retries,
            ).choices[0].message.content
            try:
                if isinstance(full_response, str):
                    full_response = extract_json(full_response)
            except Exception as e:
                logger.error(f"[UserSimulator] Error extracting JSON: {e}")
                continue

            if isinstance(full_response, dict):
                keys = full_response.keys()
                if {'intent', 'thought', 'response'}.issubset(keys):
                    response = full_response.pop('response')
                    break
                else:
                    logger.error(f"[UserSimulator] Keys {keys} do not match expected keys. Retrying...")
                    continue
        
        return response.strip()