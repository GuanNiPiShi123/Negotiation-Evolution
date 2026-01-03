import re
def parse_messages(messages, strip_sys_prompt=True):
    '''
    Args:
        messages: List[dict]
            List of dictionaries with keys 'role' and 'content'
            Example: messages = [{'role': 'user', 'content': 'Hello!'}, 
                                 {'role': 'assistant', 'content': 'Hi!'}, ...]
    '''
    if messages is None: return ''

    if strip_sys_prompt:
        messages = strip_system_prompt(messages)
    
    chat = "\n".join(
        f"**{m['role'].capitalize()}**: {m['content']}" for m in messages
    )

    return chat

def parse_user_simulation_messages(messages, task_desc, strip_sys_prompt=True):
    '''
    Args:
        messages: List[dict]
            List of dictionaries with keys 'role' and 'content'
            Example: messages = [{'role': 'user', 'content': 'Hello!'}, 
                                 {'role': 'assistant', 'content': 'Hi!'}, ...]
    '''
    if messages is None: return ''

    if strip_sys_prompt:
        messages = strip_system_prompt(messages)
    if "bargain" in task_desc:
        messages = [{"role":"buyer", "content": rid_thought(m["content"])} if m["role"] == "assistant" else {"role":"seller", "content": m["content"]} for m in messages]
    else:
        messages = [{"role":"persuader", "content": rid_thought(m["content"])} if m["role"] == "assistant" else {"role":"persuadee", "content": m["content"]} for m in messages]
 
    if messages[0]["content"]=="":
        messages = messages[1:]
        
    chat = "\n".join(
        f"**{m['role'].capitalize()}**: {m['content']}" for m in messages
    )
    return chat

def rid_thought(content):
    thought_pattern = r'<think>(.*?)</think>'
    strategy_pattern = r'<intent>(.*?)</intent>'
    cleaned_text = re.sub(thought_pattern, '', content, flags=re.DOTALL)
    cleaned_text = re.sub(strategy_pattern, '', cleaned_text, flags=re.DOTALL)
    return cleaned_text

def strip_system_prompt(messages):
    '''
    Args:
        messages: List[dict]
            List of dictionaries with keys 'role' and 'content'
            Example: messages = [{'role': 'user', 'content': 'Hello!'}, 
                                 {'role': 'assistant', 'content': 'Hi!'}, ...]
    '''
    return [msg for msg in messages if msg['role'] != 'system']
