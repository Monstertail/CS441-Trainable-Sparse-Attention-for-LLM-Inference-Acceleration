

from typing import List, Dict, Any

import torch


def pre_process_strategy_qa(
    instance,
    tokenizer,
    assistant_tokenizer,
    num_thought_tokens=2,
    device=None,
    add_bot_eot=True,
    split='train',
    base_backbone='llama',
    base_special_token=['<|end_of_text|>', '<|reserved_special_token_0|>', '<|reserved_special_token_1|>'],
    assistant_backbone='llama',
    assistant_special_token=['<|end_of_text|>', '<|reserved_special_token_0|>', '<|reserved_special_token_1|>'],
    **kwargs,
):
    base_unk_token, base_bot_token, base_eot_token = base_special_token
    base_bot_token = base_bot_token if add_bot_eot else ''
    base_eot_token = base_eot_token if add_bot_eot else ''

    assistant_unk_token, assistant_bot_token, assistant_eot_token = assistant_special_token

    answer = 'Yes' if instance['answer'] else 'No'
    reasoning_list = instance['facts']

    thought_tokens = base_unk_token * num_thought_tokens
    soft_thoughts = f'{base_bot_token}{thought_tokens}{base_eot_token}'

    input_content = f'You are required to answer the following question with `Yes` or `No`: {instance["question"]}\n\n'
    if num_thought_tokens > 0:
        input_content += f'Here are something useful for your reasoning: {soft_thoughts}\n\n'
    input_content += f'Therefore, the final answer is `Yes` or `No`?'

    if split in ['train', 'dev']:
        target_content = (f'OK. The question is {instance["question"]}\n\n'
                          f'Now let\'s start reasoning according to the following facts:\n')
    else:
        target_content = ''

    cot_template = ''
    if split in ['train', 'dev']:
        for idx in range(len(reasoning_list)):
            cot_template += f'## Step {idx + 1}: {reasoning_list[idx]}\n'
        cot_template += f'Finished reasoning, the answer is: {answer}.'

    target_content += cot_template

    input_messages = [
        {
            'role': 'user',
            'content': input_content,
        },
    ]
    target_messages = [
        {
            'role': 'user',
            'content': input_content,
        },
        {
            'role': 'assistant',
            'content': target_content,
        },
    ]

    if split in ['train', 'dev']:
        input_ids = tokenizer.apply_chat_template(target_messages)
        pure_input_length = len(tokenizer.apply_chat_template(input_messages))
    else:
        input_ids = tokenizer.apply_chat_template(input_messages)
        pure_input_length = len(input_ids)
    attention_mask = [1] * len(input_ids)

    assistant_template = (f'You are required to generate {num_thought_tokens} tokens to help another language model '
                          f'to solve the following strategy reasoning task efficiently and clearly:\n'
                          f'- The tokens should include some useful information for the reasoning problem.\n'
                          f'- Generate the tokens starts from the most important or the highest related tokens.\n'
                          f'- **Informative tokens are required**: (1) Do not need to generate a sentence or paragraph, '
                          f'(2) Do not need to generate the uninformative tokens such as serial number.\n'
                          f'- The tokens should be useful for large language model to answer the question with '
                          f'`Yes` or `No`.\n'
                          f'...\n\n'
                          f'Here is the problem: {instance["question"]}')

    assistant_messages = [
        {
            'role': 'user',
            'content': f'{assistant_template}',
        },
        {
            'role': 'assistant',
            'content': f'Here are {num_thought_tokens} tokens to help the language model solve this strategy reasoning task: '
                       f'{assistant_unk_token * num_thought_tokens}',
        }
    ]

    assistant_ids = assistant_tokenizer.apply_chat_template(assistant_messages)
    assistant_attention_mask = [1] * len(assistant_ids)

    if base_backbone in ['llama', 'qwen']:
        input_thought_start_idx = pure_input_length - 16 - num_thought_tokens
        if add_bot_eot:
            input_thought_start_idx -= 1
        if base_backbone in ['qwen']:
            input_thought_start_idx -= 1
    else:
        raise NotImplementedError
    if assistant_backbone in ['llama', 'qwen']:
        assistant_thought_start_idx = len(assistant_ids) - 1 - num_thought_tokens
        if assistant_backbone in ['qwen']:
            assistant_thought_start_idx -= 1
    else:
        raise NotImplementedError

    input_thought_end_idx = input_thought_start_idx + num_thought_tokens
    assistant_thought_end_idx = assistant_thought_start_idx + num_thought_tokens

    if split in ['train']:
        labels = [-100] * pure_input_length + input_ids[pure_input_length:]
    elif split in ['dev']:
        labels = [-100] * len(input_ids)
        labels[-3] = input_ids[-3]
    else:
        labels = [-100] * len(input_ids)

    inputs = {
        'input_ids': input_ids,
        'attention_mask': attention_mask,
        'assistant_input_ids': assistant_ids,
        'assistant_attention_mask': assistant_attention_mask,
        'labels': labels,
        'thought_index': [input_thought_start_idx, input_thought_end_idx, assistant_thought_start_idx, assistant_thought_end_idx],
    }
    if split in ['train', 'dev']:
        inputs['answer'] = answer

    if device is not None:
        inputs = {
            k: torch.tensor(v).unsqueeze(0).to(device) for k, v in inputs.items() if isinstance(v, List)
        }

    return inputs


def pre_process_gsm8k(
    instance,
    tokenizer,
    assistant_tokenizer,
    num_thought_tokens=2,
    device=None,
    add_bot_eot=True,
    split='train',
    base_backbone='llama',
    base_special_token=['<|end_of_text|>', '<|reserved_special_token_0|>', '<|reserved_special_token_1|>'],
    assistant_backbone='llama',
    assistant_special_token=['<|end_of_text|>', '<|reserved_special_token_0|>', '<|reserved_special_token_1|>'],
    **kwargs,
):
    base_unk_token, base_bot_token, base_eot_token = base_special_token
    base_bot_token = base_bot_token if add_bot_eot else ''
    base_eot_token = base_eot_token if add_bot_eot else ''

    assistant_unk_token, assistant_bot_token, assistant_eot_token = assistant_special_token

    reasoning_list = instance['answer'].split('\n')
    answer = reasoning_list[-1]
    assert answer.startswith('####')
    answer = answer.replace(',', '')
    if '.' in answer:
        answer = float(answer[4:])
    else:
        answer = int(answer[4:])
    question = instance['question']

    thought_tokens = base_unk_token * num_thought_tokens
    soft_thoughts = f'{base_bot_token}{thought_tokens}{base_eot_token}'

    input_template = (f'Solve the following math problem efficiently and clearly:\n'
                      f'- For simple problems (2 steps or fewer):\nProvide a concise solution with minimal equation.\n'
                      f'- For complex problems (3 steps or more):\n'
                      f'Use this step-by-step format:\n\n'
                      f'## Step 1: [Brief calculations]\n'
                      f'## Step 2: [Brief calculations]\n'
                      f'...\n'
                      f'Regardless of the approach, always conclude with:\n'
                      f'Therefore, the final answer is: $\\boxed{{answer}}$. I hope it is correct.\n'
                      f'Where [answer] is just the final number or expression that solves the problem.\n\n'
                      f'Problem: {question}')

    input_content = f'{input_template}\n\n'
    if num_thought_tokens > 0:
        input_content += (f'There are some prompts generated by a weaker assistant model. Some prompts maybe useful '
                          f'while others maybe unuseful for your reasoning. '
                          f'If the prompts are correct, you can use it as reference. If the prompts are not correct, '
                          f'you can ignore them and focus back to solving the problem.\n'
                          f'Here are prompts: {soft_thoughts}')

    target_content = ''

    cot_template = ''
    if split in ['train', 'dev']:
        for idx in range(len(reasoning_list)):
            cot_template += f'## Step {idx + 1}: {reasoning_list[idx]}\n'
        cot_template += f'Therefore, the final answer is: $\\boxed{{{answer}}}$.'

    target_content += cot_template

    input_messages = [
        {
            'role': 'user',
            'content': input_content,
        },
    ]
    target_messages = [
        {
            'role': 'user',
            'content': input_content,
        },
        {
            'role': 'assistant',
            'content': target_content,
        },
    ]

    if split in ['train', 'dev']:
        input_ids = tokenizer.apply_chat_template(target_messages)
        pure_input_length = len(tokenizer.apply_chat_template(input_messages))
    else:
        input_ids = tokenizer.apply_chat_template(input_messages)
        pure_input_length = len(input_ids)
    attention_mask = [1] * len(input_ids)

    if assistant_backbone in ['llama']:
        assistant_template = (
            f'You are required to generate {num_thought_tokens} tokens to help another language model '
            f'to solve the following math reasoning task efficiently and clearly. '
            f'Here are the requirements of your generated tokens:\n'
            f'- The tokens should include some useful information for the reasoning problem, '
            f'for example, the numbers and the operations needed for calculation.\n'
            f'- Generate the tokens starts from the most important or the highest related tokens.\n'
            f'- **Informative tokens are required**: (1) Do not need to generate a sentence or paragraph, '
            f'(2) Do not need to generate the uninformative tokens such as serial number.\n'
            f'- The tokens should be useful for large language model to answer the question with the numbers.\n'
            f'...\n\n'
            f'Here is the problem: {instance["question"]}.'
        )
    elif assistant_backbone in ['qwen']:
        assistant_template = (
            f'You are required to generate {num_thought_tokens} tokens to help another language model '
            f'to solve the following math reasoning task efficiently and clearly. '
            f'Here are the requirements of your generated tokens:\n'
            f'- The tokens should include some useful information for the reasoning problem, '
            f'for example, the numbers and the operations needed for calculation.\n'
            f'- Generate the tokens starts from the most important or the highest related tokens.\n'
            f'- **Informative tokens are required**: (1) Do not need to generate a sentence or paragraph, '
            f'(2) Do not need to generate the uninformative tokens such as serial number.\n'
            f'- The tokens should be useful for large language model to answer the question with the numbers.\n'
            f'- The other language model is good enough to understand the problem, so what you need to do is '
            f'generate some informative key tokens that summarize the problem.\n'
            f'...\n\n'
            f'Here is the problem: {instance["question"]}.')
    else:
        raise NotImplementedError

    assistant_messages = [
        {
            'role': 'user',
            'content': f'{assistant_template}',
        },
        {
            'role': 'assistant',
            'content': f'Here are {num_thought_tokens} tokens to help the language model solve this math reasoning task: '
                       f'{assistant_unk_token * num_thought_tokens}',
        }
    ]

    assistant_ids = assistant_tokenizer.apply_chat_template(assistant_messages)
    assistant_attention_mask = [1] * len(assistant_ids)

    if base_backbone in ['llama', 'qwen']:
        input_thought_start_idx = pure_input_length - 1 - num_thought_tokens
        if add_bot_eot:
            input_thought_start_idx -= 1
        if base_backbone in ['qwen']:
            input_thought_start_idx -= 1
    else:
        raise NotImplementedError
    if assistant_backbone in ['llama', 'qwen']:
        assistant_thought_start_idx = len(assistant_ids) - 1 - num_thought_tokens
        if assistant_backbone in ['qwen']:
            assistant_thought_start_idx -= 1
    else:
        raise NotImplementedError

    input_thought_end_idx = input_thought_start_idx + num_thought_tokens
    assistant_thought_end_idx = assistant_thought_start_idx + num_thought_tokens

    if split in ['train']:
        labels = [-100] * (input_thought_end_idx + 2) + input_ids[input_thought_end_idx + 2:]
    elif split in ['dev']:
        labels = [-100] * len(input_ids)
        labels[-5:] = input_ids[-5:]
    else:
        labels = [-100] * len(input_ids)

    inputs = {
        'input_ids': input_ids,
        'attention_mask': attention_mask,
        'assistant_input_ids': assistant_ids,
        'assistant_attention_mask': assistant_attention_mask,
        'labels': labels,
        'thought_index': [input_thought_start_idx, input_thought_end_idx, assistant_thought_start_idx,
                          assistant_thought_end_idx],
    }
    if split in ['train', 'dev']:
        inputs['answer'] = answer

    if device is not None:
        inputs = {
            k: torch.tensor(v).unsqueeze(0).to(device) for k, v in inputs.items() if isinstance(v, List)
        }

    return inputs


def pre_process_aqua(
    instance,
    tokenizer,
    assistant_tokenizer,
    num_thought_tokens=2,
    device=None,
    add_bot_eot=True,
    split='train',
    base_backbone='llama',
    base_special_token=['<|end_of_text|>', '<|reserved_special_token_0|>', '<|reserved_special_token_1|>'],
    assistant_backbone='llama',
    assistant_special_token=['<|end_of_text|>', '<|reserved_special_token_0|>', '<|reserved_special_token_1|>'],
    **kwargs,
):
    base_unk_token, base_bot_token, base_eot_token = base_special_token
    base_bot_token = base_bot_token if add_bot_eot else ''
    base_eot_token = base_eot_token if add_bot_eot else ''

    assistant_unk_token, assistant_bot_token, assistant_eot_token = assistant_special_token

    reasoning_list = instance['answer'].split('\n')
    answer = reasoning_list[-1]
    assert answer.startswith('####')
    answer = answer.replace(',', '')
    answer = answer[4:].strip()

    question = instance['question']

    thought_tokens = base_unk_token * num_thought_tokens
    soft_thoughts = f'{base_bot_token}{thought_tokens}{base_eot_token}'

    input_template = (f'You are required to solve the following math multiple choices questions.\n'
                      f'In the multiple choices question, there are five options: A, B, C, D, and E, respectively.\n'
                      f'The correct answer that solves the problem is one of these options.\n'
                      f'Your job is to solve the problem and find the correct option.\n'
                      f'Here are the instructions for solving the problem efficiently and clearly:\n'
                      f'- For simple problems (2 steps or fewer):\nProvide a concise solution with minimal equation.\n'
                      f'- For complex problems (3 steps or more):\n'
                      f'Use this step-by-step format:\n\n'
                      f'## Step 1: [Brief calculations]\n'
                      f'## Step 2: [Brief calculations]\n'
                      f'...\n'
                      f'Regardless of the approach, always conclude with the following sentence:\n'
                      f'Therefore, the final answer is: $\\boxed{{answer}}$. I hope it is correct.\n'
                      f'Where [answer] is the option from A, B, C, D, and E.\n'
                      f'Only one letter from A to E is accepted in the answer span.\n\n'
                      f'Problem: {question}')

    input_content = f'{input_template}\n\n'
    if num_thought_tokens > 0:
        input_content += (f'There are some prompts generated by a weaker assistant model. Some prompts maybe useful '
                          f'while others maybe unuseful for your reasoning. '
                          f'If the prompts are correct, you can use it as reference. If the prompts are not correct, '
                          f'you can ignore them and focus back to solving the problem.\n'
                          f'Here are prompts: {soft_thoughts}')

    target_content = ''

    cot_template = ''
    if split in ['train', 'dev']:
        for idx in range(len(reasoning_list)):
            cot_template += f'## Step {idx + 1}: {reasoning_list[idx]}\n'
        cot_template += f'Therefore, the final answer is: $\\boxed{{{answer}}}$.'

    target_content += cot_template

    input_messages = [
        {
            'role': 'user',
            'content': input_content,
        },
    ]
    target_messages = [
        {
            'role': 'user',
            'content': input_content,
        },
        {
            'role': 'assistant',
            'content': target_content,
        },
    ]

    if split in ['train', 'dev']:
        input_ids = tokenizer.apply_chat_template(target_messages)
        pure_input_length = len(tokenizer.apply_chat_template(input_messages))
    else:
        input_ids = tokenizer.apply_chat_template(input_messages)
        pure_input_length = len(input_ids)
    attention_mask = [1] * len(input_ids)

    if assistant_backbone in ['llama']:
        assistant_template = (
            f'You are required to generate {num_thought_tokens} tokens to help another language model '
            f'to solve the following math reasoning task efficiently and clearly. '
            f'Here are the requirements of your generated tokens:\n'
            f'- The tokens should include some useful information for the reasoning problem, '
            f'for example, the numbers and the operations needed for calculation.\n'
            f'- Generate the tokens starts from the most important or the highest related tokens.\n'
            f'- **Informative tokens are required**: (1) Do not need to generate a sentence or paragraph, '
            f'(2) Do not need to generate the uninformative tokens such as serial number.\n'
            f'- The tokens should be useful for large language model to answer the question with the numbers.\n'
            f'...\n\n'
            f'Here is the problem: {instance["question"]}.'
        )
    elif assistant_backbone in ['qwen']:
        assistant_template = (
            f'You are required to generate {num_thought_tokens} tokens to help another language model '
            f'to solve the following math reasoning task efficiently and clearly. '
            f'Here are the requirements of your generated tokens:\n'
            f'- The tokens should include some useful information for the reasoning problem, '
            f'for example, the numbers and the operations needed for calculation.\n'
            f'- Generate the tokens starts from the most important or the highest related tokens.\n'
            f'- **Informative tokens are required**: (1) Do not need to generate a sentence or paragraph, '
            f'(2) Do not need to generate the uninformative tokens such as serial number.\n'
            f'- The tokens should be useful for large language model to answer the question with the numbers.\n'
            f'- The other language model is good enough to understand the problem, so what you need to do is '
            f'generate some informative key tokens that summarize the problem.\n'
            f'...\n\n'
            f'Here is the problem: {instance["question"]}.')
    else:
        raise NotImplementedError

    assistant_messages = [
        {
            'role': 'user',
            'content': f'{assistant_template}',
        },
        {
            'role': 'assistant',
            'content': f'Here are {num_thought_tokens} tokens to help the language model solve this math reasoning task: '
                       f'{assistant_unk_token * num_thought_tokens}',
        }
    ]

    assistant_ids = assistant_tokenizer.apply_chat_template(assistant_messages)
    assistant_attention_mask = [1] * len(assistant_ids)

    if base_backbone in ['llama', 'qwen']:
        input_thought_start_idx = pure_input_length - 1 - num_thought_tokens
        if add_bot_eot:
            input_thought_start_idx -= 1
        if base_backbone in ['qwen']:
            input_thought_start_idx -= 1
    else:
        raise NotImplementedError
    if assistant_backbone in ['llama', 'qwen']:
        assistant_thought_start_idx = len(assistant_ids) - 1 - num_thought_tokens
        if assistant_backbone in ['qwen']:
            assistant_thought_start_idx -= 1
    else:
        raise NotImplementedError

    input_thought_end_idx = input_thought_start_idx + num_thought_tokens
    assistant_thought_end_idx = assistant_thought_start_idx + num_thought_tokens

    if split in ['train']:
        labels = [-100] * (input_thought_end_idx + 2) + input_ids[input_thought_end_idx + 2:]
    elif split in ['dev']:
        labels = [-100] * len(input_ids)
        labels[-5:] = input_ids[-5:]
    else:
        labels = [-100] * len(input_ids)

    inputs = {
        'input_ids': input_ids,
        'attention_mask': attention_mask,
        'assistant_input_ids': assistant_ids,
        'assistant_attention_mask': assistant_attention_mask,
        'labels': labels,
        'thought_index': [input_thought_start_idx, input_thought_end_idx, assistant_thought_start_idx,
                          assistant_thought_end_idx],
    }
    if split in ['train', 'dev']:
        inputs['answer'] = answer

    if device is not None:
        inputs = {
            k: torch.tensor(v).unsqueeze(0).to(device) for k, v in inputs.items() if isinstance(v, List)
        }

    return inputs


def pre_process_du(
    instance,
    tokenizer,
    assistant_tokenizer,
    num_thought_tokens=2,
    device=None,
    add_bot_eot=True,
    split='train',
    base_backbone='llama',
    base_special_token=['<|end_of_text|>', '<|reserved_special_token_0|>', '<|reserved_special_token_1|>'],
    assistant_backbone='llama',
    assistant_special_token=['<|end_of_text|>', '<|reserved_special_token_0|>', '<|reserved_special_token_1|>'],
    **kwargs,
):
    base_unk_token, base_bot_token, base_eot_token = base_special_token
    base_bot_token = base_bot_token if add_bot_eot else ''
    base_eot_token = base_eot_token if add_bot_eot else ''

    assistant_unk_token, assistant_bot_token, assistant_eot_token = assistant_special_token

    reasoning_list = instance['answer'].split('\n')
    answer = reasoning_list[-1]
    answer = answer.replace(',', '').strip()

    question = instance['question']

    thought_tokens = base_unk_token * num_thought_tokens
    soft_thoughts = f'{base_bot_token}{thought_tokens}{base_eot_token}'

    input_template = (f'You are required to solve the following math multiple choices questions.\n'
                      f'In the multiple choices question, there are five options: A, B, C, D, E, and F, respectively.\n'
                      f'The correct answer that solves the problem is one of these options.\n'
                      f'Your job is to solve the problem and find the correct option.\n'
                      f'Here are the instructions for solving the problem efficiently and clearly:\n'
                      f'- For simple problems (2 steps or fewer):\nProvide a concise solution with minimal description.\n'
                      f'- For complex problems (3 steps or more):\n'
                      f'Use this step-by-step format:\n\n'
                      f'## Step 1: [Brief reasoning step]\n'
                      f'## Step 2: [Brief reasoning step]\n'
                      f'...\n'
                      f'Regardless of the approach, always conclude with the following sentence:\n'
                      f'Therefore, the final answer is: $\\boxed{{answer}}$. I hope it is correct.\n'
                      f'Where [answer] is the option from A, B, C, D, E, and F.\n'
                      f'Only one letter from A to F is accepted in the answer span.\n\n'
                      f'Problem: {question}')

    input_content = f'{input_template}\n\n'
    if num_thought_tokens > 0:
        input_content += (f'There are some prompts generated by a weaker assistant model. Some prompts maybe useful '
                          f'while others maybe unuseful for your reasoning. '
                          f'If the prompts are correct, you can use it as reference. If the prompts are not correct, '
                          f'you can ignore them and focus back to solving the problem.\n'
                          f'Here are prompts: {soft_thoughts}')

    target_content = ''

    cot_template = ''
    if split in ['train', 'dev']:
        for idx in range(len(reasoning_list)):
            cot_template += f'## Step {idx + 1}: {reasoning_list[idx]}\n'
        cot_template += f'Therefore, the final answer is: $\\boxed{{{answer}}}$.'

    target_content += cot_template

    input_messages = [
        {
            'role': 'user',
            'content': input_content,
        },
    ]
    target_messages = [
        {
            'role': 'user',
            'content': input_content,
        },
        {
            'role': 'assistant',
            'content': target_content,
        },
    ]

    if split in ['train', 'dev']:
        input_ids = tokenizer.apply_chat_template(target_messages)
        pure_input_length = len(tokenizer.apply_chat_template(input_messages))
    else:
        input_ids = tokenizer.apply_chat_template(input_messages)
        pure_input_length = len(input_ids)
    attention_mask = [1] * len(input_ids)

    if assistant_backbone in ['llama']:
        assistant_template = (
            f'You are required to generate {num_thought_tokens} tokens to help another language model '
            f'to solve the following date understanding task efficiently and clearly. '
            f'Here are the requirements of your generated tokens:\n'
            f'- The tokens should include some useful information for the reasoning problem, '
            f'for example, the numbers and the operations needed for calculation.\n'
            f'- Generate the tokens starts from the most important or the highest related tokens.\n'
            f'- **Informative tokens are required**: (1) Do not need to generate a sentence or paragraph, '
            f'(2) Do not need to generate the uninformative tokens such as serial number.\n'
            f'- The tokens should be useful for large language model to answer the question with the numbers.\n'
            f'...\n\n'
            f'Here is the problem: {instance["question"]}.'
        )
    elif assistant_backbone in ['qwen']:
        assistant_template = (
            f'You are required to generate {num_thought_tokens} tokens to help another language model '
            f'to solve the following math reasoning task efficiently and clearly. '
            f'Here are the requirements of your generated tokens:\n'
            f'- The tokens should include some useful information for the reasoning problem, '
            f'for example, the numbers and the operations needed for calculation.\n'
            f'- Generate the tokens starts from the most important or the highest related tokens.\n'
            f'- **Informative tokens are required**: (1) Do not need to generate a sentence or paragraph, '
            f'(2) Do not need to generate the uninformative tokens such as serial number.\n'
            f'- The tokens should be useful for large language model to answer the question with the numbers.\n'
            f'- The other language model is good enough to understand the problem, so what you need to do is '
            f'generate some informative key tokens that summarize the problem.\n'
            f'...\n\n'
            f'Here is the problem: {instance["question"]}.')
    else:
        raise NotImplementedError

    assistant_messages = [
        {
            'role': 'user',
            'content': f'{assistant_template}',
        },
        {
            'role': 'assistant',
            'content': f'Here are {num_thought_tokens} tokens to help the language model solve this math reasoning task: '
                       f'{assistant_unk_token * num_thought_tokens}',
        }
    ]

    assistant_ids = assistant_tokenizer.apply_chat_template(assistant_messages)
    assistant_attention_mask = [1] * len(assistant_ids)

    if base_backbone in ['llama', 'qwen']:
        input_thought_start_idx = pure_input_length - 1 - num_thought_tokens
        if add_bot_eot:
            input_thought_start_idx -= 1
        if base_backbone in ['qwen']:
            input_thought_start_idx -= 1
    else:
        raise NotImplementedError
    if assistant_backbone in ['llama', 'qwen']:
        assistant_thought_start_idx = len(assistant_ids) - 1 - num_thought_tokens
        if assistant_backbone in ['qwen']:
            assistant_thought_start_idx -= 1
    else:
        raise NotImplementedError

    input_thought_end_idx = input_thought_start_idx + num_thought_tokens
    assistant_thought_end_idx = assistant_thought_start_idx + num_thought_tokens

    if split in ['train']:
        labels = [-100] * (input_thought_end_idx + 2) + input_ids[input_thought_end_idx + 2:]
    elif split in ['dev']:
        labels = [-100] * len(input_ids)
        labels[-5:] = input_ids[-5:]
    else:
        labels = [-100] * len(input_ids)

    inputs = {
        'input_ids': input_ids,
        'attention_mask': attention_mask,
        'assistant_input_ids': assistant_ids,
        'assistant_attention_mask': assistant_attention_mask,
        'labels': labels,
        'thought_index': [input_thought_start_idx, input_thought_end_idx, assistant_thought_start_idx,
                          assistant_thought_end_idx],
    }
    if split in ['train', 'dev']:
        inputs['answer'] = answer

    if device is not None:
        inputs = {
            k: torch.tensor(v).unsqueeze(0).to(device) for k, v in inputs.items() if isinstance(v, List)
        }

    return inputs



class CustomDataCollator:

    def __call__(self, batch: List[Dict[str, Any]]) -> Dict[str, torch.Tensor]:
        input_ids = [item['input_ids'] for item in batch]
        attention_masks = [item['attention_mask'] for item in batch]
        labels = [item['labels'] for item in batch]
        thought_index = [item['thought_index'] for item in batch]
        assistant_input_ids = [item['assistant_input_ids'] for item in batch]
        assistant_attention_mask = [item['assistant_attention_mask'] for item in batch]

        input_max_length = max([len(item) for item in input_ids])
        assistant_max_length = max([len(item) for item in assistant_input_ids])

        input_ids = [(ids + [0] * input_max_length)[:input_max_length] for ids in input_ids]
        attention_masks = [(am + [0] * input_max_length)[:input_max_length] for am in attention_masks]
        labels = [(label + [-100] * input_max_length)[:input_max_length] for label in labels]
        assistant_input_ids = [(ids + [0] * assistant_max_length)[:assistant_max_length]
                               for ids in assistant_input_ids]
        assistant_attention_mask = [(am + [0] * assistant_max_length)[:assistant_max_length]
                                    for am in assistant_attention_mask]

        return {
            'input_ids': torch.tensor(input_ids).long(),
            'attention_mask': torch.tensor(attention_masks).long(),
            'labels': torch.tensor(labels).long(),
            'thought_index': torch.tensor(thought_index).long(),
            'assistant_input_ids': torch.tensor(assistant_input_ids).long(),
            'assistant_attention_mask': torch.tensor(assistant_attention_mask).long(),
        }

