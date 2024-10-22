import asyncio
import openai
from openai import OpenAI
from typing import List, Dict


def inference(#async 
    api_key: str, 
    api_base: str, 
    client: OpenAI,
    model_path: str,
    messages,
    model_name: str = "davinci",
    message_index: int = 0,
    ):
    chat_outputs = client.chat.completions.create(
        model=model_path,
        messages=messages,
        stop="<|eot_id|>",
        max_tokens=50,
        frequency_penalty=0.5,
        temperature=0.9,
    )

    return {
        "model": model_name,
        "message_index": message_index,
        "response": chat_outputs.choices[0].message.content
    }


    

if __name__ == '__main__':
    api_key = "666"
    api_base1 = "http://localhost:8001/v1"
    client1 = OpenAI(api_key=api_key, base_url=api_base1)
    model_path1 = "/home/ubuntu/.cache/modelscope/hub/LLM-Research/Meta-Llama-3-8B"
    
    
    messages=[
            [
            {"role": "system", "content": ""},
            {"role": "user", "content": "output \"True\" or \"False\" randomly once"},
        ]
    ]
    from tqdm import tqdm
    tasks = []
    for i in tqdm(range(len(messages))):
        tasks.append(inference(api_key, api_base1, client1, model_path1, messages[i], 'llama-3-8b', i))

    for result in tasks:
        print(f"Model: {result['model']}, Message Index: {result['message_index']}, Response: {result['response']}")
        print('\n')