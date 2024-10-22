import asyncio
import openai
import json
import pandas as pd
from openai import OpenAI
from typing import List, Dict


def inference(
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
        frequency_penalty=1.0,
        temperature=0.9,
        top_p=0.95
    )

    return {
        "model": model_name,
        "message_index": message_index,
        "response": chat_outputs.choices[0].message.content
    }

def load_dataset(data_path):

    with open(data_path, "r") as f:
        content = json.load(f)
        
    return content
    

if __name__ == '__main__':
    api_key = "666"
    api_base1 = "http://localhost:8001/v1"
    data_path = 'nlp_data/test_data.json'
    res_path = 'nlp_data/output_baseline.csv'
    gt_path = 'nlp_data/ground_truth_baseline.txt'

    # load dataset
    val_data = load_dataset(data_path)
    val_data = val_data
    # input prompt
    messages=[]
    ground_truth = []
    for i in range(len(val_data)):
        instruction = val_data[i]['instruction']
        input_content = val_data[i]['input']
        ground_truth.append(val_data[i]['output'])

        
        messages.append(
            [
                {"role": "system", "content": instruction+"give the answer directly without analyze"},
                {"role": "user", "content": input_content},
            ]
        )


    # load model
    client1 = OpenAI(api_key=api_key, base_url=api_base1)
    model_path1 = "/home/ubuntu/.cache/modelscope/hub/LLM-Research/Meta-Llama-3___1-8B-Instruct"

    # inference
    from tqdm import tqdm
    tasks = []
    for i in tqdm(range(len(messages))):
        print(f"messages: {messages[i]}")
        print('\n')
        tasks.append(inference(api_key, api_base1, client1, model_path1, messages[i], 'llama-3-8b', i))

    models_res = []
    index_res = []
    response_res = []
    for result in tasks:
        
        print(f"Model: {result['model']}, Message Index: {result['message_index']}, Response: {result['response']}")
        models_res.append(result['model'])
        index_res.append(result['message_index'])
        response_res.append(result['response'])
        print('\n')
    
    # save results
    df = pd.DataFrame({'output':response_res,'ground_truth':ground_truth})
    df.to_csv(res_path,index=False,sep=',')
