import asyncio
import openai
import json
import pandas as pd
from openai import OpenAI
from typing import List, Dict
from tqdm import tqdm

def inference(
    api_key: str, 
    api_base: str, 
    client: OpenAI,
    model_path: str,
    messages,
    model_name: str,
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
        "response": chat_outputs.choices[0].message.content
    }

def load_dataset(data_path):
    with open(data_path, "r") as f:
        content = json.load(f)
    return content

def format_input_prompt(columns_used, sql, results,result_description):
    prompt = f"""
    Task Type: Tabular Question Answering (Tabular QA)

    I am a tabular QA expert, and here is a task that involves extracting and reasoning over tabular data to answer a specific question.

    Columns Used for answering the question: {columns_used}
    Executed SQL Statement: {sql}
    Query Results obtained after extracting the database by sql: {results}
    Description of Results: {result_description}

    Based on the information provided, please generate an answer to the question. Give the answer directly without analyze.
    The answer should be simple and clear, not textual information about the reasoning process.
    """
    return prompt

if __name__ == '__main__':
    api_key = "666"
    api_base1 = "http://localhost:8002/v1"
    data_path = "/root/nlp_project/RAG/retrievaled_data(latin-1).json"
    res_path = "/root/nlp_project/RAG/baseline_rag_result.csv"
    
    # load data
    data = load_dataset(data_path)

    # input prompt
    messages = []
    ground_truth = []

    for i in range(len(data)):
        instruction = format_input_prompt(data[i]['columns_used'], data[i]['sql'], data[i]['results'], data[i]['result_description'])
        question = data[i]['question']
        ground_truth.append(data[i]['answer'])

        messages.append(
            [
                {"role": "system", "content": instruction},
                {"role": "user", "content": question},
            ]
        )
    
    # load model
    client = OpenAI(api_key=api_key, base_url=api_base1)
    model_path = "/root/nlp_project/Meta-3.1-8B-Instruct"

    #inference
    tasks = []
    for i in tqdm(range(len(messages))):
        print(f"messages: {messages[i]} \n")
        tasks.append(inference(
            api_key,
            api_base1,
            client,
            model_path,
            messages[i],
            'llama-3-8b',
        ))

    model_res = []
    response_res = []
    for index, result in enumerate(tasks):
        print(f"Model: {result['model']}, Index: {index}, Response: {result['response']} \n")
        model_res.append(result['model'])
        response_res.append(result['response'])
    
    # save results
    df = pd.DataFrame({'output': response_res, 'ground_truth': ground_truth})
    df.to_csv(res_path,idnex=False, sep=',')
