import os
import json
import openai
import asyncio

import pandas as pd

from tqdm import tqdm
from openai import OpenAI
from typing import List, Dict


def inference(api_key: str, api_base: str, client: OpenAI, model_path: str,
    messages, model_name: str = "davinci", message_index: int = 0,):
    # create clinet
    chat_outputs = client.chat.completions.create(
        model=model_path,
        messages=messages,
        stop="<|eot_id|>",
        max_tokens=256,
        frequency_penalty=1.0,
        temperature=0.9,
        top_p=0.95
    )

    # return response
    return chat_outputs.choices[0].message.content

def load_dataset_csv(data_path, img_path):
    csv_data = pd.read_csv(data_path)
    images, questions, gpt_answers = csv_data['image'].to_list(), csv_data['question'].to_list(), csv_data['gpt_answer'].to_list()    
    json_data = []

    for i in range(len(images)):
        data_tmp = [{
            "role": "user",
            "content": [
                {
                    "type": "text",
                    "text": "Give hints for the question.\n<image>"+questions[i],
                },
                {
                    "type": "image_url",
                    "image_url":
                    {
                    "url": os.path.join(img_path, images[i])
                    }
                }
            ]
        }]
        json_data.append(data_tmp)

    return json_data, gpt_answers, questions, images
    

if __name__ == '__main__':
    api_key = "666"
    api_base_13b = "http://localhost:8003/v1"

    data_path = './prj_data/test_data.csv'
    img_path="/home/ubuntu/6000N/llama_factory/LLaMA-Factory-main/prj_data/sample_images"

    res_path = 'llm_res/score_dpo_2.csv'

    # load dataset
    val_data, val_data_truth, questions, images= load_dataset_csv(data_path, img_path)

    answer_path = '/home/ubuntu/6000N/llama_factory/LLaMA-Factory-main/llm_res/output_dpo_2.csv'
    answer_data = pd.read_csv(answer_path)
    answers = answer_data['response'].to_list()
    gt = answer_data['ground_truth'].to_list()



    # load model
    client_13b = OpenAI(api_key=api_key, base_url=api_base_13b)
    model_path_13b = "/home/ubuntu/.cache/modelscope/hub/swift/llava-1___5-13b-hf"

    # inference
    from tqdm import tqdm
    results = []
    results_hints = []
    for i in tqdm(range(len(gt))):

        new_val_data = [{
            "role": "user",
            "content": [
                {
                    "type": "text",
                    "text": "The question is: <image>."+questions[i]+"\n\n\nThe answer is: "+answers[i]+"\n\n\nThe ground truth is: "+gt[i]+'\n\n\nThe score of ground truth is 100. And the range of score is 0-100. Give the score of the answer directly.',
                },
                {
                    "type": "image_url",
                    "image_url":
                    {
                    "url": os.path.join(img_path, images[i])
                    }
                }
                
            ]
        }]

        response = inference(api_key, api_base_13b, client_13b, model_path_13b, new_val_data, 'llava', i)
        print(response)
        results.append(response)


        if i%25==0:
            # save results
            df = pd.DataFrame({'response':results})
            df.to_csv(res_path,index=False,sep=',')

    df = pd.DataFrame({'response':results})
    df.to_csv(res_path,index=False,sep=',')
