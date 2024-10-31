import json
import pandas as pd
import textwrap
import time
from openai import OpenAI
from tools import Sorter, Filter, Calculator, Aggregator
from tqdm import tqdm


def load_dataset(data_path):
    with open(data_path, "r") as f:
        content = json.load(f)

    return content

train_path = '/root/nlp_project/databench/train'
test_path = '/root/nlp_project/databench/test'
dataset_path = '/root/nlp_project/databench/dataset'

PROMPT_TEMPLATE = textwrap.dedent(
    """\
    {few_shot}

    Q: {question}
    """
)
EXTRACT_PROMPT_TEMPLATE = textwrap.dedent(
    """\
    {few_shot}

    Sentence: {sentence}
    """
)
TOOL_START_SEQUENCE = "<<"
TOOL_END_SEQUENCE = ">>"
MAX_CALL_TOOLS = 5

import re

def make_api_call(client, model_path, messages):
    if not isinstance(messages, list) or not all(isinstance(m, dict) and 'role' in m and 'content' in m for m in messages):
        raise ValueError("Messages should be a list of dictionaries with 'role' and 'content' keys")


def make_api_call(client, model_path, messages):

    if isinstance(messages, str):
        messages = [{"role": "user", "content": messages}]
    for message in messages:
        message['content'] = re.sub(r'<<.*?>>', '', message['content'])
    response = client.chat.completions.create(
        model=model_path,
        messages=messages,
        stop=[TOOL_END_SEQUENCE],
        max_tokens=50,
        frequency_penalty=0.5,
        temperature=0.9,
    )
    print("Messages1 being sent to API:")
    # for msg in messages:
    #     print(f"Role: {msg['role']}, Content: {msg['content'][:100]}")
    return response.choices[0].message.content

def extract_finalanswer(answer, model, client):
    with open("extract_few_shot.txt") as f:
        few_shot = f.read()
    lines = answer.split("\n")
    sentence = None
    for line in lines:
        if line.startswith("Therefore"):
            sentence = line
            break
    if sentence == None:
        return answer
    
    extract_prompt = EXTRACT_PROMPT_TEMPLATE.format(
        few_shot=few_shot, sentence=sentence
    )
    response = client.chat.completions.create(
        model=model,
        messages=[{"role": "user", "content": extract_prompt}],
        stop="<|eot_id|>",
        max_tokens=50,
        frequency_penalty=0.5,
        temperature=0.9,
    )
    final_answer_tokens = response.choices[0].message.content.split()[1]
    if len(final_answer_tokens) > 1:
        final_answer = final_answer_tokens[1]
    else:
        final_answer = final_answer_tokens[0] if final_answer_tokens else "N/A"
    # print("final_answer:", final_answer)
    # print("--------------------")
    print("Messages2 being sent to API:")
    # for msg in messages:
    #     print(f"Role: {msg['role']}, Content: {msg['content'][:100]}")
    return final_answer
'''
def inference(
        api_key: str,
        api_base: str,
        client,
        model_path: str,
        messages,
        model_name: str = "davinci",
        message_index: int = 0,
):
    chat_outputs = client.chat.completions.create(
        model=model_path,
        messages=messages,
        stop="<|eot_id|>",
        max_tokens=256,
        frequency_penalty=0.5,
        temperature=0.9,
    )

    return {
        "model": model_name,
        "message_index": message_index,
        "response": chat_outputs.choices[0].message.content
    }
'''

if __name__ == '__main__':
    api_key = "666"
    api_base1 = "http://localhost:8001/v1"
    # load model
    client1 = OpenAI(api_key=api_key, base_url=api_base1)
    model_path1 = "/home/ubuntu/.cache/modelscope/hub/LLM-Research/Meta-Llama-3___1-8B-Instruct"
    res_path = "/home/ubuntu/6000N/llama_factory/LLaMA-Factory-main/output_cot_lora.jsonl"
    data_path = 'nlp_data/test_data.json'

    # Load few-shot examples
    with open("few_shot.txt") as f:
        few_shot = f.read()

    # load dataset
    val_data = load_dataset(data_path)
    val_data = val_data

    # input prompt
    ground_truth = []
    response_res = []
    final_answers = []
    for i in tqdm(range(len(val_data))):
        question = val_data[i]['input']
        ground_truth.append(val_data[i]['output'])

        # Create the prompt as a list of messages
        prompt = [{"role": "user", "content": PROMPT_TEMPLATE.format(few_shot=few_shot, question=question)}]
        answer = ""
        for _ in range(MAX_CALL_TOOLS):
            # Wait for 1 second to avoid the API rate limit
            time.sleep(1)
            # Get response from the model
            response_llama = make_api_call(client1, model_path1, prompt)
            # Update prompt and answer
            prompt.append({"role": "assistant", "content": response_llama})
            answer += response_llama

            if TOOL_START_SEQUENCE not in response_llama:
                break

            # Extract tool name and parameters
            tool_name_start = response_llama.find(TOOL_START_SEQUENCE) + len(TOOL_START_SEQUENCE)
            tool_name_end = response_llama.find(TOOL_END_SEQUENCE, tool_name_start)
            tool_name = response_llama[tool_name_start:tool_name_end].strip()

            # Use appropriate tool
            if tool_name == "Calculator":
                try:
                    expression = response_llama[tool_name_end + len(TOOL_END_SEQUENCE):].strip()
                    response_tool = Calculator.calculate(expression)
                    prompt.append({"role": "assistant", "content": TOOL_END_SEQUENCE + " " + str(response_tool)})
                    answer += TOOL_END_SEQUENCE + " " + str(response_tool) + "\n"
                except Exception as e:
                    print("Error in Calculator:", e)
                    prompt.append({"role": "assistant", "content": TOOL_END_SEQUENCE})
                    answer += TOOL_END_SEQUENCE

            elif tool_name == "Filter":
                try:
                    params = response_llama[tool_name_end + len(TOOL_END_SEQUENCE):].strip().split()
                    column = params[0]
                    value = params[1]
                    response_tool = Filter.filter(answer, column, value)
                    prompt.append({"role": "assistant", "content": TOOL_END_SEQUENCE + " " + str(response_tool)})
                    answer += TOOL_END_SEQUENCE + " " + str(response_tool) + "\n"
                except Exception as e:
                    print("Error in Filter:", e)
                    prompt.append({"role": "assistant", "content": TOOL_END_SEQUENCE})
                    answer += TOOL_END_SEQUENCE

            elif tool_name == "Sorter":
                try:
                    params = response_llama[tool_name_end + len(TOOL_END_SEQUENCE):].strip().split()
                    column = params[0]
                    ascending = params[1].lower() == "true"
                    response_tool = Sorter.sort(answer, column, ascending)
                    prompt.append({"role": "assistant", "content": TOOL_END_SEQUENCE + " " + str(response_tool)})
                    answer += TOOL_END_SEQUENCE + " " + str(response_tool) + "\n"
                except Exception as e:
                    print("Error in Sorter:", e)
                    prompt.append({"role": "assistant", "content": TOOL_END_SEQUENCE})
                    answer += TOOL_END_SEQUENCE

            elif tool_name == "Aggregator":
                try:
                    column = response_llama[tool_name_end + len(TOOL_END_SEQUENCE):].strip()
                    response_tool = Aggregator.sum_column(answer, column)
                    prompt.append({"role": "assistant", "content": TOOL_END_SEQUENCE + " " + str(response_tool)})
                    answer += TOOL_END_SEQUENCE + " " + str(response_tool) + "\n"
                except Exception as e:
                    print("Error in Aggregator:", e)
                    prompt.append({"role": "assistant", "content": TOOL_END_SEQUENCE})
                    answer += TOOL_END_SEQUENCE
            else:
                prompt.append({"role": "assistant", "content": TOOL_END_SEQUENCE})
                answer += TOOL_END_SEQUENCE

        response_res.append(answer)
        final_answer = extract_finalanswer(answer, model_path1, client1)
        final_answers.append(final_answer)

    # save results
    # df = pd.DataFrame({'output': response_res, 'final_answer': final_answers, 'ground_truth': ground_truth})
    df = pd.DataFrame({'output': final_answers, 'ground_truth': ground_truth})
    df.to_json(res_path,orient='records',lines=True)

