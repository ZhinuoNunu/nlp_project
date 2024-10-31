# Chain-of-Thought

## Introduction

This part aims to perform prompt engineering on the  model to enable it to operate in a Chain-of-Thought (CoT) manner, led by Meisi Fu.

## Implementation

**Dataset Preparation** : Load the test dataset `test_data.json`, which contains questions and corresponding answers that the model needs to address.

**Define Prompt Templates** : Use the content from `few_shot.txt` and `extract_few_shot.txt` to construct prompt templates for the model input, guiding the model to follow the specified format and logical steps in its responses.

**Model Inference** : Use the script `infer_llama3_vllm_CoT.py` to perform multi-turn inference with the `llama3-8b` model. As the model generates responses, it may invoke custom tools (e.g., `Sorter`, `Filter`, `Calculator`, `Aggregator`) to process data.

**Tool Invocation and Execution** : Parse the tool invocation commands from the model's responses and execute the corresponding tool functions defined in `tools.py`, then feed the results back to the model.

**Extract Final Answers** : Use the examples in `extract_few_shot.txt` and the `extract_finalanswer` function to extract the final answer from the model's complete response.

## Code Structure
```
|-- prompt/
    |-- few_shot.txt                    # templates about how to use tools and how to think step by stpe
    |-- extract_few_shot.txt            # templates about how to extract the final answer from the deduction process
|-- infer_llama3_vllm_CoT.sh            # bash script for large model inference with vllm
|-- infer_llama3_vllm_CoT.py            # large model inference with baseline model also with LoRA
|-- eval.py                             # evaluate python code to get accuracy and f1 score
|-- eval_BLEU.py                        # evaluation with BLEU
|-- data/
    |-- output_cot_baseline.jsonl       # final output from baseline model inference
    |-- output_cot_lora.jsonl           # final output from LoRA model inference
```