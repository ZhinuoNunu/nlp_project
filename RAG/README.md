# Retrieval-Agumented Generation


## Introduction
This section pertains to the project's Retrieval-Augmented Generation (RAG), led by Linbing Xiang. 

I first contruct a database using sqlite3.

And then use a small model from huggingface to generate SQL query based on the `question`, `columns_used` and `dataset_name`.

Third, using SQL to query the database to get the `context` related to the problem, accompanied by a short description of the context.

Finally, I format `question`, `columns_used`, `dataset`, `SQL`, `context`, `context_description` togather to inference with baseline model and LoRA fine-tuned model.

## The code structure is as follows:
```
|-- build_sqlite3.py                    # construct database
|-- SQLGeneration.py                    # use a model to generate SQL
|-- retrievalFomDB.py                   # generate relevant context from database using SQL, accompanied by a description.
|-- qa_data.py                          # data process
|-- infer_llama3_vllm_RAG.sh            # bash script for large model inference with vllm
|-- infer_llama3_vllm_RAG.py            # large model inference with baseline model also with LoRA
|-- eval.py                             # evaluate python code to get accuracy and f1 score
|-- eval_BLEU.py                        # evaluation with BLEU
|-- data/
    |-- data_w_sql.json                 # data format with SQL obtained from SQLGeneration.py
    |-- retrievaled_data(latin-1).json  # retrieved context accompanied by description obtained from retrievalFromDB.py
    |-- output_baseline_rag.jsonl       # final output from baseline model inference
    |-- output_baseline_rag.jsonl       # final output from LoRA model inference
```