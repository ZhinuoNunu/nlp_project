from pandas import read_parquet # type: ignore
import json
from datasets import load_from_disk

table_path = "./databench/dataset/051_Pokemon/all.parquet"
table = read_parquet(table_path)
csv_path = "./RAG/051_Pokemon_data.csv"
table.to_csv(csv_path, index=False)      


# test_path = "/root/nlp_project/databench/train"
# semeval_test_qa = load_from_disk(test_path)
# data = []
# for i in range(len(semeval_test_qa)):
#     data.append({
#         "question": semeval_test_qa[i]['question'],
#         "answer": semeval_test_qa[i]['answer'],
#         "dataset": semeval_test_qa[i]['dataset'],
#         "columns_used": semeval_test_qa[i]['columns_used']
#     })

# with open("/root/nlp_project/RAG/train_qa.json","w") as f:
#     json.dump(data,f)