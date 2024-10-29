import pandas as pd
from sklearn.metrics import accuracy_score, recall_score, precision_score, f1_score


df = pd.read_csv('/root/nlp_project/RAG/output_lora_rag.csv', encoding='latin-1')
label = df['label'].to_list()
print(f"label:{label}")
gt = [1 for i in range(len(label))]

acc, f1s = accuracy_score(gt, label), f1_score(gt, label)
print(acc, f1s)