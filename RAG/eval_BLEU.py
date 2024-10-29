from nltk.translate.bleu_score import sentence_bleu
from nltk.translate.bleu_score import SmoothingFunction
from sklearn.metrics import accuracy_score, f1_score
import json

def read_jsonl(file_path):
    data = []
    with open(file_path, 'r') as file:
        for line in file:
            data.append(json.loads(line))
    return data

def calculate_bleu(output, ground_truth):
    # 预处理输出和参考文本
    # 这里假设输出和真实值都是字符串
    # 将字符串分割成词汇列表
    reference = [ground_truth.split()]
    candidate = output.split()
    
    # 使用NLTK的sentence_bleu计算BLEU分数
    # SmoothingFunction可以处理候选句子和参考句子很短时的特殊情况
    smoothing = SmoothingFunction().method1
    score = sentence_bleu(reference, candidate, smoothing_function=smoothing)
    return score

def evaluate_predictions(data):
    bleu_scores = []
    labels = []
    for item in data:
        bleu_score = calculate_bleu(item['output'], item['ground_truth'])
        bleu_scores.append(bleu_score)
        # 假设阈值为0.5
        label = 1 if bleu_score > 0.05 else 0
        labels.append(label)

    # 真实的标签，我们假设所有输入的ground_truth都是正确的，即所有的标签都是1
    true_labels = [1] * len(labels)
    
    # 计算accuracy和f1 score
    accuracy = accuracy_score(true_labels, labels)
    f1 = f1_score(true_labels, labels)
    
    return accuracy, f1, bleu_scores


data = read_jsonl('/root/nlp_project/RAG/output_baseline_rag.jsonl')
accuracy, f1, bleu_scores = evaluate_predictions(data)
print(f"accuracy, f1, bleu: {accuracy}, {f1}, {bleu_scores}")

# accuracy : 0.222(baseline),  0.397(LoRA)
