import os
import json

from datasets import load_dataset, load_from_disk
from pandas import read_parquet

'''
# download the dataset from huggingface

# # Load all QA pairs
# all_qa = load_dataset("cardiffnlp/databench", name="qa", split="train")
# all_qa.save_to_disk('/home/ubuntu/6000N/5018/databench/all_qa')
# # Load SemEval 2025 task 8 Question-Answer splits
# semeval_train_qa = load_dataset("cardiffnlp/databench", name="semeval", split="train")
# semeval_dev_qa = load_dataset("cardiffnlp/databench", name="semeval", split="dev")
# semeval_train_qa.save_to_disk('/home/ubuntu/6000N/5018/databench/train')
# semeval_dev_qa.save_to_disk('/home/ubuntu/6000N/5018/databench/test')
'''

'''
# data format of llama train/infer
[
    {
    "instruction": "Describe a process of making crepes.",
    "input": "",
    "output": "Making crepes is an easy and delicious process! Here are step-by-step instructions on how to make them:\n\n1. Assemble your ingredients. For basic crepes, you'll need: 1 cup all-purpose flour, 2 eggs, 1/2 cup milk, 1/2 cup water, 1/4 teaspoon salt, and 2 tablespoons melted butter.\n\n2. Mix the batter: In a large mixing bowl, whisk together the flour and the eggs. Gradually add the milk and water, stirring constantly to ensure that there are no lumps. Add salt and melted butter, and mix well.\n\n3. Let the batter rest: If you can, let the batter sit for an hour or so. This will help the flour to absorb the liquid and make the crepes more tender.\n\n4. Heat your pan: Preheat a non-stick pan over medium heat. Lightly butter the pan or use cooking spray to prevent the crepes from sticking.\n\n5. Pour the batter: Using a ladle or a measuring cup, pour a small amount of batter (about 1/4 cup) onto the center of the pan. Immediately tilt the pan in a circular motion to spread the batter evenly and thinly over the bottom of the pan.\n\n6. Cook the crepe: Cook the crepe for 1-2 minutes until the bottom is lightly golden. Carefully loosen the edges with a spatula and flip the crepe over to cook the other side for another minute.\n\n7. Remove and repeat: Gently slide the crepe onto a plate, and then repeat the process with the remaining batter. Remember to re-butter the pan between each crepe if necessary.\n\n8. Fill and serve: Fill your cooked crepes with your desired filling, such as fresh fruit, whipped cream, Nutella, or ham and cheese. Roll or fold, and serve immediately. Enjoy!"
  },]
'''

'''
# data format of databench
{'question': 'Is the person with the highest net worth self-made?', 'answer': 'True', 'type': 'boolean', 'columns_used': ['finalWorth', 'selfMade'], 'column_types': ['number[uint32]', 'boolean'], 'sample_answer': 'False', 'dataset': '001_Forbes'}
'''


train_path = "/home/ubuntu/6000N/5018/databench/train"
dataset_path = "/home/ubuntu/6000N/5018/databench/dataset/data"
save_path = "/home/ubuntu/6000N/llama_factory/LLaMA-Factory-main/nlp_data/train_data.json"
semeval_train_qa = load_from_disk(train_path)

data_json = []
for i in range(len(semeval_train_qa)):
    dataset_name = semeval_train_qa[i]['dataset']
    table_path = os.path.join(dataset_path, dataset_name, 'all.parquet')
    table = read_parquet(table_path)
    qa_data = {}
    if semeval_train_qa[i]['columns_used']:  # 'columns_used' is not NoneType
        for col in semeval_train_qa[i]['columns_used']:
            if col in table:
                qa_data[col] = table[col].to_list()
    
    # print(qa_data)
    data_tmp = {
        "instruction": "Combine the infomation in the table to answer the question. The table is: "+str(qa_data),
        "input": semeval_train_qa[i]['question'],
        "output": semeval_train_qa[i]['answer']
    }
    # print(data_tmp)
    data_json.append(data_tmp)
    if i%100 == 0:
        print(f"Nmber {i} finished !")


with open(save_path,"w") as f:
    json.dump(data_json,f)