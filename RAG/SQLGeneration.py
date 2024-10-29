import torch
import os
from tqdm import tqdm
import json
import pandas as pd
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline
from datasets import load_from_disk
print("---")
print(torch.cuda.is_available())  # 如果输出为True，说明GPU可用
print(torch.cuda.device_count())  # 查看可用的GPU数量
print(torch.cuda.get_device_name(0))  # 获取GPU的名称

# os.environ["CUDA_VISIBLE_DEVICES"] = "0"
model_path = '/root/nlp_project/sqlcoder-7b-2'


# generate SQL based on prompt template
prompt_template = """
    You are an SQL generation assistant. You will be provided with a question, a dataset (table name), and columns used. 
    You will output a corresponding SQL query in SQLite format. Do not include any additional text or explanations.

    The SQL query should:
    1. Select the relevant columns from the table "dataset".
    2. If the question involves filtering, apply the appropriate conditions.
    3. If aggregation (e.g., counting, averaging) or sorting is required based on the question, apply GROUP BY, ORDER BY, and filtering clauses as needed.
    4. If there are ranking or limiting requirements (e.g., top/bottom records), use LIMIT.

    ### When using SQLite3, ensure to enclose column names with spaces or special characters (such as hyphens, underscores) in double quotes, e.g., for :columns_used:["Openness to Change"], SQL should be SELECT \"Openness to Change\".

    ### Output the SQL query only. Do not include any additional text or explanations. Just output the SQL query. Just like output in the example below, outputs the SQL statement, do not need to include input.

    Example 1:
    "question": "Does the youngest billionaire identify as male?", 
    "dataset": "001_Forbes", 
    "columns_used": ["age", "gender"]
    
    Output:
    SELECT age, gender
    FROM "001_Forbes"
    WHERE age = (SELECT MIN(age) FROM "001_Forbes");

    Example 2:
    "question": "Which are the top 6 clubs in terms of the average potential score of their players?", 
    "dataset": "007_Fifa", 
    "columns_used": ["Club<gx:category>", "Potential<gx:number>"]
    
    Output:
    SELECT "Club<gx:category>", AVG("Potential<gx:number>") as avg_potential
    FROM "007_Fifa"
    GROUP BY "Club<gx:category>"
    ORDER BY avg_potential DESC
    LIMIT 6;

    Example 3:
    "question": "List the bottom 3 departments by the number of promotions in the last 5 years.", 
    "dataset": "018_Staff", 
    "columns_used": ["Department", "Promoted in the last 5 years?"]
    
    Output:
    SELECT Department, COUNT(*) as num_promotions
    FROM "018_Staff"
    WHERE "Promoted in the last 5 years?" = 'Yes'
    GROUP BY Department
    ORDER BY num_promotions ASC
    LIMIT 3;
    """

def generate_sql_prompt(question, dataset, columns_used):
    prompt = f"""
    Generate an SQL query based on the following information:

    Question: {question}

    - Dataset (table name): {dataset}
    - Columns to select: {columns_used}

    The SQL query should:
    1. Select the relevant columns from the table `"{dataset}"`.
    2. If the question involves filtering, apply the appropriate conditions.
    3. If aggregation (e.g., counting, averaging) or sorting is required based on the question, apply `GROUP BY`, `ORDER BY`, and filtering clauses as needed.
    4. If there are ranking or limiting requirements (e.g., top/bottom records), use `LIMIT`.

    ### Output the SQL query only. Do not include any additional text or explanations. Just output the SQL query. Just like output in the example below, outputs the SQL statement, do not need to include input.

    ### SQL Output Example:

    1. Input:
    "question": "Does the youngest billionaire identify as male?", 
    "dataset": "001_Forbes", 
    "columns_used": ["age", "gender"]
    
    Output:
    SELECT age, gender
    FROM "001_Forbes"
    WHERE age = (SELECT MIN(age) FROM "001_Forbes");

    2. Input:
    "question": "Which are the top 6 clubs in terms of the average potential score of their players?", 
    "dataset": "007_Fifa", 
    "columns_used": ["Club<gx:category>", "Potential<gx:number>"]
    
    Output:
    SELECT "Club<gx:category>", AVG("Potential<gx:number>") as avg_potential
    FROM "007_Fifa"
    GROUP BY "Club<gx:category>"
    ORDER BY avg_potential DESC
    LIMIT 6;

    3. Input:
    "question": "List the bottom 3 departments by the number of promotions in the last 5 years.", 
    "dataset": "018_Staff", 
    "columns_used": ["Department", "Promoted in the last 5 years?"]
    
    Output:
    SELECT Department, COUNT(*) as num_promotions
    FROM "018_Staff"
    WHERE "Promoted in the last 5 years?" = 'Yes'
    GROUP BY Department
    ORDER BY num_promotions ASC
    LIMIT 3;

    """
    return prompt


# load model and tokenizer
def get_tokenizer_model(model_directory):
    tokenizer = AutoTokenizer.from_pretrained(model_directory)
    model = AutoModelForCausalLM.from_pretrained(
        model_directory,
        trust_remote_code=True,
        torch_dtype=torch.float16,
        device_map="auto",
        use_cache=True,
    )
    # device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    # model.to(device)
    return tokenizer, model

def generate_sql(question, dataset, columns_used, model, tokenizer):
    prompt = f"""
        {prompt_template}

        Input:
        "question": "{question}", 
        "dataset": "{dataset}", 
        "columns_used": {columns_used}
        
        Output:
    """
    eos_token_id = tokenizer.eos_token_id
    # model = model.to("cuda")

    pipe = pipeline(
        'text-generation',
        model=model,
        tokenizer=tokenizer,
        max_new_tokens=150,
        do_sample=False,
        return_full_text=False, # added return_full_text parameter to prevent splitting issues with prompt

    )
    generated_query = (
        pipe(
            prompt,
            num_return_sequences=1,
            eos_token_id=eos_token_id,
            pad_token_id=eos_token_id,
        )[0]["generated_text"]
        # .split(";")[0]
        # .split("```")[0]
        # .strip()
        # + ";"
    )
    print(f"--generated_query:{generated_query}")
    return generated_query

def main():
    model_path = "/root/nlp_project/sqlcoder-7b-2"
    tokenizer, model = get_tokenizer_model(model_path)
    test_path = "/root/nlp_project/databench/test"
    semeval_test_qa = load_from_disk(test_path)
    data_sql = []
    # Open a file to write the SQL queries
    output_file_path = "/root/nlp_project/RAG/output.sql"
    os.makedirs(os.path.dirname(output_file_path), exist_ok=True)
    with open(output_file_path, 'w') as file:
        for i in tqdm(range(len(semeval_test_qa)), desc="Generating SQL Queries"):

            dataset_name = semeval_test_qa[i]['dataset']
            question = semeval_test_qa[i]['question']
            columns_used = semeval_test_qa[i]['columns_used']
            sql_query = generate_sql(question,dataset_name,columns_used,model,tokenizer)
            
            data_saved = {
                "question": question,
                "dataset": dataset_name,
                "answer": semeval_test_qa[i]['answer'],
                "columns_used": columns_used,
                "sql": sql_query
            }
            data_sql.append(data_saved)

            # Write each SQL query to the file, followed by a newline
            file.write(sql_query + "\n")
 
    with open("/root/nlp_project/RAG/data_w_sql.json", 'w') as file:
        json.dump(data_sql,file)


if __name__ == "__main__":
    main()
