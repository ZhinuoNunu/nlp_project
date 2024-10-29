import os
import pandas as pd
import sqlite3

# SQLite connection，数据库保存在指定路径
db_path = "/root/nlp_project/databench/dataset.sqlite3"  # SQLite数据库文件
conn = sqlite3.connect(db_path)  # 建立SQLite连接

def create_table(table_name, dataframe):
    """
    创建SQLite表，并定义字段类型
    :param table_name: 表名
    :param dataframe: 要创建表的DataFrame
    """
    # 保留原始列名，并检测大小写不同的列名冲突
    unique_columns = []
    seen_columns = set()
    for col in dataframe.columns:
        # 将列名转为小写以检测冲突
        col_lower = col.lower()
        if col_lower not in seen_columns:
            unique_columns.append(col)
            seen_columns.add(col_lower)
            

    # 动态生成SQL语句，用于创建表
    print(table_name)
    columns = ', '.join([f'"{col}" TEXT COLLATE BINARY' for col in unique_columns])  # 假设所有字段为TEXT类型
    print(columns)
    create_table_query = f'CREATE TABLE IF NOT EXISTS "{table_name}" ({columns})'
    
    # 执行创建表的SQL语句
    conn.execute(create_table_query)
    conn.commit()

def process_parquet_file(file_path, dataset_name):
    """
    处理parquet文件，创建相应的SQLite表并插入数据
    :param file_path: parquet文件路径
    :param dataset_name: 表名
    """
    df = pd.read_parquet(file_path)

    # 创建对应的表
    create_table(dataset_name, df)

    # 将DataFrame中的数据插入SQLite表
    df.to_sql(dataset_name, conn, if_exists='append', index=False)  # 将数据插入表中

def process_all_datasets(root_directory):
    """
    遍历指定目录下的所有parquet文件，并处理它们
    :param root_directory: 根目录，包含所有数据集
    """
    for dataset_name in os.listdir(root_directory):
        dataset_path = os.path.join(root_directory, dataset_name, "all.parquet")
        if os.path.exists(dataset_path):
            process_parquet_file(dataset_path, dataset_name)

if __name__ == "__main__":
    root_directory = "/root/nlp_project/databench/dataset"
    process_all_datasets(root_directory)
    print("数据已插入SQLite数据库。")

    # 关闭SQLite连接
    conn.close()