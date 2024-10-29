import sqlite3
import json
import re



# Generate a short description of the results obtained for the retrieval of the database
def extract_select_fields(sql_query):
    # Converts the query to lower case and splits it into a list of words
    words = sql_query.lower().split()
    
    # choose words between "SELECT" and "FROM".
    select_index = words.index('select') + 1
    from_index = words.index('from')
    selected_fields = words[select_index:from_index]
    
    # Split by comma
    fields = ' '.join(selected_fields).split(',')

    return [field.strip() for field in fields]

# This function is used to put quotes around column and table names
def quote_identifiers(sql_query):
    # Use expressions to match parts that contain underscores or hyphens
    # make sure don't match independent < or >
    pattern = re.compile(r'\b[a-zA-Z0-9_\-<>:]+\b')
    keywords = {
        "SELECT", "FROM", "WHERE", "ORDER", "GROUP", "BY", "LIMIT", "JOIN", 
        "INNER", "LEFT", "RIGHT", "ON", "ASC", "DESC", "AND", "OR", "NOT", 
        "NULL", "COUNT", "AVG", "SUM", "MIN", "MAX", "AS", "IS", "NULLS", "LAST"
    }

    def replace(match):
        text = match.group(0)
        if  not text.startswith('"') and not text.endswith('"') and text not in keywords:
            start_index = match.start(0)
            end_index = match.end(0)
            # if the preceding and following characters are not quotation marks, the word is quoted
            if (sql_query[start_index-1] not in ('"', '.')) and ( sql_query[end_index] not in ('"', '.')):
                return f'"{text}"'
        return text

    quoted_sql = pattern.sub(replace, sql_query)
    return quoted_sql

def quote_select_identifiers(sql_query):
    # 正则表达式匹配 SELECT 和 FROM 之间的标识符
    # 不匹配已被引号包围的部分
    pattern = re.compile(r'(?<=SELECT\s)(.*?)(?=\sFROM)', re.IGNORECASE)

    # 替换函数
    def replace(match):
        fields = match.group(0)
        # 用逗号分隔多个字段
        parts = fields.split(',')
        # 处理每个字段，给标识符加引号
        quoted_parts = []
        for part in parts:
            # 如果没有引号，则添加引号
            if not part.startswith('"') and not part.endswith('"'):
                quoted_parts.append(f'"{part}"')
            else:
                quoted_parts.append(part)
        return ', '.join(quoted_parts)

    quoted_sql = pattern.sub(replace, sql_query)

    return quoted_sql

def convert_bytes_to_str(results):
    """Converts the bytes type in the query result to a string"""
    results = results[:200]
    # 限制字段的字符数
    if isinstance(results, str) and len(results) > 2000:
        results = results[:2000] + '...'
    if isinstance(results, list):
        # 处理多行的情况，逐行检查
        return [
            tuple(item.decode('latin-1') if isinstance(item, bytes) else item for item in row)
            for row in results
        ]
    elif isinstance(results, tuple):
        # 处理单行的情况
        return tuple(item.decode('latin-1') if isinstance(item, bytes) else item for item in results)
    return results

def main():
    file_path = '/root/nlp_project/RAG/data_w_sql.json'

    # Load JSON data from the specified file
    with open(file_path, 'r') as file:
        data = json.load(file)

    # Connect to the SQLite database
    conn = sqlite3.connect('/root/nlp_project/databench/dataset.sqlite3')

    # Create a cursor object using the cursor() method
    cursor = conn.cursor()

    for item in data:
        # Execute the SQL query
        print(f"executing sql: {item["sql"]}")

        try:
            cursor.execute(item["sql"])
            # Fetch all rows from the last executed query
            results = cursor.fetchall()
            # Extract selected fields from the SQL query
            selected_columns = extract_select_fields(item["sql"])
        except sqlite3.OperationalError as e:
            print(f"Error occurred: {e}. Attempting to fix SQL query...")
            processed_sql = quote_identifiers(item["sql"])
            print(f"processed sql: {processed_sql}")
            try:
                cursor.execute(processed_sql)
                results = cursor.fetchall()
                selected_columns = extract_select_fields(processed_sql)

            except sqlite3.Error as fallback_error:
                # If an error occurs, execute a fallback query
                print(f"Fallback error occurred: {fallback_error}. Using fallback query...")
                selected_columns = ', '.join(item["columns_used"])
                fallback_sql = f"SELECT {selected_columns} FROM '{item['dataset']}'"
                process_fallback_sql = quote_select_identifiers(fallback_sql)
                print(f"fallback_sql: {fallback_sql}")
                cursor.execute(process_fallback_sql)
                item["sql"] = process_fallback_sql
                result = cursor.fetchall()
                result = result[:200]
                print(f"len result: {len(result)}")
                results = f"Please make your own results based on the context of the table. Table data: {result}."

        item["results"] = convert_bytes_to_str(results)
        item["result_description"] = str(selected_columns)
    
    cursor.close()
    output_file_path = '/root/nlp_project/RAG/retrievaled_data(latin-1).json'
    # Save the updated JSON data to a new file
    with open(output_file_path, 'w') as file:
        json.dump(data, file, indent=2)
    print(f"Updated data has been saved to {output_file_path}")

if __name__ == "__main__":
    main()