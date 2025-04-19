import json
import os


def get_table_and_question(data_file: str, output_dir: str):
    # 确保输出目录存在
    os.makedirs(output_dir, exist_ok=True)

    # 读取输入文件
    with open(data_file, "r", encoding='utf-8') as f:
        data = json.load(f)
        tables = data["db_infos"]
        question = data["question"]
        # question = translate_text(data["user_question"])
        utterance = [{"utterance": question}]

    # 转换表结构为 schema
    tables = transform_to_schema(tables)

    # 写入 tables.json 文件
    tables_output_file = os.path.join(output_dir, "tables.json")
    with open(tables_output_file, "w", encoding='utf-8') as f:
        json.dump(tables, f, ensure_ascii=False)

    # 写入 question.json 文件
    question_output_file = os.path.join(output_dir, "question.json")
    with open(question_output_file, "w", encoding='utf-8') as f:
        json.dump(utterance, f, ensure_ascii=False)


# 原来的 transform_to_schema 函数
def transform_to_schema(data):
    for i, db in enumerate(data):
        schema = []
        db_name = db["db_id"]
        for j, table_name in enumerate(db["table_names"]):
            table_schema = f"{db_name}.{table_name}("
            columns = [column[1]
                       for column in db["column_names"] if column[0] == j]
            column_name = ", ".join(columns)
            table_schema += column_name + ")"
            schema.append(table_schema)
        data[i]["schema"] = schema
    return data


# 如果你想直接运行这个脚本，可以调用下面的主程序入口
if __name__ == '__main__':
    data_file_path = "/home/kyxu/text2sql/cache/data.json"
    output_directory = "./data_test"

    get_table_and_question(data_file_path, output_directory)
