
import sqlite3
from typing import List, Any
# from utils.print_utils import print_debug
import json
import glob
import os

# 将原始名称转换为小写并用空格替换下划线
def convert_original_names(ori_names: List[Any]) -> List[Any]:
    names = []
    for ori_name in ori_names:
        # table_names_original
        if isinstance(ori_name, str):
            name = ori_name.lower().replace('_', ' ')
            names.append(name)
        # column_names_original
        else:
            table_id, original_name = ori_name
            name = original_name.lower().replace('_', ' ')
            names.append([table_id, name])
    return names


# 将列类型转换为更通用的类型
def convert_types(types: List[str]) -> List[str]:
    type_mapping = {
        "text": "text",
        "char": "text",
        "int": "number",
        "real": "number",
        "decimal": "number",
        "double": "number",
        "bit": "number",
        "float": "number",
        "numeric": "number",
        "datetime": "time",
        "timestamp": "time",
        "year": "time",
        "bool": "boolean"
    }
    return [
        next((target_type for keyword, target_type in type_mapping.items() if keyword in t), "others")
        for t in types
    ]

# 从数据库中提取信息
def extract_db_info(database_path: str, db_name: str) -> dict:
    # 连接到数据库
    conn = sqlite3.connect(database_path)
    cursor = conn.cursor()

    # 获取数据库中的表名
    cursor.execute("SELECT name FROM sqlite_master WHERE type='table';")
    tables = cursor.fetchall()

    db_info = {
        "column_names": [[-1, "*"]],
        "column_names_original": [[-1, "*"]],
        "column_types": ["text"],
        "foreign_keys": [],
        "primary_keys": [],
        "table_names": [],
        "table_names_original": [],
        "db_id": db_name
        # "db_id": os.path.basename(database_path).split('.')[0]  # 使用文件名作为 db_id
    }

    db_info["table_names_original"] = [table[0] for table in tables]
    db_info["table_names"] = convert_original_names(db_info["table_names_original"])

    # 遍历每个表来获取详细信息
    for ti, table_name in enumerate(tables):
        table_name = table_name[0]

        # 获取列信息
        cursor.execute(f"PRAGMA table_info({table_name});")
        columns = cursor.fetchall()
        # print_debug(columns=columns)

        col_names_ori = [[ti, column[1]] for column in columns]
        col_names = convert_original_names(col_names_ori)
        col_types = [column[2].lower() for column in columns]
        col_types = convert_types(col_types)
        db_info["column_names_original"].extend(col_names_ori)
        db_info["column_names"].extend(col_names)
        db_info["column_types"].extend(col_types)
        # 列信息和主键信息
        for column in columns:
            # print(column)
            column_id, column_name, column_type, not_null, default_value, pk = column
            if pk == 1:
                # print(column)
                for ci, col in enumerate(db_info["column_names_original"]):
                    if col[0] == ti and col[1] == column_name:
                        db_info["primary_keys"].append(ci)
                        

    for ti, table_name in enumerate(tables):
        table_name = table_name[0]
        # 获取外键信息
        cursor.execute(f"PRAGMA foreign_key_list({table_name});")
        foreign_keys = cursor.fetchall()
        # print(f"foreign_keys: {foreign_keys}")
        
        for foreign_key in foreign_keys:
            try:
                _, _, table, from_col, to_col, _, _, _ = foreign_key
                # print(f"foreign_key: {foreign_key}, table: {table_name}")
                tid_1 = ti
                tid_2 = db_info["table_names_original"].index(table)
                from_col_id = db_info["column_names_original"].index([tid_1, from_col])
                to_col = db_info["column_names_original"].index([tid_2, to_col])
                db_info["foreign_keys"].append([from_col_id, to_col])
            except Exception as e:
                print(f"Error: {e}")

    # 关闭数据库连接
    conn.close()

    return db_info

# 比较两个字典是否相等
def compare_dicts(d1, d2):
    # 检查两个字典的键集合是否相同
    if d1.keys() != d2.keys():
        print(f"keys not equal: {d1.keys()} != {d2.keys()}")
        return False
    
    # 检查每个键的对应值是否相等
    for key in d1:
        if d1[key] != d2[key]:
            print(f"values not equal: {d1[key]} !=\n {d2[key]}")
            return False
    
    return True

# 递归查找指定目录中的所有 SQLite 文件
def find_sqlite_files(directory: str) -> list:
    # 使用 glob 模式匹配 .sqlite 文件，递归搜索
    pattern = os.path.join(directory, '**', '*.sqlite')
    sqlite_files = glob.glob(pattern, recursive=True)
    return sqlite_files

# 从所有 SQLite 文件中提取数据库信息
def extract_db_infos(sqlite_files: list) -> list:
    db_infos = []
    for sqlite_file in sqlite_files:
        print(f"Extracting database info from {sqlite_file}")
        db_name = os.path.basename(sqlite_file).split('.')[0]
        db_info = extract_db_info(sqlite_file, db_name)
        db_infos.append(db_info)
    return db_infos


if __name__ == "__main__":
    # # database_path = "perpetrator.sqlite"
    # database_path = "E:/text2sql/demo/dataset/Spider/database/academic/academic.sqlite"
    # with open("../dataset/Spider/tables.json", "r") as table_file:
    #     tables = json.load(table_file)
    # result = extract_db_info(database_path)
    # gold = [table for table in tables if table["db_id"] == result["db_id"]][0]
    # # for key, value in result.items():
    # #     print(f"{key}: {value}")
    # # for key, value in gold.items():
    # #     print(f"{key}: {value}")
    # equal = compare_dicts(result, gold)
    # print(equal)

    db_path = "../dataset/Spider/database"
    sqlite_files = find_sqlite_files(db_path)
    # print_debug(sqlite_files=sqlite_files)
    db_infos = extract_db_infos(sqlite_files)
    with open("../dataset/Spider/tables.json", "r") as table_file:
        tables = json.load(table_file)
    # print_debug(db_infos=db_infos)
    for db_info in db_infos:
        # print(f"db_id: {db_info['db_id']}")
        gold = [table for table in tables if table["db_id"] == db_info["db_id"]][0]
        # print_debug(gold=gold)
        equal = compare_dicts(db_info, gold)
        if not equal:
            print(f"db_id: {db_info['db_id']}")
            print()

    # types = []
    # for db_info in db_infos:
    #     for type in db_info["column_types"]:
    #         if type not in types:
    #             types.append(type)
    # print(types)