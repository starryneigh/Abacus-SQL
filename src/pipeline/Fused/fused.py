import os 
import json
import sqlite3
from typing import List, Dict, Any
from .converse import pro_converse
from .cluster import pro_cluster
from .initialize import pro_initialize
from .generate_sql import pro_generate_sql
from .generate_question import pro_generate_question
from .filt import pro_filt
from .synthesis import to_demo_file
from ..utils.my_logger import MyLogger
from ..utils.stream_handler import StreamDataHandler

logger = MyLogger("fused", "logs/fused.log")


def merge_json_files_in_folder(folder_path, output_file=None, recursive=False):
    """
    打开指定文件夹中的所有 JSON 文件，并将它们的内容合并成一个列表。
    
    :param folder_path: 文件夹的路径
    :param output_file: 可选的输出文件路径，如果提供，会将合并的内容写入该文件
    :param recursive: 是否递归读取子文件夹中的文件，默认为 True
    :return: 合并后的 JSON 列表
    """
    merged_data = []  # 用于存放所有 JSON 文件的数据
    
    if recursive:
        # 递归遍历文件夹及其子文件夹中的所有文件
        for root, dirs, files in os.walk(folder_path):
            for file in files:
                # 检查文件扩展名是否是 .json
                if file.endswith('.json'):
                    if file_path == output_file:
                        continue
                    file_path = os.path.join(root, file)
                    print(f"正在读取文件: {file_path}")
                    
                    # 打开并读取 JSON 文件内容
                    with open(file_path, 'r', encoding='utf-8') as f:
                        try:
                            data = json.load(f)  # 读取 JSON 文件
                            if isinstance(data, list):
                                merged_data.extend(data)  # 合并列表
                            else:
                                merged_data.append(data)  # 合并单个对象
                        except json.JSONDecodeError:
                            print(f"文件 {file_path} 不是有效的 JSON 格式。跳过此文件。")
    else:
        # 非递归，仅读取指定文件夹中的文件
        for file in os.listdir(folder_path):
            file_path = os.path.join(folder_path, file)
            if os.path.isfile(file_path) and file.endswith('.json'):
                if file_path == output_file:
                    continue
                print(f"正在读取文件: {file_path}")
                
                # 打开并读取 JSON 文件内容
                with open(file_path, 'r', encoding='utf-8') as f:
                    try:
                        data = json.load(f)  # 读取 JSON 文件
                        if isinstance(data, list):
                            merged_data.extend(data)  # 合并列表
                        else:
                            merged_data.append(data)  # 合并单个对象
                    except json.JSONDecodeError:
                        print(f"文件 {file_path} 不是有效的 JSON 格式。跳过此文件。")
    
    # 如果指定了输出文件，则将合并后的内容写入文件
    if output_file:
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(merged_data, f, indent=4, ensure_ascii=False)
        print(f"合并后的内容已保存到: {output_file}")
    
    # 返回合并后的 JSON 列表
    return merged_data


def extract_sqlite_metadata_from_folder(folder_path: str) -> List[Dict[str, Any]]:
    data = []

    # 遍历所有子文件夹及其中的文件
    for root, _, files in os.walk(folder_path):
        for file_name in files:
            if file_name.endswith('.sqlite'):
                db_id = file_name.split('.')[0]
                db_path = os.path.join(root, file_name)
                
                # 连接SQLite数据库
                conn = sqlite3.connect(db_path)
                cursor = conn.cursor()
                
                # 获取表名
                cursor.execute("SELECT name FROM sqlite_master WHERE type='table';")
                table_names = [row[0] for row in cursor.fetchall()]
                
                column_names = [[-1, "*"]]
                column_types = []
                foreign_keys = []
                primary_keys = []
                
                for table_index, table_name in enumerate(table_names):
                    # 获取列名和类型
                    cursor.execute(f"PRAGMA table_info('{table_name}');")
                    columns_info = cursor.fetchall()
                    
                    table_column_names = [(table_index, col[1]) for col in columns_info]
                    table_column_types = [col[2] for col in columns_info]
                    column_names.extend(table_column_names)
                    column_types.extend(table_column_types)
                    
                    # 获取主键
                    table_primary_keys = [col[0] for col in columns_info if col[5] == 1]
                    if len(table_primary_keys) > 0:
                        primary_keys.extend(table_primary_keys)
                    else:
                        primary_keys.extend([None])
                    
                    # 获取外键
                    cursor.execute(f"PRAGMA foreign_key_list('{table_name}');")
                    foreign_key_info = cursor.fetchall()
                    
                    # 通过列名找到对应的列序号
                    table_foreign_keys = []
                    for fk in foreign_key_info:
                        # 获取外键列序号
                        fk_column_name = fk[3]
                        fk_column_index = next((col[0] for col in columns_info if col[1] == fk_column_name), None)
                        
                        # 获取引用列序号
                        ref_table_name = fk[2]
                        ref_column_name = fk[4]
                        cursor.execute(f"PRAGMA table_info('{ref_table_name}');")
                        ref_columns_info = cursor.fetchall()
                        ref_column_index = next((col[0] for col in ref_columns_info if col[1] == ref_column_name), None)
                        
                        if fk_column_index is not None and ref_column_index is not None:
                            table_foreign_keys.append((fk_column_index, ref_column_index))
                    if len(table_foreign_keys) > 0:
                        foreign_keys.extend(table_foreign_keys)
                
                # 组织数据
                data.append({
                    "db_id": db_id,
                    "table_names": table_names,
                    "table_names_original": table_names,
                    "column_names": column_names,
                    "column_names_original": column_names,
                    "column_types": column_types,
                    "primary_keys": primary_keys,
                    "foreign_keys": foreign_keys
                })
                
                # 关闭数据库连接
                conn.close()
    data.sort(key=lambda x: x["db_id"])
    
    return data

def yield_process(turn, step, total_turn, total_step):
    num_per_turn = 100 / total_turn
    num_per_step = num_per_turn / total_step
    i = (turn * num_per_turn + step * num_per_step) / 100
    # print(f"i = {i}")
    yield f"data: {json.dumps({'process': i})}\n\n"

def pro_fused(to_fused_path, cache_path, encoder_name_or_path, table_path, 
              database_path, stream_handler, to_path, 
              num_turn=2, cluster_number=4, test=False
):
    """
    to_fused_path: str, path to the fused data
    cache_path: str, path to the cache data
    encoder_name_or_path: str, path to the encoder model
    table_path: str, path to the table data
    database_path: str, path to the database
    stream_handler: StreamDataHandler, stream handler for the model
    to_path: str, path to the output data
    num_turn: int, number of turns
    cluster_number: int, number of clusters
    """
    logger.info(f"\nbegin processing fused data ...")
    for turn in range(num_turn):
        logger.info(f"Processing turn{turn} ...")
        prev_turn = turn - 1
        data_path = os.path.join(cache_path, f"turn{turn}")
        prev_data_path = os.path.join(cache_path, f"turn{prev_turn}")
        os.makedirs(data_path, exist_ok=True)
        if turn == 0:
            logger.info(f"Initializing data ...")
            pro_converse(
                data_file=to_fused_path,
                table_file=table_path,
                dump_path=os.path.join(data_path, "fused.filt.json"),
            )
            yield from yield_process(turn, 0, num_turn, 2)
            logger.info(f"Clustering question ...")
            pro_cluster(
                data_file=os.path.join(data_path, "fused.filt.json"),
                dump_file=os.path.join(data_path, "fused.filt.cluster.json"),
                encoder_name_or_path=encoder_name_or_path,
                model_dump_file=os.path.join(data_path, "cluster.pkl"),
                cluster_number=cluster_number,
            )
            yield from yield_process(turn, 1, num_turn, 2)
            continue
        logger.info(f"Sampling data ...")
        pro_initialize(
            table_file=table_path,
            example_file=os.path.join(prev_data_path, "fused.filt.cluster.json"),
            dump_path=os.path.join(data_path, "table.json"),
        )
        yield from yield_process(turn, 0, num_turn, 6)
        logger.info(f"Generating SQL ...")
        pro_generate_sql(
            data_file=os.path.join(data_path, "table.json"),
            database_path=database_path,
            dump_file=os.path.join(data_path, "sql.json"),
            stream_handler=stream_handler,
            data_size=8 if test else None,
        )
        yield from yield_process(turn, 1, num_turn, 6)
        logger.info(f"Generating question ...")
        pro_generate_question(
            data_file=os.path.join(data_path, "sql.json"),
            dump_file=os.path.join(data_path, "fused.json"),
            stream_handler=stream_handler,
            database_path=database_path,
        )
        yield from yield_process(turn, 2, num_turn, 6)
        logger.info(f"Filtering data ...")
        pro_filt(
            data_file=os.path.join(data_path, "fused.json"),
            previous_data_file=os.path.join(prev_data_path, "fused.filt.json"),
            dump_file=os.path.join(data_path, "fused.filt.json"),
            database_path=database_path,
            stream_handler=stream_handler,
        )
        yield from yield_process(turn, 3, num_turn, 6)
        logger.info(f"Clustering question ...")
        pro_cluster(
            data_file=os.path.join(data_path, "fused.filt.json"),
            dump_file=os.path.join(data_path, "fused.filt.cluster.json"),
            encoder_name_or_path=encoder_name_or_path,
            model_dump_file=os.path.join(data_path, "cluster.pkl"),
            cluster_number=cluster_number,
        )
        yield from yield_process(turn, 4, num_turn, 6)
        logger.info(f"constructing demostration ...")
        to_demo_file(
            cluster_file=os.path.join(data_path, "fused.filt.cluster.json"),
            dump_file=os.path.join(to_path, "demo.json"),
        )
        yield from yield_process(turn, 5, num_turn, 6)
    # shutil.rmtree(database_path)


if __name__ == "__main__":
    port = 5000
    gpu_node = "gpu06"
    predict_url = f"http://{gpu_node}:{port}/predict"
    stream_handler = StreamDataHandler(predict_url)

    # fused_path = "./cache/aaa/New Chat_86009c5c-3cc0-4912-b347-aac0dd5c25ef/fused"
    # to_fused_path = os.path.join(fused_path, "to_fused.json")
    # merge_json_files_in_folder(fused_path, to_fused_path, recursive=False)

    # table_path = os.path.join(fused_path, "tables", "tables.json")
    # os.makedirs(os.path.dirname(table_path), exist_ok=True)
    # data = extract_sqlite_metadata_from_folder(fused_path)
    # with open(table_path, 'w', encoding='utf-8') as f:
    #     json.dump(data, f, indent=4, ensure_ascii=False)

    # cache_path = os.path.join(fused_path, "fused/")
    # encoder_name_or_path = "./model/SGPT/125m"
    # database_path = fused_path
    # to_path = "./cache/aaa/New Chat_86009c5c-3cc0-4912-b347-aac0dd5c25ef"

    # pro_fused(to_fused_path, cache_path, encoder_name_or_path, table_path, database_path, stream_handler, to_path, num_turn=2, cluster_number=4)

    to_fused_path = "../Chase/test.json"
    cache_path = "./cache/Chase/fused"
    encoder_name_or_path = "./model/SGPT/125m"
    table_path = "../Chase/tables.json"
    database_path = "../Chase/database"
    to_path = "../Chase/fused"
    notice = pro_fused(
        to_fused_path=to_fused_path,
        cache_path=cache_path,
        encoder_name_or_path=encoder_name_or_path,
        table_path=table_path,
        database_path=database_path,
        stream_handler=stream_handler,
        to_path=to_path,
        num_turn=4,
        cluster_number=8,
    )
    os.makedirs(to_path, exist_ok=True)
    for n in notice:
        print(n)