import os
import math
import json
import argparse

from copy import deepcopy
from typing import List, Dict, Any, Tuple
from .tutils import filter_ret_tables_from_db, pack_table

def find_json_files(path: str) -> List[str]:
    json_files = []
    for root, dirs, files in os.walk(path):
        for file in files:
            if file.endswith('.json'):
                json_files.append(os.path.join(root, file))
    return json_files

def pos(x: float) -> float:
    return (x+2)/2

def judge_stop(utte: str) -> bool:
    if "There is no" in utte or "None of the given tables" in utte or "No additional tables" in utte or "No completion needed" in utte or ("None" in utte):
        return True
    return False

def locate_idx(tables: List[str], pref: str):
    # print(tables)
    # print(pref)
    for ti, t in enumerate(tables):
        if pref == t.split("(")[0]:
            return ti
    return 100

def construct_tables_input(tables: List[str], dbs_dict: Dict[str, Any]):
    db_tables: Dict[str, List[str]] = {}
    tables_input: List[str] = ["" for i in range(len(tables))]
    for si, schema in enumerate(tables):
        db_id = schema.split(".")[0]
        if db_tables.get(db_id):
            db_tables[db_id].append(schema)
        else:
            db_tables.setdefault(db_id, [schema])
    for db, dtables in db_tables.items():
        # print("\n")
        # print(db)
        db_dict = dbs_dict[db]
        # print([dt.split("(")[0].split(".")[1] for dt in dtables])
        filt_db_dict = filter_ret_tables_from_db(db_dict, db, [dt.split("(")[0].split(".")[1] for dt in dtables])
        # print(filt_db_dict)
        table_names = filt_db_dict["table_names"]
        packed_tables = pack_table(filt_db_dict, True)
        # print(packed_tables)
        for pi, p in enumerate(packed_tables.split("\n\n")):
            idx = locate_idx(tables, f"{db}.{table_names[pi]}")
            p = "database: " + db + "\n" + p
            tables_input[idx] = p
            # print(f"{idx}: {tables_input}")
    # print(tables_input)
    for t in tables_input:
        if len(t) == 0:
            print(t)
        assert len(t) > 0
    # print(tables_input)
    return "\n".join(tables_input)

def extract_db_names(tables: List[str]) -> List[str]:
    db_names = {}
    for schema in tables:
        db_id = schema.split(".")[0]  # 提取数据库名称
        db_names[db_id] = db_names.get(db_id, 0) + 1
    # 选择出现次数最多的数据库
    db_names = sorted(db_names.items(), key=lambda x: x[1], reverse=True)
    # return db_names[0][0]
    return tables[0].split(".")[0]