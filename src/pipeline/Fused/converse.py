import json
import argparse
from typing import List, Dict, Any, Tuple


def get_table_map(tables: List[Dict[str, Any]]) -> Dict[str, Dict[str, Any]]:
    table_map = {}
    for table in tables:
        table_map[table['db_id']] = table
    return table_map


def con_train_single(data: Dict[str, Any], table_map: Dict[str, Dict[str, Any]]) -> Dict[str, Any]:
    single = {}
    single['reference'] = []
    single['table'] = table_map[data['db_id']]
    single['query'] = data['query']
    single['question'] = data['question']
    return single


def con_train(data: List[Dict[str, Any]], table_map: Dict[str, Dict[str, Any]]) -> List[Dict[str, Any]]:
    result = []
    for d in data:
        result.append(con_train_single(d, table_map))
    return result
    
    
def parser():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_file', type=str, help='path to the data file to fused')
    parser.add_argument('--table_file', type=str, help='path to the table file')
    parser.add_argument('--dump_path', type=str, help='path to the dump file')
    return parser.parse_args()


def pro_converse(data_file, table_file, dump_path):
    with open(data_file, 'r', encoding='utf-8') as f:
        data: List[Dict[str, Any]] = json.load(f)
    with open(table_file, 'r', encoding='utf-8') as f:
        tables: List[Dict[str, Any]] = json.load(f)
    table_map = get_table_map(tables)
    # print(table_map['department_management'])

    result = con_train(data, table_map)
    with open(dump_path, 'w', encoding='utf-8') as f:
        json.dump(result, f, ensure_ascii=False, indent=4)
    

# if __name__ == "__main__":
    # args = parser()
    # with open(args.data_file, 'r', encoding='utf-8') as f:
    #     data: List[Dict[str, Any]] = json.load(f)
    # with open(args.table_file, 'r', encoding='utf-8') as f:
    #     tables: List[Dict[str, Any]] = json.load(f)
    # table_map = get_table_map(tables)
    # print(table_map['department_management'])

    # result = con_train(data, table_map)
    # with open(args.dump_path, 'w', encoding='utf-8') as f:
    #     json.dump(result, f, ensure_ascii=False, indent=4)
    
    
    