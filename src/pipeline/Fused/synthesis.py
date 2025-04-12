import json
import argparse
from typing import List, Dict, Any, Tuple

def parser():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_file', type=str, help='path to the data file to fused', nargs='+')
    parser.add_argument('--dump_path', type=str, help='path to the dump file')
    return parser.parse_args()

def read_jsons(file_paths: List[str]) -> List[Dict[str, Any]]:
    data = []
    for file_path in file_paths:
        with open(file_path, 'r', encoding='utf-8') as f:
            data.extend(json.load(f))
    return data

def syn_datas(datas: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    results = []
    for data in datas:
        result = {}
        result['question'] = data['question']
        result['query'] = data['query']
        result['db_id'] = data['table']['db_id']
        results.append(result)
    return results
    
def to_demo_file(cluster_file, dump_file, demo_path=None):
    with open(cluster_file, 'r', encoding='utf-8') as f:
        data = json.load(f)
    datas = []
    for d in data:
        datas.extend(d)
    if demo_path:
        with open(demo_path, 'r', encoding='utf-8') as f:
            datas += json.load(f)
    results = syn_datas(datas)
    with open(dump_file, 'w', encoding='utf-8') as f:
        json.dump(results, f, ensure_ascii=False, indent=4)

if __name__ == "__main__":
    args = parser()
    datas = read_jsons(args.data_file)
    results = syn_datas(datas)
    with open(args.dump_path, 'w', encoding='utf-8') as f:
        json.dump(results, f, ensure_ascii=False, indent=4)


