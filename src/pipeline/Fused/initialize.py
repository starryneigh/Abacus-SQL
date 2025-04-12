import json
import random
import argparse

from typing import List, Dict, Any, Tuple


def sample_example_pair(examples: List[List[Dict[str, Any]]], source_table: str) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    cluster1, cluster2 = random.sample(examples, 2)
    example1 = random.choice(cluster1)
    example2 = random.choice(cluster2)
    if example1['table']['db_id'] == example2['table']['db_id']:
        return sample_example_pair(examples, source_table)
    if source_table in [example1['table']['db_id'], example2['table']['db_id']]:
        return sample_example_pair(examples, source_table)
    return example1, example2


def random_example_pair(examples: List[List[Dict[str, Any]]], source_table: str) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    def random_sample(examples: List[List[Dict[str, Any]]]) -> Dict[str, Any]:
        return random.choice(random.choice(examples))

    example1 = random_sample(examples)
    example2 = random_sample(examples)
    if example1['table']['db_id'] == example2['table']['db_id']:
        return random_example_pair(examples, source_table)
    if source_table in [example1['table']['db_id'], example2['table']['db_id']]:
        return random_example_pair(examples, source_table)
    return example1, example2


def pro_initialize(table_file: str, example_file: str, dump_path: str, selection: str='cluster', random_seed: int=42):
    random.seed(random_seed)
    with open(table_file, 'r', encoding='utf-8') as f:
        tables: List[Dict[str, Any]] = json.load(f)

    if example_file:
        with open(example_file, 'r', encoding='utf-8') as f:
            examples: List[List[Dict[str, Any]]] = json.load(f)
        results: List[Dict[str, Any]] = []
        for t in tables:
            if selection == 'random':
                example1, example2 = random_example_pair(examples, t['db_id'])
            elif selection == 'cluster':
                example1, example2 = sample_example_pair(examples, t['db_id'])
            for example in [example1, example2]:
                if 'reference' in example:
                    example.pop('reference')
            results.append({
                "reference": [example1, example2],
                "table": t
            })
    else:
        results: List[Dict[str, Any]] = [{
            "reference": [],
            "table": t
        } for t in tables]

    with open(dump_path, 'w', encoding='utf-8') as f:
        json.dump(results, f, indent=4, ensure_ascii=False)


# if __name__ == '__main__':
#     parser = argparse.ArgumentParser()
#     parser.add_argument("--example_file", type=str)
#     parser.add_argument("--table_file", type=str)
#     parser.add_argument("--dump_path", type=str, help="dump path")
#     parser.add_argument("--selection", type=str,
#                         choices=['random', 'cluster'], default='cluster')
#     parser.add_argument("--random_seed", type=int,
#                         default=42, help="random seed")
#     args = parser.parse_args()
#     random.seed(args.random_seed)

#     with open(args.table_file, 'r', encoding='utf-8') as f:
#         tables: List[Dict[str, Any]] = json.load(f)

#     if args.example_file:
#         with open(args.example_file, 'r', encoding='utf-8') as f:
#             examples: List[List[Dict[str, Any]]] = json.load(f)
#         results: List[Dict[str, Any]] = []
#         for t in tables:
#             if args.selection == 'random':
#                 example1, example2 = random_example_pair(examples, t['db_id'])
#             elif args.selection == 'cluster':
#                 example1, example2 = sample_example_pair(examples, t['db_id'])
#             for example in [example1, example2]:
#                 if 'reference' in example:
#                     example.pop('reference')
#             results.append({
#                 "reference": [example1, example2],
#                 "table": t
#             })
#     else:
#         results: List[Dict[str, Any]] = [{
#             "reference": [],
#             "table": t
#         } for t in tables]

#     with open(args.dump_path, 'w', encoding='utf-8') as f:
#         json.dump(results, f, indent=4, ensure_ascii=False)
