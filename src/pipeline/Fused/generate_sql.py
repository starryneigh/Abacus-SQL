import os
import sys
import json
import random
import argparse

from tqdm import tqdm
from copy import deepcopy
from typing import Dict, List, Any

from transformers import set_seed

sys.path.append('.')

EXAMPLE_SEP = '\n\n\n'
INSTRUCTION = f"""
Synthesize one SQL query for the given database .
""".strip()
PROMPT_WITHOUT_REFERENCE = """
{database}
-- Synthesize one SQL query for the given database . Do not select more than {column_number} columns .
{sql}
""".strip()
PROMT_WITH_REFERENCE = """
{database}
-- Synthesize a new single for the above database imitating {reference} .
{sql}
"""


def pro_generate_sql(data_file: str, database_path: str, dump_file: str, stream_handler: str, data_size: int=None, random_seed: int=42):
    from .sql_process import format_db, parse_sql_safe, fix_sql
    set_seed(random_seed)

    db_path = database_path
    with open(data_file, 'r', encoding='utf-8') as f:
        data: List[Dict[str, Any]] = json.load(f)
        print(f"Loaded {len(data)} examples")
        if data_size:
            data = random.sample(data, data_size)
            print(f"Sampled {data_size} examples")

    if len(data[0]['reference']):
        PROMPT = PROMT_WITH_REFERENCE
    else:
        PROMPT = PROMPT_WITHOUT_REFERENCE

    prompts: List[str] = []
    prompts_map: List[int] = []
    for column_number in range(1, 4):
        for i, d in enumerate(data):
            params = {
                "database": format_db(d['table'], db_path, 3),
                "column_number": column_number,
                "sql": 'SELECT'
            }
            if '{reference}' in PROMPT:
                params["reference"] = ' and '.join(
                    [f"\"{r['query']}\"" for r in d['reference']])
            data_prompt = PROMPT.format(**params)
            prompts.append(EXAMPLE_SEP.join(
                [INSTRUCTION] + [data_prompt]).strip())
            prompts_map.append(i)
        print(f"Generated {len(prompts)} prompts for {column_number} columns")
    # print(prompts[-1])
    sys.stdout.flush()

    predictions = stream_handler.generate_with_llm(prompts)

    predictions_new = [[] for _ in range(len(data))]
    for idx, p in zip(prompts_map, predictions):
        predictions_new[idx].extend(p)
    predictions = predictions_new
    # print(predictions[0])
    sys.stdout.flush()

    results = []
    for d, p in tqdm(zip(data, predictions), total=len(data), desc='Evaluating SQL'):
        sql = [fix_sql(x[0]) for x in p]
        sql = list(set(sql))
        current_db_path = os.path.join(
            db_path, d['table']['db_id'], d['table']['db_id'] + '.sqlite')
        sql = [x for x in sql if parse_sql_safe(current_db_path, x)]
        for s in sql:
            d_copy = deepcopy(d)
            d_copy['query'] = s
            results.append(d_copy)
    with open(dump_file, 'w', encoding='utf-8') as f:
        json.dump(results, f, indent=4, ensure_ascii=False)


# if __name__ == '__main__':
#     from utils.generator import generate_with_llm
#     from sql_process import format_db, parse_sql_safe, fix_sql
#     print(sys.path)

#     parser = argparse.ArgumentParser()
#     parser.add_argument("--llm_name_or_path", type=str, help="llm path")
#     parser.add_argument("--config_file", type=str, help="config path")
#     parser.add_argument("--data_file", type=str, help="data path")
#     parser.add_argument("--database_path", type=str)
#     parser.add_argument("--dump_file", type=str, help="dump path")
#     parser.add_argument("--data_size", type=int, help="data size")
#     parser.add_argument("--random_seed", type=int,
#                         default=42, help="random seed")
#     args = parser.parse_args()
#     set_seed(args.random_seed)

#     db_path = args.database_path
#     with open(args.data_file, 'r', encoding='utf-8') as f:
#         data: List[Dict[str, Any]] = json.load(f)
#         if args.data_size:
#             data = random.sample(data, args.data_size)
#             print(f"Sampled {args.data_size} examples")
#     with open(args.config_file, 'r', encoding='utf-8') as f:
#         config: Dict[str, Any] = json.load(f)

#     if len(data[0]['reference']):
#         PROMPT = PROMT_WITH_REFERENCE
#     else:
#         PROMPT = PROMPT_WITHOUT_REFERENCE

#     prompts: List[str] = []
#     prompts_map: List[int] = []
#     for column_number in range(1, 4):
#         for i, d in enumerate(data):
#             params = {
#                 "database": format_db(d['table'], db_path, 3),
#                 "column_number": column_number,
#                 "sql": 'SELECT'
#             }
#             if '{reference}' in PROMPT:
#                 params["reference"] = ' and '.join(
#                     [f"\"{r['query']}\"" for r in d['reference']])
#             data_prompt = PROMPT.format(**params)
#             prompts.append(EXAMPLE_SEP.join(
#                 [INSTRUCTION] + [data_prompt]).strip())
#             prompts_map.append(i)
#         print(f"Generated {len(prompts)} prompts for {column_number} columns")
#     print(prompts[-1])
#     sys.stdout.flush()

#     predictions = generate_with_llm(args.llm_name_or_path, prompts, config)
#     predictions_new = [[] for _ in range(len(data))]
#     for idx, p in zip(prompts_map, predictions):
#         predictions_new[idx].extend(p)
#     predictions = predictions_new
#     print(predictions[0])
#     sys.stdout.flush()

#     results = []
#     for d, p in tqdm(zip(data, predictions), total=len(data), desc='Evaluating SQL'):
#         sql = [fix_sql(x[0]) for x in p]
#         sql = list(set(sql))
#         current_db_path = os.path.join(
#             db_path, d['table']['db_id'], d['table']['db_id'] + '.sqlite')
#         sql = [x for x in sql if parse_sql_safe(current_db_path, x)]
#         for s in sql:
#             d_copy = deepcopy(d)
#             d_copy['query'] = s
#             results.append(d_copy)
#     with open(args.dump_file, 'w', encoding='utf-8') as f:
#         json.dump(results, f, indent=4, ensure_ascii=False)
