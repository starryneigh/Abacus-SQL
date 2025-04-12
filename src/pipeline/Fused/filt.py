import os
import sys
import json
import random
import argparse

from typing import Dict, List, Any
from transformers import set_seed

sys.path.append('.')

EXAMPLE_SEP = '\n\n\n'
INSTRUCTION = f"""
Using natural language, generate a question corresponding to the given SQL .
Different example is seperated with '{EXAMPLE_SEP}' .
""".strip()
PROMPT = """
{table}
-- Using valid SQLite , answer the following questions for the tables provided above .
Question: {question}
{sql}
""".strip()


def pro_filt(data_file: str, previous_data_file, dump_file: str, database_path: str, stream_handler, data_size: int = None, random_seed: int = 42):
    from .sql_process import format_db, check_sql_equivalence_safe, fix_sql
    set_seed(random_seed)

    db_path = database_path
    with open(data_file, 'r', encoding='utf-8') as f:
        data: List[Dict[str, Any]] = json.load(f)
    if data_size:
        data = random.sample(data, data_size)

    prompts = []
    data_available = []
    for d in data:
        if 'question' not in d:
            continue
        data_prompt = PROMPT.format(
            table=format_db(d['table'], db_path, 3),
            question=d['question'],
            sql="SELECT"
        )
        reference_prompt = [
            PROMPT.format(
                table=format_db(r['table'], db_path, 3),
                question=r['question'],
                sql=r['query']
            ) for r in d['reference']]
        prompts.append(EXAMPLE_SEP.join(
            [INSTRUCTION] + reference_prompt + [data_prompt]).strip())
        data_available.append(d)
    # print(prompts[-1])
    data = data_available

    results = []
    predictions = stream_handler.generate_with_llm(prompts)
    for p, d in zip(predictions, data):
        if not p or not p[0] or not p[0][0]:
            continue
        pred_sql = fix_sql(p[0][0])
        current_db_path = os.path.join(
            db_path, d['table']['db_id'], d['table']['db_id'] + '.sqlite')
        if check_sql_equivalence_safe(current_db_path, pred_sql, d['query']):
            results.append(d)

    if previous_data_file:
        with open(previous_data_file, 'r', encoding='utf-8') as f:
            results = json.load(f) + results
            
    with open(dump_file, 'w', encoding='utf-8') as f:
        json.dump(results, f, indent=4, ensure_ascii=False)


# if __name__ == '__main__':
#     from sql_process import format_db, check_sql_equivalence_safe, fix_sql
#     from utils.generator import generate_with_llm

#     parser = argparse.ArgumentParser()
#     parser.add_argument("--llm_name_or_path", type=str, help="llm path")
#     parser.add_argument("--config_file", type=str, help="config path")
#     parser.add_argument("--data_file", type=str, help="data path")
#     parser.add_argument("--dump_file", type=str, help="dump path")
#     parser.add_argument("--database_path", type=str, help="database path")
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
#     with open(args.config_file, 'r', encoding='utf-8') as f:
#         config: Dict[str, Any] = json.load(f)

#     prompts = []
#     data_available = []
#     for d in data:
#         if 'question' not in d:
#             continue
#         data_prompt = PROMPT.format(
#             table=format_db(d['table'], db_path, 3),
#             question=d['question'],
#             sql="SELECT"
#         )
#         reference_prompt = [
#             PROMPT.format(
#                 table=format_db(r['table'], db_path, 3),
#                 question=r['question'],
#                 sql=r['query']
#             ) for r in d['reference']]
#         prompts.append(EXAMPLE_SEP.join(
#             [INSTRUCTION] + reference_prompt + [data_prompt]).strip())
#         data_available.append(d)
#     print(prompts[-1])
#     data = data_available

#     results = []
#     predictions = generate_with_llm(args.llm_name_or_path, prompts, config)
#     for p, d in zip(predictions, data):
#         if not p or not p[0] or not p[0][0]:
#             continue
#         pred_sql = fix_sql(p[0][0])
#         current_db_path = os.path.join(
#             db_path, d['table']['db_id'], d['table']['db_id'] + '.sqlite')
#         if check_sql_equivalence_safe(current_db_path, pred_sql, d['query']):
#             results.append(d)

#     with open(args.dump_file, 'w', encoding='utf-8') as f:
#         json.dump(results, f, indent=4, ensure_ascii=False)
