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
{database}
-- Using natural language, generate a question corresponding to the given SQL : {SQL}.
Question: {question}
""".strip()


def pro_generate_question(data_file: str, dump_file: str, stream_handler, database_path: str, data_size: int=None, random_seed: int=42):
    from .sql_process import format_db
    set_seed(random_seed)

    db_path = database_path
    with open(data_file, 'r', encoding='utf-8') as f:
        data: List[Dict[str, Any]] = json.load(f)
        if data_size:
            data = random.sample(data, data_size)

    prompts: List[str] = []
    for d in data:
        data_prompt = PROMPT.format(
            database=format_db(d['table'], db_path, 3),
            SQL=d['query'],
            question=""
        )
        reference_prompt = [PROMPT.format(
            database=format_db(r['table'], db_path, 3),
            SQL=r['query'],
            question=r['question']
        ) for r in d['reference']]
        prompts.append(EXAMPLE_SEP.join(
            [INSTRUCTION] + reference_prompt + [data_prompt]).strip())
    # print(prompts[-1])

    predictions = stream_handler.generate_with_llm(prompts)
    for d, p in zip(data, predictions):
        if not p or not p[0] or not p[0][0]:
            continue
        d['question'] = p[0][0].strip().split('\n')[0]
    with open(dump_file, 'w', encoding='utf-8') as f:
        json.dump(data, f, indent=4, ensure_ascii=False)


# if __name__ == '__main__':
#     from sql_process import format_db
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

#     prompts: List[str] = []
#     for d in data:
#         data_prompt = PROMPT.format(
#             database=format_db(d['table'], db_path, 3),
#             SQL=d['query'],
#             question=""
#         )
#         reference_prompt = [PROMPT.format(
#             database=format_db(r['table'], db_path, 3),
#             SQL=r['query'],
#             question=r['question']
#         ) for r in d['reference']]
#         prompts.append(EXAMPLE_SEP.join(
#             [INSTRUCTION] + reference_prompt + [data_prompt]).strip())
#     print(prompts[-1])

#     predictions = generate_with_llm(args.llm_name_or_path, prompts, config)
#     for d, p in zip(data, predictions):
#         if not p or not p[0] or not p[0][0]:
#             continue
#         d['question'] = p[0][0].strip().split('\n')[0]
#     with open(args.dump_file, 'w', encoding='utf-8') as f:
#         json.dump(data, f, indent=4, ensure_ascii=False)
