import sys
import json
import argparse
from tqdm import tqdm

from typing import Dict, Any
from transformers import AutoTokenizer

from .selector import select_multiple
from .database import database_to_string, fix_sql, pack_db_path
from .my_logger import MyLogger

sys.path.append(".")
logger = MyLogger("prompt_gen", "logs/text2sql.log")

SYSTEM_PROMPT_CH = {
    "role": "system",
    "content": """你是数据库领域和 SQL 的专家，专门从事分析和编写 SQL 查询的任务。
你可以为改进数据库架构和为各种数据库管理系统编写优化的 SQL 语句，来提供详细、高效和准确的建议。

你的任务是：

1. 根据用户问题，仔细分析用户的需求，并生成一个推理过程以辅助生成 SQL。
2. 根据推理过程，生成一个 SQL 查询语句来回答用户的问题。
3. 聚焦于多轮对话中的信息，并利用这些信息来更好的生成 SQL 查询语句。
4. 利用给出的示例，从中提取出生成 SQL 查询语句的一般方法，并应用到用户的问题中，但不要直接复制其中的信息；
输出格式如下：
推理过程: 

{分步生成推理过程}

SQL: 
```sql
{SQL 查询语句}
```
""",
}

SYSTEM_PROMPT_EN = {
    "role": "system",
    "content": """You are an expert in the field of databases and SQL, specializing in analyzing and writing SQL queries.
You can provide detailed, efficient, and accurate suggestions to improve database architecture and write optimized SQL statements for various database management systems.

Your tasks are:

1. Analyze the user's requirements carefully based on their questions and generate a reasoning process to assist in creating SQL.
2. Based on the reasoning process, generate an SQL query to answer the user's question.
3. Focus on the information provided in multi-turn conversations and use it to better generate SQL queries.
4. Extract general methods for generating SQL queries from the given examples and apply them to the user's questions, but do not directly copy the information from the examples.

The output format is as follows:
rationale: {rationale step by step}  

SQL: 
```sql
{SQL}
```
"""
}

SYSTEM_PROMPT_CH_FB = {
    "role": "system",
    "content": """
你是数据库领域和 SQL 的专家，专注于分析和纠正 SQL 查询的错误。
你的任务是：
1. 根据用户提供的反馈和上文相关问题的rationale，仔细分析问题所在。
2. 修改错误的 SQL 查询以确保其满足用户的需求和语境要求。
3. 输出应该包括：
  - rationale：结合用户的问题和反馈更新上文的rationale。
  - 优化的 SQL 查询，确保：
    - 使用纠正错误后的rationale准确回答问题。
    - 仅包含必要的语句（以 SELECT 开头，以分号结尾）。
    - 确保包含用户明确提及的所有列和条件，避免遗漏。
4. 提供准确的、优化的语法，并避免包含任何无关信息。
5. 生成的SQL要能回答用户的问题。
输出格式如下：
推理过程: {rationale}

SQL: 
```sql
{SQL}
```
"""
}

SYSTEM_PROMPT_EN_FB = {
    "role": "system",
    "content": """
You are an expert in the field of databases and SQL, specializing in analyzing and correcting SQL query errors.
Your tasks are:
1. Carefully analyze the problem based on the feedback provided by the user and the rationale related to the relevant question from the context.
2. Modify the erroneous SQL query to ensure it meets the user's requirements and contextual needs.
3. The output should include:
  - Rationale: Update the rationale based on the user's question and feedback.
  - Optimized SQL query, ensuring:
    - It uses the corrected rationale to accurately answer the question.
    - It contains only the necessary statements (starting with SELECT and ending with a semicolon).
    - It includes all columns and conditions explicitly mentioned by the user, leaving nothing out.
4. Provide precise, optimized syntax while avoiding any irrelevant information.
5. The generated SQL must be able to answer the user's question.
The output format is as follows:
rationale: {rationale}  

SQL: 
```sql
{SQL}
```
"""
}

SYSTEM_PROMPT_EN_ALIGN = {
    "role": "system",
    "content": """
Align the tokens in the given question to the table entities or the column entities of the schema above, considering the given SQL.
Present the aligned tokens in the python format List[Dict[str, str]], where each Dict[str, str] denoting each token in the question containing the following keys:
{{
    "token": the token in the question
    "schema": the schema entity aligned to the token
    "type": the type of the entity aligned to the token
}}
The "type" can be one of the following:
* "tbl": the table name
* "col": the column name
* "val": the value
"schema" and "type" are either both null or not null at the same time.
"""
}

SYSTEM_PROMPT_CH_ALIGN = {
    "role": "system",
    "content": """
将给定问题中的词汇与上述模式中的表实体或列实体进行对齐，并考虑给定的 SQL。
以 Python 格式 List[Dict[str, str]] 展示对齐后的词汇，每个 Dict[str, str] 表示问题中的一个词汇，包含以下键：
{{
    "token": 问题中的词汇
    "schema": 与词汇对齐的模式实体
    "type": 与词汇对齐的实体类型
}}
"type" 可以是以下之一：
* "tbl": 表名
* "col": 列名
* "val": 值
"schema" 和 "type" 要么都为 null，要么都不为 null。
"""
}

SYSTEM_PROMPT_EN_HALLUCINATE = {
    "role": "system",
    "content": """
Hallucinate a SQL to answer the question.
Quote your answer with: 
```sql
<answer sql>
```
"""
}

SYSTEM_PROMPT_CH_HALLUCINATE = {
    "role": "system",
    "content": """
幻想一个 SQL 来回答问题，你只需要按照以下格式生成SQL语句，不要生成额外的东西。
你的回答格式如下：
```sql
<answer sql>
```
"""
}

SYSTEM_PROMPT_EN_DEBUG = {
    "role": "system",
    "content": """
Fix the sql to answer the question based on the given database and the error information.
Present your sql in the format:
```sql
<your sql>
```
"""
}

SYSTEM_PROMPT_CH_DEBUG = {
    "role": "system",
    "content": """
根据给定的数据库和错误信息修正 SQL，以回答问题。
以以下格式展示你的 SQL：
```sql
<your sql>
```
"""
}

DEMO_EXAMPLE_CH = """
---

使用以下给定示例回答问题:

{demonstration}

---
"""

DEMO_EXAMPLE_EN = """
---

Answer the question using the following examples:

{demonstration}

---
"""

SCHEMA_EXAMPLE_CH = """
请用给定的schema生成一个SQL来回答以下问题：
schema: ```sql
{schema}
```
""".strip()

SCHEMA_EXAMPLE_EN = """
Generate an SQL to answer the question with the given schema:
schema: ```sql
{schema}
```
""".strip()

SCHEMA_EXAMPLE_CH_ALIGN = """
将给定问题中的词汇与schema中的表实体或列实体进行对齐。
schema: ```sql
{schema}
```
""".strip()

SCHEMA_EXAMPLE_EN_ALIGN = """
Align the tokens in the given question to the table entities or the column entities of the schema.
schema: ```sql
{schema}
```
""".strip()

PROMPT = """
请用给定的schema生成一个SQL来回答以下问题：

---

例子如下:

{demonstration}

---

基于上述例子，回答以下问题：

{user}

""".strip()

EXAMPLE = """
schema: ```sql
{schema}
```
Question: {question}
SQL: {sql}
""".strip()

EXAMPLE_que = """
Question: {question}
SQL: {sql}
""".strip()

EXAMPLE_que_align = """
Question: {question}
SQL: {sql}
Align: {align}
""".strip()


def db_to_questions(datas):
    # 为相同数据库的问题建立一个字典
    db_to_questions = {}
    for data in datas:
        if data["db_id"] not in db_to_questions:
            db_to_questions[data["db_id"]] = []
        db_to_questions[data["db_id"]].append(data)
    return db_to_questions


def format_demo(demo, db_path):
    return EXAMPLE.format(
        schema=database_to_string(
            pack_db_path(db_path, demo["db_id"]),
            granularity="table",
            sql=demo["query"],
            question=demo["question"],
        ),
        question=demo["question"],
        sql=demo["query"],
    )


def format_user(data: Dict[str, Any], schema: str) -> str:
    return EXAMPLE.format(
        schema=schema,
        question=(
            data["user_question"]
            if len(data["history"]) <= 1
            else data["history"][0]["content"]
        ),
        sql="",
    )


def gen_demo(
    demos, data: Dict, database_path="./data/database", que_num=0, args=None, mode="ch"
) -> list[Dict[str, Any]]:
    demonstration = select_multiple([data], demos, args=args)[0]
    db_que_map = db_to_questions(demos)

    # 提取生成示例的公共逻辑
    def generate_schema_and_question(demo) -> Dict[str, Any]:
        demo_dict = {}
        schema=database_to_string(
            pack_db_path(database_path, demo["db_id"]),
            sql=demo["query"],
            question=demo["question"]
        )
        related_schema=database_to_string(
            pack_db_path(database_path, demo["db_id"]),
            sql=demo["query"],
            question=demo["question"],
            granularity="table"
        )

        examples = []
        examples.append({
            "question": demo["question"],
            "query": demo["query"],
        })
        examples.extend([
            db_que_map[demo["db_id"]][i] 
            for i in range(min(que_num, len(db_que_map[demo["db_id"]])))
        ])
        demo_dict["schema"] = schema
        demo_dict["related_schema"] = related_schema
        demo_dict["examples"] = examples
        return demo_dict

    prompt_demo = []
    for demo in demonstration:
        prompt_demo.append(generate_schema_and_question(demo))
    return prompt_demo


def construct_demos(
    data: dict, demo_db_path, demos, que_num=0, demo_num=3, encore_file=None, mode="ch"
)-> dict:
    args = argparse.Namespace(selector_type="bm25", demonstration_number=demo_num)
    if encore_file:
        with open(encore_file, "r", encoding="utf-8") as f:
            demos += json.load(f)
    demostration = gen_demo(demos, data, demo_db_path, que_num, args, mode)
    data["demo"] = demostration
    return data


def construct_schemas(data: dict, db_map) -> dict:
    db_path = db_map[data["db_id"]]
    schema = database_to_string(
        db_path,
        question=data["question"],
    )
    data["schema"] = schema
    return data


def _get_schema_demo_prompt(mode: str, prompt_type: str):
    """
    根据语言模式和用户类型获取对应的 schema、demo 和 prompt。
    
    :param mode: 语言模式 ("en" 或 "ch")。
    :param user_type: 用户类型 ("feedback" 或 "question")。
    :return: schema, demo, prompt 三个值。
    """
    schema_dict = {
        "en": {
            "default": SCHEMA_EXAMPLE_EN,
            "align": SCHEMA_EXAMPLE_EN_ALIGN,
        },
        "ch": {
            "default": SCHEMA_EXAMPLE_CH,
            "align": SCHEMA_EXAMPLE_CH_ALIGN,
        }
    }
    demo_dict = {
        "en": DEMO_EXAMPLE_EN,
        "ch": DEMO_EXAMPLE_CH
    }
    prompt_dict = {
        "en": {
            "question": SYSTEM_PROMPT_EN,
            "feedback": SYSTEM_PROMPT_EN_FB,
            "align": SYSTEM_PROMPT_EN_ALIGN,
            "hallucinate": SYSTEM_PROMPT_EN_HALLUCINATE,
            "debug": SYSTEM_PROMPT_EN_DEBUG
        },
        "ch": {
            "question": SYSTEM_PROMPT_CH,
            "feedback": SYSTEM_PROMPT_CH_FB,
            "align": SYSTEM_PROMPT_CH_ALIGN,
            "hallucinate": SYSTEM_PROMPT_CH_HALLUCINATE,
            "debug": SYSTEM_PROMPT_CH_DEBUG
        }
    }
    schema = schema_dict.get(mode, {}).get(prompt_type, schema_dict.get(mode, {}).get("default"))
    demo = demo_dict.get(mode, DEMO_EXAMPLE_CH)  # 默认中文 demo
    prompt = prompt_dict.get(mode, {}).get(prompt_type, prompt_dict.get(mode, {}).get("question"))
    return schema, demo, prompt


def generate_prompt(
    prompt_data, tokenizer: AutoTokenizer, mode: str = "ch", demo_related=False, schema_related=False,
    prompt_type=None, align_flag=False, use_schema=True
) -> tuple[str, list]:
    """
    生成用于模型生成的 prompt。
    :param tokenizer: 用于将消息格式化的 tokenizer。
    :param data: 用户问题、历史对话等数据。
    :param demos_file: 示例数据文件路径。
    :param demo_db_path: 示例数据库路径。
    :param schema: 数据库表的 schema。
    :param shot: 从示例中选择的个数。
    :return: 生成的 prompt 字符串。
    """

        

    # 获取历史对话记录
    history = prompt_data.get("history", [])
    if not prompt_type:
        prompt_type = history[-1]["type"] if len(history) > 0 else "question"

    # 根据参数选择相关示例和 schema
    demonstrations = prompt_data.get("demo", [])
    schema = prompt_data.get("related_schema" if schema_related else "schema", "")

    SCHEMA, DEMO, PROMPT = _get_schema_demo_prompt(mode, prompt_type)

    def _demo_string(demos, use_schema=True, align=False, related=False):
        demos_strs = []
        for i, demo in enumerate(demos):
            if related:
                schema_str = SCHEMA.format(schema=demo["related_schema"])
            else:
                schema_str = SCHEMA.format(schema=demo["schema"])
            if align:
                question_strs = [
                    EXAMPLE_que_align.format(
                        question=example["question"],
                        sql=example["query"],
                        Align=example["align"]
                    ) for example in demo["examples"]
                ]
            else:
                question_strs = [
                    EXAMPLE_que.format(
                        question=example["question"],
                        sql=example["query"],
                    ) for example in demo["examples"]
                ]
            if use_schema:
                demo_str = "\n".join([schema_str] + question_strs)
            else:
                demo_str = "\n".join(question_strs)
            demos_strs.append(demo_str)
        return DEMO.format(demonstration="\n\n".join(demos_strs))

    demo_strs = _demo_string(demonstrations, related=demo_related, use_schema=use_schema)
    user_schema = {
        "role": "user",
        "content": SCHEMA.format(schema=schema),
    }
    user_demo = {
        "role": "user",
        "content": demo_strs,
    }

    messages = [PROMPT]
    if demonstrations and prompt_type != "align":
        messages.append(user_demo)
    if use_schema:
        messages.append(user_schema)
    
    if prompt_type != "align" and prompt_type != "debug":
        history_messages = [
            {
                "role": "user" if h["role"] == "user" else "assistant",
                "content": h["content"],
            }
            for h in history[0 : len(history)-1]
            if h["type"] != "execution"
        ]
        question = history[-1]["content"]
        if not align_flag:
            history_messages.append(
                {
                    "role": "user",
                    "content": question,
                }
            )
        else:
            from DAC.generate import pack_question
            question = pack_question(prompt_data, use_alignment="pred", mode=mode)
            history_messages.append(
                {
                    "role": "user",
                    "content": question,
                }
            )
        messages.extend(history_messages)
    else:
        question = history[-1]["content"] if len(history) > 0 else ""
        messages.append(
            {
                "role": "user",
                "content": question,
            }
        )

    try:
        prompt = tokenizer.apply_chat_template(
            messages, add_generation_prompt=True, tokenize=False
        )
    except Exception as e:
        logger.error(f"构建 prompt 时出错：{e}")
        raise

    return prompt, messages


# if __name__ == "__main__":
#     # test
#     tokenizer = AutoTokenizer.from_pretrained("./model/CodeLlama/7b")
#     data = {
#         "user_question": "请告诉我员工雇佣关系",
#         "history": [
#             {"role": "user", "content": "请告诉我员工雇佣关系", "type": "question"}
#         ],
#         "db_file_path": "./cache/temp_database.sqlite",
#     }
#     demos_file = "./dataset/Spider/train.json"
#     db_path = "./dataset/Spider/database"
#     schema = "employee"
#     prompt = generate_prompt(
#         tokenizer=tokenizer, data=data, demos_file=demos_file, db_path=db_path, shot=1, schema=schema
#     )
#     print(prompt)
