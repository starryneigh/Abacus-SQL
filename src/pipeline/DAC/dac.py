import os
import json
from transformers import AutoTokenizer
from .align import generate_prompt, unpack_single_generation
from .debug import prompt_entity_linking, prompt_execution_error, prompt_hallucination
from ..utils.prompt_gen import generate_prompt
from ..utils.database import extract_skeleton
from ..utils.stream_generator import consistency
from ..utils.extract_from_sql import extract_sql_from_text
from ..utils.notice import send_notice


def align(prompt_data, stream_handler, tokenizer, mode="ch"):
    prompt = generate_prompt(prompt_data, tokenizer, mode, prompt_type="align")
    # print(prompt)

    prediction = stream_handler.generate_with_llm([prompt])[0]
    # print(prediction)
    data_and_predictions = (prompt_data, prediction, prompt_data["db_info"])
    prompt_data = unpack_single_generation(data_and_predictions)
    return prompt_data

def hallucinate(prompt_data, stream_handler, tokenizer, mode="ch"):
    prompt = generate_prompt(prompt_data, tokenizer, mode, prompt_type="hallucinate", use_schema=False)
    # print(prompt)

    prediction = stream_handler.generate_with_llm([prompt])[0]
    results = []
    skeleton_prev = extract_skeleton(prompt_data['query_pred'])
    for pred in prediction:
        prediction = pred[0].strip()
        # print(prediction)
        skeleton = extract_skeleton(extract_sql_from_text(prediction))
        if len(skeleton.split()) > len(skeleton_prev.split()):
            continue
        results.append((skeleton, prediction, pred[1]))

    if not results:
        prompt_data['hallucination'] = extract_skeleton(extract_sql_from_text(prediction[0][0].strip()))
    prompt_data['hallucination'] = consistency(results)[0]
    return prompt_data

    
def single_prompt(prompt_data, pack_function, db_map, use_oracle=False, mode="ch"):
    db_id = prompt_data["db_id"]
    schema = prompt_data["db_info"]
    database_path = db_map[db_id]
    args_tuple = (prompt_data, schema, database_path, use_oracle)
    return pack_function(args_tuple)


def debug(prompt_data, text, stream_handler, tokenizer, entity, skeleton, mode="ch"):
    def process_prediction(prompt, messages):
        """Handle the prompt and stream the generated data."""
        # Stream the data and get the prediction
        api_data = {
            "prompt": [prompt],
            "messages": messages,
        }
        api_data = {**api_data, **prompt_data["api_data"]}
        yield from stream_handler.stream_data({"prompt": [prompt]}, prompt_data["db_infos"], prompt_data["db_id"])
        # Get the prediction and extract SQL
        err_prediction = stream_handler.get_prediction()
        sql = extract_sql_from_text(err_prediction)
        print(f"err Prediction: {err_prediction}")
        print(f"SQL: {sql}")
        # If prediction or SQL is invalid, use the original text as fallback
        if not err_prediction or not sql:
            print(f"Invalid prediction: {err_prediction}")
            err_prediction = text
            sql = extract_sql_from_text(text)
        # Append the corrected prediction and SQL to the result
        prompt_data["prediction"].append({"text": err_prediction, "query": sql})
    
    prompt_data["prediction"] = []
    sql = extract_sql_from_text(text)
    db_path = prompt_data["db_path"]
    schema = prompt_data["db_info"]
    use_oracle = False
    prompt_data["prediction"].append({"text": text, "query": sql})

    err_input = {
        "db_id": prompt_data["db_id"],
        "question": prompt_data["question"],
        "rationale": prompt_data["rationale"],
        "alignment": prompt_data["alignment"],
        "hallucination": prompt_data["hallucination"],
        "history": prompt_data["history"],
    }
    # Process entity linking if entity is provided
    if entity:
        err_input["prediction"] = prompt_data["prediction"][-1]
        # print(err_input)
        prompt, messages, flag = prompt_entity_linking((err_input, schema, db_path, use_oracle), tokenizer, mode=mode)
        # print(prompt)
        yield from send_notice(flag, type="entity_debug_flag")
        if flag:
            yield from process_prediction(prompt, messages)

    # Process hallucination if skeleton is provided
    if skeleton:
        err_input["prediction"] = prompt_data["prediction"][-1]
        prompt, messages, flag = prompt_hallucination((err_input, schema, db_path, use_oracle), tokenizer, mode=mode)
        # print(prompt)
        yield from send_notice(flag, type="skeleton_debug_flag")
        if flag:
            yield from process_prediction(prompt, messages)

    # Process execution error
    err_input["prediction"] = prompt_data["prediction"][-1]
    prompt, messages, flag = prompt_execution_error((err_input, schema, db_path), tokenizer, mode=mode)
    # print(prompt)
    yield from send_notice(flag, type="debug_flag")
    if flag:
        yield from process_prediction(prompt, messages)

    # print(prompt_data["prediction"])

def test_debug():
    test_path = "/home/kyxu/demo-fb/cache/prompt_data.json"
    with open(test_path, "r", encoding="utf-8") as f:
        prompt_data = json.load(f)
    # print(prompt_data.keys())
    text = "select 书名id, 平台id from 电子书 where 电子书售价 = 100"
    prompt_data["db_path"] = "/home/kyxu/demo-t2s/dataset/Chase/database/购书平台/购书平台.sqlite"
    prompt_data["db_info"] = prompt_data["db_infos"][0]
    entity = True
    skeleton = True
    generator = debug(prompt_data, text, stream_handler, tokenizer, entity, skeleton, mode="ch")
    for chunk in generator:
        pass


if __name__ == '__main__':
    from utils.stream_handler import StreamDataHandler
    port = 5000
    gpu_node = "gpu16"
    predict_url = f"http://{gpu_node}:{port}/predict"
    stream_handler = StreamDataHandler(predict_url)
    llm_name_or_path = "./model/Qwen2.5-Coder/7b"
    tokenizer = AutoTokenizer.from_pretrained(llm_name_or_path)
    db_map = {
        "购书平台": "/home/kyxu/demo-t2s/dataset/Chase/database/购书平台/购书平台.sqlite"
    }
    prompt_data = {
    "question": "有电子书吗？售价多少？",
    "db_id": "购书平台",
    "history": [
        {
            "role": "user",
            "content": "我需要图书的详细信息。",
            "type": "question"
        },
        {
            "role": "ai",
            "content": "rationale: 为了获取图书的详细信息，我们需要查询图书表中的所有列，包括图书id、书名、作者和类型。\n\nSQL: select 图书id, 书名, 作者, 类型 from 图书",
            "type": "prediction"
        },
        {
            "role": "user",
            "content": "有电子书吗？售价多少？",
            "type": "feedback"
        }
    ],
    "db_info": {
            "table_names_original": [
                "图书",
                "平台",
                "图书与平台",
                "电子书"
            ],
            "table_names": [
                "图书",
                "平台",
                "图书与平台",
                "电子书"
            ],
            "column_names_original": [
                [
                    -1,
                    "*"
                ],
                [
                    0,
                    "图书id"
                ],
                [
                    0,
                    "书名"
                ],
                [
                    0,
                    "作者"
                ],
                [
                    0,
                    "类型"
                ],
                [
                    1,
                    "平台id"
                ],
                [
                    1,
                    "平台名"
                ],
                [
                    1,
                    "成立时间"
                ],
                [
                    1,
                    "年营业额"
                ],
                [
                    1,
                    "是否自营"
                ],
                [
                    1,
                    "会员费"
                ],
                [
                    2,
                    "书名id"
                ],
                [
                    2,
                    "平台id"
                ],
                [
                    2,
                    "售价"
                ],
                [
                    2,
                    "购买人数"
                ],
                [
                    2,
                    "评分"
                ],
                [
                    2,
                    "评分人数"
                ],
                [
                    2,
                    "加入购物车人数"
                ],
                [
                    2,
                    "收藏人数"
                ],
                [
                    2,
                    "缺货"
                ],
                [
                    3,
                    "书名id"
                ],
                [
                    3,
                    "平台id"
                ],
                [
                    3,
                    "电子书售价"
                ],
                [
                    3,
                    "会员价格"
                ],
                [
                    3,
                    "购买人数"
                ]
            ],
            "column_names": [
                [
                    -1,
                    "*"
                ],
                [
                    0,
                    "图书id"
                ],
                [
                    0,
                    "书名"
                ],
                [
                    0,
                    "作者"
                ],
                [
                    0,
                    "类型"
                ],
                [
                    1,
                    "平台id"
                ],
                [
                    1,
                    "平台名"
                ],
                [
                    1,
                    "成立时间"
                ],
                [
                    1,
                    "年营业额"
                ],
                [
                    1,
                    "是否自营"
                ],
                [
                    1,
                    "会员费"
                ],
                [
                    2,
                    "书名id"
                ],
                [
                    2,
                    "平台id"
                ],
                [
                    2,
                    "售价"
                ],
                [
                    2,
                    "购买人数"
                ],
                [
                    2,
                    "评分"
                ],
                [
                    2,
                    "评分人数"
                ],
                [
                    2,
                    "加入购物车人数"
                ],
                [
                    2,
                    "收藏人数"
                ],
                [
                    2,
                    "缺货"
                ],
                [
                    3,
                    "书名id"
                ],
                [
                    3,
                    "平台id"
                ],
                [
                    3,
                    "电子书售价"
                ],
                [
                    3,
                    "会员价格"
                ],
                [
                    3,
                    "购买人数"
                ]
            ],
            "column_types": [
                "text",
                "text",
                "text",
                "text",
                "text",
                "text",
                "text",
                "time",
                "number",
                "others",
                "number",
                "text",
                "text",
                "number",
                "number",
                "number",
                "number",
                "number",
                "number",
                "others",
                "text",
                "text",
                "number",
                "number",
                "number"
            ],
            "primary_keys": [
                1,
                5
            ],
            "foreign_keys": [
                [
                    12,
                    5
                ],
                [
                    11,
                    1
                ],
                [
                    21,
                    5
                ],
                [
                    20,
                    1
                ]
            ],
            "db_id": "购书平台"
        },
    "db_infos": [
        {
            "table_names_original": [
                "图书",
                "平台",
                "图书与平台",
                "电子书"
            ],
            "table_names": [
                "图书",
                "平台",
                "图书与平台",
                "电子书"
            ],
            "column_names_original": [
                [
                    -1,
                    "*"
                ],
                [
                    0,
                    "图书id"
                ],
                [
                    0,
                    "书名"
                ],
                [
                    0,
                    "作者"
                ],
                [
                    0,
                    "类型"
                ],
                [
                    1,
                    "平台id"
                ],
                [
                    1,
                    "平台名"
                ],
                [
                    1,
                    "成立时间"
                ],
                [
                    1,
                    "年营业额"
                ],
                [
                    1,
                    "是否自营"
                ],
                [
                    1,
                    "会员费"
                ],
                [
                    2,
                    "书名id"
                ],
                [
                    2,
                    "平台id"
                ],
                [
                    2,
                    "售价"
                ],
                [
                    2,
                    "购买人数"
                ],
                [
                    2,
                    "评分"
                ],
                [
                    2,
                    "评分人数"
                ],
                [
                    2,
                    "加入购物车人数"
                ],
                [
                    2,
                    "收藏人数"
                ],
                [
                    2,
                    "缺货"
                ],
                [
                    3,
                    "书名id"
                ],
                [
                    3,
                    "平台id"
                ],
                [
                    3,
                    "电子书售价"
                ],
                [
                    3,
                    "会员价格"
                ],
                [
                    3,
                    "购买人数"
                ]
            ],
            "column_names": [
                [
                    -1,
                    "*"
                ],
                [
                    0,
                    "图书id"
                ],
                [
                    0,
                    "书名"
                ],
                [
                    0,
                    "作者"
                ],
                [
                    0,
                    "类型"
                ],
                [
                    1,
                    "平台id"
                ],
                [
                    1,
                    "平台名"
                ],
                [
                    1,
                    "成立时间"
                ],
                [
                    1,
                    "年营业额"
                ],
                [
                    1,
                    "是否自营"
                ],
                [
                    1,
                    "会员费"
                ],
                [
                    2,
                    "书名id"
                ],
                [
                    2,
                    "平台id"
                ],
                [
                    2,
                    "售价"
                ],
                [
                    2,
                    "购买人数"
                ],
                [
                    2,
                    "评分"
                ],
                [
                    2,
                    "评分人数"
                ],
                [
                    2,
                    "加入购物车人数"
                ],
                [
                    2,
                    "收藏人数"
                ],
                [
                    2,
                    "缺货"
                ],
                [
                    3,
                    "书名id"
                ],
                [
                    3,
                    "平台id"
                ],
                [
                    3,
                    "电子书售价"
                ],
                [
                    3,
                    "会员价格"
                ],
                [
                    3,
                    "购买人数"
                ]
            ],
            "column_types": [
                "text",
                "text",
                "text",
                "text",
                "text",
                "text",
                "text",
                "time",
                "number",
                "others",
                "number",
                "text",
                "text",
                "number",
                "number",
                "number",
                "number",
                "number",
                "number",
                "others",
                "text",
                "text",
                "number",
                "number",
                "number"
            ],
            "primary_keys": [
                1,
                5
            ],
            "foreign_keys": [
                [
                    12,
                    5
                ],
                [
                    11,
                    1
                ],
                [
                    21,
                    5
                ],
                [
                    20,
                    1
                ]
            ],
            "db_id": "购书平台"
        }
    ],
    "demo": [
        {
            "schema": "create table 动物 (\n动物id TEXT,\n中文学名 TEXT,\n所属纲 TEXT,\n所属科 TEXT,\n食性 TEXT,\n濒危级别 TEXT,\n保护级别 TEXT,\nprimary key (动物id);\n)\n/*\nColumns in 动物 and 3 distinct examples in each column:\n动物id: item_animal_food_9_71, item_animal_food_9_72, item_animal_food_9_73;\n中文学名: 大象, 鸵鸟, 青蛙;\n所属纲: 哺乳纲, 鸟纲, 爬行纲;\n所属科: 象科, 猴科, 猫科;\n食性: 食肉, 食草, 杂食;\n濒危级别: 灭绝, 野生灭绝, 极危;\n保护级别: 一级, 二级;\n*/\n\ncreate table 城市 (\n城市id TEXT,\n城市 TEXT,\n气候地带 TEXT,\n所属国家 TEXT,\n所属洲 TEXT,\nprimary key (城市id);\n)\n/*\nColumns in 城市 and 3 distinct examples in each column:\n城市id: item_animal_food_9_66, item_animal_food_9_67, item_animal_food_9_68;\n城市: 清迈, 内比都, 莫斯科;\n气候地带: 热带, 亚热带, 温带;\n所属国家: 泰国, 缅甸, 俄罗斯;\n所属洲: 亚洲, 欧洲, 南美洲;\n*/\n\ncreate table 动物分布城市 (\n动物id TEXT,\n城市id TEXT,\n现存数量 number,\nforeign key (城市id) references 城市(城市id),\nforeign key (动物id) references 动物(动物id),\n)\n/*\nColumns in 动物分布城市 and 3 distinct examples in each column:\n动物id: item_animal_food_9_71, item_animal_food_9_73, item_animal_food_9_74;\n城市id: item_animal_food_9_70, item_animal_food_9_69, item_animal_food_9_67;\n现存数量: 10, 100000;\n*/\n\ncreate table 动物寓言故事 (\n动物id TEXT,\n寓言故事 TEXT,\n形象 TEXT,\nforeign key (动物id) references 动物(动物id),\n)\n/*\nColumns in 动物寓言故事 and 3 distinct examples in each column:\n动物id: item_animal_food_9_74, item_animal_food_9_73, item_animal_food_9_75;\n寓言故事: 兔死狐悲, 东郭先生, 狐假虎威;\n形象: 好, 坏;\n*/\n\ncreate table 动物电影 (\n影片id TEXT,\n影片名 TEXT,\n动物id TEXT,\n拍摄国家 TEXT,\n类型 TEXT,\nforeign key (动物id) references 动物(动物id),\nprimary key (影片id);\n)\n/*\nColumns in 动物电影 and 3 distinct examples in each column:\n影片id: item_animal_food_9_76, item_animal_food_9_77, item_animal_food_9_78;\n影片名: 忠犬八公的故事, 帝企鹅日记, 一条狗的使命;\n动物id: item_animal_food_9_74, item_animal_food_9_71, item_animal_food_9_73;\n拍摄国家: 美国, 日本, 德国;\n类型: 剧情, 纪录片, 科幻;\n*/",
            "related_schema": "create table 动物 (\n动物id TEXT,\n中文学名 TEXT,\n所属纲 TEXT,\n所属科 TEXT,\n食性 TEXT,\n濒危级别 TEXT,\n保护级别 TEXT,\nprimary key (动物id);\n)\n/*\nColumns in 动物 and 3 distinct examples in each column:\n动物id: item_animal_food_9_71, item_animal_food_9_72, item_animal_food_9_73;\n中文学名: 大象, 鸵鸟, 青蛙;\n所属纲: 哺乳纲, 鸟纲, 爬行纲;\n所属科: 象科, 猴科, 猫科;\n食性: 食肉, 食草, 杂食;\n濒危级别: 灭绝, 野生灭绝, 极危;\n保护级别: 一级, 二级;\n*/\n\ncreate table 动物分布城市 (\n动物id TEXT,\n城市id TEXT,\n现存数量 number,\nforeign key (动物id) references 动物(动物id),\n)\n/*\nColumns in 动物分布城市 and 3 distinct examples in each column:\n动物id: item_animal_food_9_71, item_animal_food_9_73, item_animal_food_9_74;\n城市id: item_animal_food_9_70, item_animal_food_9_69, item_animal_food_9_67;\n现存数量: 10, 100000;\n*/",
            "examples": [
                {
                    "question": "你知道有哪些动物已经濒临灭绝了吗？",
                    "query": "select T2.中文学名 from 动物分布城市 AS T1 JOIN 动物 AS T2 ON T1.动物id = T2.动物id where 现存数量 = (select min(现存数量) from 动物分布城市)"
                }
            ]
        },
        {
            "schema": "create table 车型 (\n车辆id TEXT,\n车型 TEXT,\n上牌时间 time,\n上牌地 TEXT,\n里程数 number,\n排量 number,\n过户记录 number,\n汽车品牌 TEXT,\nprimary key (车辆id);\n)\n/*\nColumns in 车型 and 3 distinct examples in each column:\n车辆id: 1, item_product_3_11, item_product_3_12;\n车型: X5, E100, 雅阁;\n上牌时间: 2015, 2017, 2010;\n上牌地: 北京, 上海, 天津;\n里程数: 4, 1, 10;\n排量: 4.8L, 2.4L;\n过户记录: 0, 2, 1;\n汽车品牌: 宝马, 奔驰, 本田;\n*/\n\ncreate table 车型平台售卖 (\n车辆id TEXT,\n平台 TEXT,\n售价 number,\n服务费比例 number,\nforeign key (车辆id) references 车型(车辆id),\n)\n/*\nColumns in 车型平台售卖 and 3 distinct examples in each column:\n车辆id: item_product_3_11, item_product_3_12, item_product_3_13;\n平台: 瓜子二手车, 优信二手车, 人人车;\n售价: 23, 35, 56;\n服务费比例: 5%, 10%, 8%;\n*/",
            "related_schema": "create table 车型 (\n车辆id TEXT,\n车型 TEXT,\n上牌时间 time,\n上牌地 TEXT,\n里程数 number,\n排量 number,\n过户记录 number,\n汽车品牌 TEXT,\nprimary key (车辆id);\n)\n/*\nColumns in 车型 and 3 distinct examples in each column:\n车辆id: 1, item_product_3_11, item_product_3_12;\n车型: X5, E100, 雅阁;\n上牌时间: 2015, 2017, 2010;\n上牌地: 北京, 上海, 天津;\n里程数: 4, 1, 10;\n排量: 4.8L, 2.4L;\n过户记录: 0, 2, 1;\n汽车品牌: 宝马, 奔驰, 本田;\n*/",
            "examples": [
                {
                    "question": "有哪些排量为4.8L的汽车车型？",
                    "query": "select 车型 from 车型 where 排量 = \"4.8L\""
                }
            ]
        },
        {
            "schema": "create table 公司 (\n公司id TEXT,\n公司名 TEXT,\n成立时间 time,\n位于城市 TEXT,\nprimary key (公司id);\n)\n/*\nColumns in 公司 and 3 distinct examples in each column:\n公司id: item_software_8_86, item_software_8_87, item_software_8_88;\n公司名: 新浪, 腾讯, 字节跳动;\n成立时间: 2000, 1995, 2013;\n位于城市: 北京, 深圳, 杭州;\n*/\n\ncreate table 功能 (\n功能id TEXT,\n名称 TEXT,\n简介 TEXT,\nprimary key (功能id);\n)\n/*\nColumns in 功能 and 3 distinct examples in each column:\n功能id: item_software_8_100, item_software_8_96, item_software_8_97;\n名称: 视频, 聊天, 新闻资讯;\n简介: 播放电视剧等视频资源, 用于朋友, 陌生人间聊天;\n*/\n\ncreate table 社交APP (\nappid TEXT,\napp名称 TEXT,\n软件大小 number,\n注册用户量 number,\n日活跃用户量 time,\n母公司id TEXT,\nforeign key (母公司id) references 公司(公司id),\nprimary key (appid);\n)\n/*\nColumns in 社交APP and 3 distinct examples in each column:\nappid: item_software_8_91, item_software_8_92, item_software_8_93;\napp名称: 微信, 微博, 多闪;\n软件大小: 30M, 90M;\n注册用户量: 3000000000, 100000000, 20000000;\n日活跃用户量: 1000000000, 400000000, 500000000;\n母公司id: item_software_8_87, item_software_8_89;\n*/\n\ncreate table APP支持的功能 (\n功能id TEXT,\nappid TEXT,\n是否主要功能 binary,\nforeign key (appid) references 社交APP(appid),\nforeign key (功能id) references 功能(功能id),\n)\n/*\nColumns in APP支持的功能 and 3 distinct examples in each column:\n功能id: item_software_8_98, item_software_8_100, item_software_8_99;\nappid: item_software_8_95, item_software_8_93;\n是否主要功能: 是, 否;\n*/",
            "related_schema": "create table 公司 (\n公司id TEXT,\n公司名 TEXT,\n成立时间 time,\n位于城市 TEXT,\nprimary key (公司id);\n)\n/*\nColumns in 公司 and 3 distinct examples in each column:\n公司id: item_software_8_86, item_software_8_87, item_software_8_88;\n公司名: 新浪, 腾讯, 字节跳动;\n成立时间: 2000, 1995, 2013;\n位于城市: 北京, 深圳, 杭州;\n*/",
            "examples": [
                {
                    "question": "哪些公司成立的更早？",
                    "query": "select 公司名 from 公司 where 成立时间 < (select 成立时间 from 公司 where 公司名 = \"字节跳动\")"
                }
            ]
        }
    ],
    "schema": "create table 图书 (\n图书id TEXT,\n书名 TEXT,\n作者 TEXT,\n类型 TEXT,\nprimary key (图书id);\n)\n/*\nColumns in 图书 and 3 distinct examples in each column:\n图书id: item_book.2_2_16, item_book.2_2_17, item_book.2_2_18;\n书名: 平凡的世界, 巴菲特的估值逻辑, 半小时世界漫画史;\n作者: 路遥, 芦叶飞, 李碧龙;\n类型: 小说, 经管, 社科;\n*/\n\ncreate table 平台 (\n平台id TEXT,\n平台名 TEXT,\n成立时间 time,\n年营业额 number,\n是否自营 binary,\n会员费 number,\nprimary key (平台id);\n)\n/*\nColumns in 平台 and 3 distinct examples in each column:\n平台id: item_book.2_2_11, item_book.2_2_12, item_book.2_2_13;\n平台名: 亚马逊, 京东, 当当;\n成立时间: 1995-09-07, 1998-04-06, 2000-09-06;\n年营业额: 10亿, 14亿, 20亿;\n是否自营: 是, 否;\n会员费: 128元, 140元, 150元;\n*/\n\ncreate table 图书与平台 (\n书名id TEXT,\n平台id TEXT,\n售价 number,\n购买人数 number,\n评分 number,\n评分人数 number,\n加入购物车人数 number,\n收藏人数 number,\n缺货 binary,\nforeign key (平台id) references 平台(平台id),\nforeign key (书名id) references 图书(图书id),\n)\n/*\nColumns in 图书与平台 and 3 distinct examples in each column:\n书名id: item_book.2_2_19, item_book.2_2_16;\n平台id: item_book.2_2_14, item_book.2_2_15, item_book.2_2_12;\n售价: 20元, 50元, 100元;\n购买人数: 100, 50000, 7000000;\n评分: 5, 6, 7;\n评分人数: 1000, 120000, 400000;\n加入购物车人数: 1000, 50000, 540000;\n收藏人数: 1000, 23000, 500000;\n缺货: 是, 否;\n*/\n\ncreate table 电子书 (\n书名id TEXT,\n平台id TEXT,\n电子书售价 number,\n会员价格 number,\n购买人数 number,\nforeign key (平台id) references 平台(平台id),\nforeign key (书名id) references 图书(图书id),\n)\n/*\nColumns in 电子书 and 3 distinct examples in each column:\n书名id: item_book.2_2_16, item_book.2_2_17, item_book.2_2_20;\n平台id: item_book.2_2_12, item_book.2_2_15, item_book.2_2_11;\n电子书售价: 2.99元, 10.2元, 11.33元;\n会员价格: 0.99元, 4.35元, 4.74元;\n购买人数: 100, 50000, 600000;\n*/",
    "related_schema": "create table 电子书 (\n书名id TEXT,\n平台id TEXT,\n电子书售价 number,\n会员价格 number,\n购买人数 number,\n)\n/*\nColumns in 电子书 and 3 distinct examples in each column:\n书名id: item_book.2_2_16, item_book.2_2_17, item_book.2_2_20;\n平台id: item_book.2_2_12, item_book.2_2_15, item_book.2_2_11;\n电子书售价: 2.99元, 10.2元, 11.33元;\n会员价格: 0.99元, 4.35元, 4.74元;\n购买人数: 100, 50000, 600000;\n*/",
    "query_pred": "select 书名id, 平台id, 电子书售价 from 电子书",
    "alignment": {}
}
    test_debug()
    # print("begin align")
    # prompt_data = align(prompt_data, stream_handler, tokenizer)
    # # prompt = generate_prompt(prompt_data, tokenizer, "ch", align_flag=True)
    # # print(prompt)
    # print("begin hallucinate")
    # prompt_data = hallucinate(prompt_data, stream_handler, tokenizer)

    # cache_path = "./cache/prompt_data.json"
    # with open(cache_path, "w", encoding="utf-8") as f:
    #     json.dump(prompt_data, f, ensure_ascii=False, indent=4)


