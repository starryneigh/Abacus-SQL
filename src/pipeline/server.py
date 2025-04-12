import os
import json
import time
import subprocess
import requests
import logging
import sys
import threading
from flask import Flask, request, jsonify, Response, stream_with_context
from transformers import AutoTokenizer
from .utils.my_logger import MyLogger
from .utils.prompt_gen import generate_prompt, construct_demos, construct_schemas
from .utils.convert_data import get_table_and_question
from .utils.stream_handler import StreamDataHandler
from .murre import embd_tables, retrieve, score_data, construct_db_input
from .utils.database import database_to_string
from .fused import (
    pro_fused,
    merge_json_files_in_folder,
    extract_sqlite_metadata_from_folder,
)
from .utils.extract_from_sql import extract_sql_from_text, get_rationale
from .dac import align, hallucinate, debug
from .utils.notice import send_notice
from ..text.back import text_back
from .utils.server_thread import UvicornServerThread


logger = MyLogger("sserver", "logs/text2sql.log")


def get_db_files(request, save_dir="./cache"):
    if len(request.files) == 0:
        return jsonify({"error": "No files uploaded"}), 400

    files_dict = {}
    for file_key in request.files:
        file = request.files[file_key]
        # 确保文件名不为空
        if file.filename == "":
            continue  # 忽略没有文件名的文件
        file_save_path = os.path.join(save_dir, file_key + ".sqlite")
        file.save(file_save_path)
        files_dict[file_key] = file_save_path
    logger.debug(f"Files saved: {files_dict}")
    return files_dict


def murre_process(user_question, db_map, cache_path="./cache"):
    """
    处理 SQL 数据生成的过程，包括检索和评分。
    """
    data_file = os.path.join(cache_path, "data.json")
    get_table_and_question(data_file, cache_path)

    tables_file = os.path.join(cache_path, "tables.json")
    embd_path = os.path.join(cache_path, "embedding.json")
    embd_tables(tables_file, embd_path)

    retrieve_dir = os.path.join(cache_path, "retrieve", "turn0")
    os.makedirs(retrieve_dir, exist_ok=True)
    que_file = os.path.join(cache_path, "question.json")
    retrieve_path = os.path.join(retrieve_dir, "retrieved.json")
    retrieve(embd_path, que_file, retrieve_path)

    score_path = os.path.join(cache_path, "scored.json")
    score_data([retrieve_dir], score_path)

    schema, cho_db = construct_db_input(score_path, user_question, db_map)
    return cho_db


def link_table(prompt_data, db_map, tokenizer, stream_handler, mode="ch"):
    prompt, messages = generate_prompt(
        prompt_data=prompt_data, tokenizer=tokenizer, mode=mode, demo_related=False, schema_related=False
    )
    api_data = prompt_data["api_data"]
    api_data = {**api_data, "messages": messages}
    # print(f"link prompt: {prompt}")
    try:
        prediction = stream_handler.generate_with_llm([prompt], api_data=api_data)[0]
    except Exception as e:
        raise e
    prediction = prediction[0][0].strip()
    # print(f"link prediction: {prediction}")
    schema_strs = []
    # 提取表格，生成schema
    db_info_map = {}
    for table in prompt_data["db_infos"]:
        db_info_map[table["db_id"]] = table

    sql = extract_sql_from_text(prediction)
    prompt_data["query_pred"] = sql
    db_path = db_map[prompt_data["db_id"]]
    # 找到db_infos中的db_id对应的数据库表格
    schema = db_info_map[prompt_data["db_id"]]
    # print(f"sql: {sql}")
    # print(f"db_path: {db_path}")
    # print(f"schema: {schema}")
    schema_str = database_to_string(
        db_path, granularity="table", sql=sql, schema=schema
    )
    # print(f"schema_str: {schema_str}")
    if schema_str == "":
        print("No table found")
        schema_str = prompt_data["schema"]
    schema_strs.append(schema_str)
    prompt_data["related_schema"] = schema_str
    return prompt_data


def generate_response(
    data: dict,
    stream_handler: StreamDataHandler,
    demos,
    demos_en,
    cache_path: str = "./cache",
    tokenizer=None,
    demo_db_path: str = "./dataset/Spider/database",
    demo_db_path_en: str = "./dataset/Spider/database",
):
    """
    生成响应流，通知用户进度并返回 SQL 预测结果。
    """

    mode = data.get("mode", "ch")
    model_name = data.get("model_name", "Qwen2.5-Coder_7b")
    api_key = data.get("api_key", "None")
    api_base = data.get("api_base", "None")
    demo_flag = str_to_bool(data["demonstration"])
    demo_num = int(data["demo_num"])
    question_num = int(data["question_num"])
    pre_generate_sql = str_to_bool(data["pre_generate_sql"])
    self_debug = str_to_bool(data["self_debug"])
    encore_flag = str_to_bool(data["encore"])
    entity_debug = str_to_bool(data.get("entity_debug", "True"))
    skeleton_debug = str_to_bool(data.get("skeleton_debug", "True"))
    align_flag = str_to_bool(data.get("align_flag", "True"))

    api_data = {
        "model_name": model_name,
        "api_key": api_key,
        "api_base": api_base,
    }

    if mode == "zh":
        mode = "ch"
    print(mode)

    context = text_back[mode]
    demos_chose = demos if mode == "ch" else demos_en
    demo_db_path_chose = demo_db_path if mode == "ch" else demo_db_path_en
    
    logger.info(
        f"demo_flag: {demo_flag}, demo_num: {demo_num}, question_num: {question_num}, \
        pre_generate_sql: {pre_generate_sql}, self_debug: {self_debug}, encore_flag: {encore_flag}, \
        entity_debug: {entity_debug}, skeleton_debug: {skeleton_debug}, align_flag: {align_flag}"
    )

    yield from send_notice(context["notice_find"])
    db_map = data["db_id_path_map"]
    cho_db = murre_process(
        user_question=data["user_question"],
        db_map=db_map,
        cache_path=cache_path,
    )

    yield from send_notice(context["notice_find_success"].format(cho_db=cho_db))
    id_info_map = {}
    for table in data["db_infos"]:
        id_info_map[table["db_id"]] = table
    db_info = id_info_map[cho_db]
    prompt_data = {
            "api_data": api_data,
            "question": data["user_question"],
            "db_id": cho_db,
            "db_info": db_info,
            "history": data["history"],
            "db_infos": data["db_infos"],
            "db_path": data["db_id_path_map"][cho_db],
            "demo": [],
            "schema": "",
            "related_schema": "",
            "query_pred": "",
            "alignment": {},
            "hallucination": "",
        }
    if demo_flag:
        if encore_flag:
            yield from send_notice(context["notice_use_encore"])
            encore_file = os.path.join(cache_path, "demo.json")
            prompt_data = construct_demos(
                prompt_data,
                demo_db_path_chose,
                demos_chose,
                question_num,
                demo_num,
                encore_file,
                mode=mode,
            )
        else:
            # print("aaa")
            prompt_data = construct_demos(
                prompt_data, demo_db_path_chose, demos_chose, question_num, demo_num, mode=mode
            )
        # print(f"demo: {prompt_data[0]['demo']}")
    prompt_data = construct_schemas(prompt_data, db_map)

    yield from send_notice(context["notice_presql"])
    try:
        prompt_data = link_table(
            prompt_data, db_map, tokenizer, stream_handler, mode=mode
        )
    except Exception as e:
        err_dict = {"error": str(e)}
        print(f"err_dict: {err_dict}")
        yield f"data: {json.dumps(err_dict)}\n\n"
        return
    if not pre_generate_sql:
        prompt_data["related_schema"] = prompt_data["schema"]
    
    if align_flag or entity_debug:
        yield from send_notice(context["notice_entity"])
        prompt_data = align(prompt_data, stream_handler, tokenizer, mode=mode)

    if skeleton_debug:
        yield from send_notice(context["notice_sk"])
        prompt_data = hallucinate(prompt_data, stream_handler, tokenizer, mode=mode)

    yield from send_notice(context["notice_sql_gen"])
    prompt, messages = generate_prompt(
        prompt_data=prompt_data,
        tokenizer=tokenizer,
        mode=mode,
        demo_related=True,
        schema_related=True,
        align_flag=align_flag,
    )
    # print(f"prompt: {prompt}")
    api_data = {
        "prompt": [prompt],
        "messages": messages,
        "api_key": api_key,
        "api_base": api_base,
        "model_name": model_name,
    }
    yield from stream_handler.stream_data(api_data, data["db_infos"], cho_db=cho_db)

    prediction = stream_handler.get_prediction()
    prompt_data["rationale"] = get_rationale(prediction)
    # # 发送rationale
    # chunk = {"rationale": prompt_data["rationale"]}
    # yield f"data: {json.dumps(chunk)}\n\n"

    dump_prompt_path = os.path.join(cache_path, "prompt_data.json")

    if self_debug:
        yield from send_notice(context["notice_debug"])
        yield from debug(prompt_data, prediction, stream_handler, tokenizer, entity_debug, skeleton_debug, mode=mode)
    
    dump_prompt_path = os.path.join(cache_path, "prompt_data.json")
    with open(dump_prompt_path, "w") as f:
        json.dump(prompt_data, f, indent=4, ensure_ascii=False)


def str_to_bool(string):
    return string.lower() == "true"

def create_server_app(
    predict_url,
    demos,
    demos_en,
    cache_path="./cache",
    model_name_or_path="./model/Qwen/7b",
    demo_db_path="./dataset/Spider/database",
    demo_db_path_en="./dataset/Spider/database",
):
    app = Flask(__name__)
    os.makedirs(cache_path, exist_ok=True)
    tokenizer = AutoTokenizer.from_pretrained(model_name_or_path)
    stream_handler = StreamDataHandler(predict_url)

    @app.route("/generate_sql", methods=["POST"])
    def generate_sql():
        try:
            user_question = request.form["question"]
            history = json.loads(request.form["history"])
            prompts_user = json.loads(request.form["prompts_user"])
            db_infos = json.loads(request.form["db_infos"])
            cache_path = request.form.get("cache_path")
            logger.debug(f"cache_path: {cache_path}")
            os.makedirs(cache_path, exist_ok=True)
            db_file = get_db_files(request, cache_path)

            data = {
                "user_question": user_question,
                "history": history,
                "db_id_path_map": db_file,
                "prompts_user": prompts_user,
                "db_infos": db_infos,
                "cache_path": cache_path,
                "demonstration": request.form.get("demonstration", "False"),
                "demo_num": request.form.get("demo_num", 5),
                "question_num": request.form.get("question_num", 5),
                "pre_generate_sql": request.form.get("pre_generate_sql", "False"),
                "self_debug": request.form.get("self_debug", "False"),
                "encore": request.form.get("encore"),
                "entity_debug": request.form.get("entity_debug", "True"),
                "skeleton_debug": request.form.get("skeleton_debug", "True"),
                "align_flag": request.form.get("align_flag", "True"),
                "mode": request.form.get("mode", "ch"),
                "model_name": request.form.get("model_name", "Qwen2.5-Coder_7b"),
                "api_key": request.form.get("api_key", "None"),
                "api_base": request.form.get("api_base", "None"),
            }

            dump_path = os.path.join(cache_path, "data.json")
            with open(dump_path, "w") as f:
                json.dump(data, f)

            return Response(
                stream_with_context(
                    generate_response(
                        data,
                        stream_handler,
                        demos,
                        demos_en,
                        cache_path,
                        tokenizer,
                        demo_db_path,
                        demo_db_path_en,
                    )
                ),
                content_type="text/event-stream",
            )

        except Exception as e:
            logger.error(f"Error in /generate_sql: {e}")
            return jsonify({"error": str(e)}), 500

    @app.route("/encore", methods=["POST"])
    def encore():
        import zipfile

        try:
            uploaded_file = request.files["file"]
            logger.info(f"uploaded_file: {uploaded_file}")
            cache_path = request.form.get("cache_path")
            mode = request.form.get("mode", "ch")
            if mode == "zh":
                mode = "ch"
            context = text_back[mode]
            logger.info(f"cache_path: {cache_path}")
            # 定义保存路径
            save_path = os.path.join(cache_path, "fused/")
            os.makedirs(save_path, exist_ok=True)
            file_path = os.path.join(save_path, "uploaded.zip")
            # 保存上传的ZIP文件
            uploaded_file.save(file_path)
        except Exception as e:
            return jsonify({"error": str(e)}), 400
        # 解压缩ZIP文件
        try:
            with zipfile.ZipFile(file_path, "r") as zip_ref:
                zip_ref.extractall(save_path)
        except zipfile.BadZipFile:
            return "Error: Bad ZIP file", 400

        def gen_encore_response():
            to_fused_path = os.path.join(save_path, "to_fused.json")
            cache_fused_path = os.path.join(save_path, "fused/")
            encoder_name_or_path = "./model/SGPT/125m"
            table_path = os.path.join(save_path, "tables", "tables.json")
            database_path = os.path.join(save_path, "database")

            merge_json_files_in_folder(save_path, to_fused_path, recursive=False)
            os.makedirs(os.path.dirname(table_path), exist_ok=True)
            tables = extract_sqlite_metadata_from_folder(save_path)
            with open(table_path, "w", encoding="utf-8") as f:
                json.dump(tables, f, indent=4, ensure_ascii=False)

            yield from send_notice(context["notice_encore_init"])
            yield from pro_fused(
                to_fused_path=to_fused_path,
                cache_path=cache_fused_path,
                encoder_name_or_path=encoder_name_or_path,
                table_path=table_path,
                database_path=database_path,
                stream_handler=stream_handler,
                to_path=cache_path,
                num_turn=2,
                cluster_number=4,
            )
            yield from send_notice(context["notice_encore_success"])
        
        try:
            return Response(
                stream_with_context(gen_encore_response()),
                content_type="text/event-stream",
            )
        except Exception as e:
            return jsonify({"error": str(e)}), 500

    return app


def start_server(
    predict_url,
    demos,
    demos_en,
    host="127.0.0.1",
    flask_port=53683,
    cache_path="./cache",
    model_name_or_path="./model/Qwen/7b",
    demo_db_path="./dataset/Spider/database",
    demo_db_path_en="./dataset/Spider/database",
):
    app = create_server_app(
        predict_url=predict_url,
        cache_path=cache_path,
        model_name_or_path=model_name_or_path,
        demos=demos,
        demos_en=demos_en,
        demo_db_path=demo_db_path,
        demo_db_path_en=demo_db_path_en,
    )
    server_thread = threading.Thread(target=app.run, kwargs={"host": host, "port": flask_port})
    server_thread.start()
    return server_thread


def parser():
    import argparse

    args = argparse.ArgumentParser()
    args.add_argument("--predict_port", type=int, default=5000)
    args.add_argument("--flask_port", type=int, default=53683)
    args.add_argument("--gpu_node", type=str, default="gpu16")
    args.add_argument("--model_name_or_path", type=str, default="./model/Qwen2.5-Coder/7b")
    args.add_argument("--cache_path", type=str, default="./cache")
    args.add_argument(
        "--demos_file", type=str, default="./dataset/Cosql/prepare/demo.json"
    )
    args.add_argument(
        "--demos_file_en", type=str, default="./dataset/Spider/prepare/demo.json"
    )
    args.add_argument("--db_path", type=str, default="./dataset/Cosql/database")
    args.add_argument("--db_path_en", type=str, default="./dataset/Spider/database")
    args.add_argument("--shot", type=int, default=1)
    return args.parse_args()


if __name__ == "__main__":
    args = parser()
    port = args.predict_port
    gpu_node = args.gpu_node
    model_name_or_path = args.model_name_or_path
    cache_path = args.cache_path
    demos_file = args.demos_file
    demos_file_en = args.demos_file_en
    demo_db_path = args.db_path
    demo_db_path_en = args.db_path_en
    SHOT = args.shot
    flask_port = args.flask_port
    predict_url = f"http://{gpu_node}:{port}/predict"

    with open(demos_file, "r", encoding="utf-8") as f:
        demos = json.load(f)
        print(demos[0])
    with open(demos_file_en, "r", encoding="utf-8") as f:
        demos_en = json.load(f)
        print(demos_en[0])

    start_server(
        predict_url=predict_url,
        host="127.0.0.1",
        flask_port=53683,
        cache_path=cache_path,
        model_name_or_path=model_name_or_path,
        demos=demos,
        demos_en=demos_en,
        demo_db_path=demo_db_path,
        demo_db_path_en=demo_db_path_en,
        SHOT=SHOT,
    )
    while True:
        print("Server is running")
        time.sleep(20)
