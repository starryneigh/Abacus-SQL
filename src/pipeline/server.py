import os
import json
import time
import logging
import zipfile
from anyio import to_thread
from fastapi.responses import StreamingResponse, JSONResponse
from fastapi import FastAPI, Request, Depends, UploadFile, HTTPException
from transformers import AutoTokenizer
from .utils.my_logger import MyLogger
from .utils.prompt_gen import generate_prompt, construct_demos, construct_schemas
from .utils.stream_handler import StreamDataHandler
from .utils.database import database_to_string
from .Fused.fused import (
    pro_fused,
    merge_json_files_in_folder,
    extract_sqlite_metadata_from_folder,
)
from .utils.extract_from_sql import extract_sql_from_text, get_rationale
from .DAC.dac import align, hallucinate, debug
from .utils.notice import send_notice
from ..text.back import text_back
from .utils.server_thread import UvicornServerThread
from .utils.data import GenerateSQLRequest
from .Murre.murre import murre_process


logger = MyLogger("sserver", "logs/text2sql.log")


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
    data: GenerateSQLRequest,
    stream_handler: StreamDataHandler,
    demos,
    demos_en,
    tokenizer=None,
    demo_db_path: str = "./dataset/Spider/database",
    demo_db_path_en: str = "./dataset/Spider/database",
):
    """
    生成响应流，通知用户进度并返回 SQL 预测结果。
    """
    context = text_back[data.mode]
    demos_chose = demos if data.mode == "ch" else demos_en
    demo_db_path_chose = demo_db_path if data.mode == "ch" else demo_db_path_en

    yield from send_notice(context["notice_find"])
    cho_db = murre_process(
        user_question=data.question,
        db_map=data.db_id_path_map,
        cache_path=data.cache_path,
    )

    yield from send_notice(context["notice_find_success"].format(cho_db=cho_db))
    id_info_map = {}
    for table in data.db_infos:
        id_info_map[table["db_id"]] = table
    db_info = id_info_map[cho_db]
    prompt_data = {
            "api_data": data.api_data,
            "question": data.question,
            "db_id": cho_db,
            "db_info": db_info,
            "history": data.history,
            "db_infos": data.db_infos,
            "db_path": data.db_id_path_map[cho_db],
            "demo": [],
            "schema": "",
            "related_schema": "",
            "query_pred": "",
            "alignment": {},
            "hallucination": "",
        }
    if data.demonstration:
        if data.encore:
            yield from send_notice(context["notice_use_encore"])
            encore_file = os.path.join(data.cache_path, "demo.json")
            prompt_data = construct_demos(
                prompt_data,
                demo_db_path_chose,
                demos_chose,
                data.question_num,
                data.demo_num,
                encore_file,
                mode=data.mode,
            )
        else:
            # print("aaa")
            prompt_data = construct_demos(
                prompt_data, demo_db_path_chose, demos_chose, data.question_num, data.demo_num, mode=data.mode
            )
        # print(f"demo: {prompt_data[0]['demo']}")
    prompt_data = construct_schemas(prompt_data, data.db_id_path_map)

    yield from send_notice(context["notice_presql"])
    try:
        prompt_data = link_table(
            prompt_data, data.db_id_path_map, tokenizer, stream_handler, mode=data.mode
        )
    except Exception as e:
        err_dict = {"error": str(e)}
        print(f"err_dict: {err_dict}")
        yield f"data: {json.dumps(err_dict)}\n\n"
        return
    if not data.pre_generate_sql:
        prompt_data["related_schema"] = prompt_data["schema"]
    
    if data.align_flag or data.entity_debug:
        yield from send_notice(context["notice_entity"])
        prompt_data = align(prompt_data, stream_handler, tokenizer, mode=data.mode)

    if data.skeleton_debug:
        yield from send_notice(context["notice_sk"])
        prompt_data = hallucinate(prompt_data, stream_handler, tokenizer, mode=data.mode)

    yield from send_notice(context["notice_sql_gen"])
    prompt, messages = generate_prompt(
        prompt_data=prompt_data,
        tokenizer=tokenizer,
        mode=data.mode,
        demo_related=True,
        schema_related=True,
        align_flag=data.align_flag,
    )
    # print(f"prompt: {prompt}")
    api_data = {
        "prompt": [prompt],
        "messages": messages,
        **data.api_data,
    }
    yield from stream_handler.stream_data(api_data, data.db_infos, cho_db=cho_db)

    prediction = stream_handler.get_prediction()
    prompt_data["rationale"] = get_rationale(prediction)
    # # 发送rationale
    # chunk = {"rationale": prompt_data["rationale"]}
    # yield f"data: {json.dumps(chunk)}\n\n"

    dump_prompt_path = os.path.join(data.cache_path, "prompt_data.json")

    if data.self_debug:
        yield from send_notice(context["notice_debug"])
        yield from debug(prompt_data, prediction, stream_handler, tokenizer, data.entity_debug, data.skeleton_debug, mode=data.mode)
    
    dump_prompt_path = os.path.join(data.cache_path, "prompt_data.json")
    with open(dump_prompt_path, "w") as f:
        json.dump(prompt_data, f, indent=4, ensure_ascii=False)

async def get_generate_sql_payload(request: Request) -> GenerateSQLRequest:
    form_data = await request.form()
    save_dir = form_data.get("cache_path", "./cache")
    os.makedirs(save_dir, exist_ok=True)

    files_dict = {}
    for key, value in form_data.items():
        if value.__class__.__name__ == "UploadFile": 
            file: UploadFile = value
            if file.filename:
                file_save_path = os.path.join(save_dir, key + ".sqlite")
                try:
                    contents = await file.read()
                    with open(file_save_path, "wb") as f:
                        f.write(contents)
                    files_dict[key] = file_save_path
                    logging.debug(f"File '{file.filename}' saved to: {file_save_path}")
                except Exception as e:
                    logging.error(f"Error saving file '{file.filename}': {e}")
                    raise HTTPException(status_code=500, detail=f"Error saving file '{file.filename}': {str(e)}")
            else:
                logging.warning(f"Ignored file with empty filename for key: {key}")

    data = GenerateSQLRequest(**form_data)
    data.api_data = {
        "model_name": data.model_name,
        "api_key": data.api_key,
        "api_base": data.api_base,
    }
    data.db_id_path_map = files_dict
    if data.mode == "zh":
        data.mode = "ch"
    dump_path = os.path.join(data.cache_path, "data.json")
    with open(dump_path, "w") as f:
        json.dump(data.model_dump(), f, indent=4, ensure_ascii=False)
    return data

def safe_next(gen):
    """包装next()调用，返回标记值代替抛出异常"""
    try:
        return next(gen)
    except StopIteration:
        return None  # 返回哨兵值

def create_server_app(
    predict_url,
    demos,
    demos_en,
    cache_path="./cache",
    model_name_or_path="./model/Qwen/7b",
    demo_db_path="./dataset/Spider/database",
    demo_db_path_en="./dataset/Spider/database", 
):
    app = FastAPI()
    os.makedirs(cache_path, exist_ok=True)
    tokenizer = AutoTokenizer.from_pretrained(model_name_or_path)
    stream_handler = StreamDataHandler(predict_url)

    @app.post("/generate_sql")
    async def generate_sql(payload: GenerateSQLRequest = Depends(get_generate_sql_payload)):
        """
        生成 SQL 的接口。接收用户问题、历史记录、数据库信息等，
        使用 GenerateSQLRequest 数据类接收请求的 form 数据。
        """
        async def event_stream():
            sync_generator = generate_response(
                payload,
                stream_handler,
                demos=demos,
                demos_en=demos_en,
                tokenizer=tokenizer,
                demo_db_path=demo_db_path,
                demo_db_path_en=demo_db_path_en
            )
            try:
                while True:
                    try:
                        content = await to_thread.run_sync(
                            lambda: safe_next(sync_generator),
                            abandon_on_cancel=True
                        )
                        if content == None:
                            break
                        yield content
                    except Exception as e:
                        logging.error(f"Generator error: {e}")
                        break
            finally:
                if hasattr(sync_generator, "close"):
                    sync_generator.close()

        return StreamingResponse(event_stream(), media_type="text/event-stream")

    @app.post("/encore")
    async def encore(request: Request):
        """
        处理上传的 ZIP 文件，并进行后续处理 (流式响应)。
        """
        try:
            form_data = await request.form()
            file = form_data.get("file")
            cache_path = form_data.get("cache_path")
            mode = form_data.get("mode", "ch")
            if mode == "zh":
                mode = "ch"
            context = text_back[mode]
            logging.info(f"cache_path: {cache_path}")
            # 定义保存路径
            save_path = os.path.join(cache_path, "fused/")
            os.makedirs(save_path, exist_ok=True)
            file_path = os.path.join(save_path, "uploaded.zip")
            # 保存上传的 ZIP 文件
            with open(file_path, "wb") as f:
                content = await file.read()  # 使用 await 读取文件内容
                f.write(content)
        except Exception as e:
            return JSONResponse({"error": str(e)}, status_code=400)
        # 解压缩 ZIP 文件
        try:
            with zipfile.ZipFile(file_path, "r") as zip_ref:
                zip_ref.extractall(save_path)
        except zipfile.BadZipFile:
            return JSONResponse({"error": "Error: Bad ZIP file"}, status_code=400)

        async def async_gen_encore_response():
            # 初始化路径（同步操作）
            to_fused_path = os.path.join(save_path, "to_fused.json")
            cache_fused_path = os.path.join(save_path, "fused/")
            table_path = os.path.join(save_path, "tables", "tables.json")
            database_path = os.path.join(save_path, "database")

            # 同步操作封装到线程池
            merge_json_files_in_folder(save_path, to_fused_path, recursive=False)
            os.makedirs(os.path.dirname(table_path), exist_ok=True)
            
            tables = extract_sqlite_metadata_from_folder(save_path)
            with open(table_path, "w", encoding="utf-8") as f:
                json.dump(tables, f, indent=4, ensure_ascii=False)

            # 处理同步生成器流式输出
            sync_gen = gen_encore_stream(
                to_fused_path, 
                cache_fused_path,
                table_path,
                database_path
            )
            
            try:
                while True:
                    try:
                        content = await to_thread.run_sync(
                            lambda: safe_next(sync_gen),
                            abandon_on_cancel=True
                        )
                        if content is None:
                            break
                        yield content
                    except Exception as e:
                        logging.error(f"Generator error: {e}")
                        break
            finally:
                if hasattr(sync_gen, "close"):
                    sync_gen.close()

        def gen_encore_stream(to_fused_path, cache_fused_path, table_path, database_path):
            """同步生成器核心逻辑（保持原样）"""
            yield from send_notice(context["notice_encore_init"])
            
            yield from pro_fused(
                to_fused_path=to_fused_path,
                cache_path=cache_fused_path,
                encoder_name_or_path=os.getenv("ENCODER_NAME_OR_PATH", "./model/SGPT/125m"),
                table_path=table_path,
                database_path=database_path,
                stream_handler=stream_handler,
                to_path=cache_path,
                num_turn=2,
                cluster_number=4,
            )
            
            yield from send_notice(context["notice_encore_success"])

        try:
            return StreamingResponse(
                async_gen_encore_response(),
                media_type="text/event-stream"
            )
        except Exception as e:
            logging.error(f"Endpoint error: {e}")
            raise HTTPException(status_code=500, detail=str(e))
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
    server_thread = UvicornServerThread(app, host, flask_port)
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
