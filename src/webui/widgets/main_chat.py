import sys
import json
import os
import sqlite3
import requests
import tempfile
import time
import pandas as pd
import threading
import streamlit as st
import numpy as np
from collections.abc import Iterable

from .history import StreamDataHandler, ChatHistory
from ..utils.database import database_to_string
from ..utils.my_logger import MyLogger
from ..utils.file import save_session_json, load_session_json, save_uploadfiles, ds_sqlite_files, get_save_dir
from ..utils.extract_db import extract_db_info
from ...text.front import text

lang = st.query_params.get("lang", "zh")
context = text[lang]["main_chat"]
sys.path.append(".")
server_port = os.getenv("SERVER_PORT", 53683)
server_url = f"http://localhost:{server_port}/generate_sql"
stream_handler = StreamDataHandler(server_url, lang=lang)
history = ChatHistory(lang=lang)
cache_path = os.getenv("CACHE_PATH", "cache")
logger = MyLogger(name="main", log_file="logs/text2sql_demo.log")


def generate_sql_query(question: str, db_map: dict, db_infos: list[dict], message_block):
    """Generate SQL query from the given question."""
    prompts_user = []
    files = {}
    for file_name, db_path in db_map.items():
        db_file = open(db_path, "rb")
        files[file_name] = db_file
        prompt_user = database_to_string(db_path, question=question)
        prompts_user.append(
            {"db_id": file_name.split(".")[0], "prompt_user": prompt_user}
        )
        
    his_mess = [
        history[i]
        for i in range(len(history))
        if history[i].get("type") not in ("execution", "notice")
    ]
    user_type = history[-1].get("type")
    data = {
        "question": question,
        "history": json.dumps(his_mess, ensure_ascii=False),
        "prompts_user": json.dumps(prompts_user, ensure_ascii=False),
        "db_infos": json.dumps(db_infos, ensure_ascii=False),
        "cache_path": get_save_dir(user=True, chat=True),
        "demonstration": st.session_state.get("demonstration", False),
        "demo_num": st.session_state.get("demo_num", 5),
        "question_num": st.session_state.get("question_num", 5),
        "pre_generate_sql": st.session_state.get("pre_generate_sql", False),
        "self_debug": st.session_state.get("self_debug", False),
        "encore": st.session_state.get("encore_flag", False) and st.session_state.get("encore", False),
        "user_type": user_type,
        "entity_debug": st.session_state.get("entity_debug", False),
        "skeleton_debug": st.session_state.get("skeleton_debug", False),
        "align_flag": st.session_state.get("align_flag", False),
        "mode": lang,
        "model_name": st.session_state.get("model_name", "Qwen2.5-Coder_7b"),
        "api_key": st.session_state.get("api_key", None),
        "api_base": st.session_state.get("api_base", None),
    }

    response = stream_handler.stream_data(data, files, message_block)
    return response


def display_chat_history(db_infos: list[dict], db_map: dict):
    """Display the chat history."""
    history.show_history()

    # If last message is not from assistant, generate a new response
    if history[-1]["role"] == "user":
        user_mess = history[-1]
        with st.chat_message("ai"):
            message_block = st.container()
            question = user_mess["content"]
            response = generate_sql_query(
                question, db_map, db_infos, message_block)
            message_block.write_stream(response)

        prediction = stream_handler.get_prediction()
        ai_mess = {
            "role": "ai",
            "content": prediction["prediction"],
            "sql": {
                "query": prediction["query"],
                "database": prediction["database"],
            },
            "type": "prediction",
            "db_map": db_map,
        }
        print(ai_mess)
        history.add_message(ai_mess)
        history.show_execution(len(history) - 1)
        if st.session_state["LOGGED_IN"]:
            current_chat = st.session_state["history_chats"][st.session_state["current_chat_index"]]
            save_dir = os.path.join(
                cache_path, st.session_state['username'], current_chat)
            current_db_map = ds_sqlite_files(save_dir, db_map)
            st.session_state["current_db_map"] = current_db_map
            save_session_json(st.session_state, save_dir)
        st.rerun()

def init_page():
    st.write(context["header_text"])
    st.info(
        context["info_text"],
        icon="📃",
    )


def user_input_area(files_map):
    """Handle user input area and append messages to session state."""
    files = list(files_map.values())
    prompt = st.chat_input("Ask a question")
    if prompt:
        message = {"role": "user", "content": prompt,
                   "files": files, "type": "question"}
        history.add_message(message)

def main_area(files_map: dict):
    global lang 
    global context
    global history
    global stream_handler
    lang = st.query_params.get("lang", "zh")
    context = text[lang]["main_chat"]
    stream_handler = StreamDataHandler(server_url, lang=lang)
    history = ChatHistory(lang=lang)
    init_page()

    # logger.info(f"uploaded files: {files_map}")

    if files_map:
        db_infos = []
        for db_name, db_path in files_map.items():
            db_name = db_name.split(".")[0]
            db_info = extract_db_info(db_path, db_name)
            db_infos.append(db_info)

        # main area for conversation
        # input_area
        user_input_area(files_map)
        display_chat_history(db_infos, files_map)
    else:
        st.info(context["info_upload_text"])
