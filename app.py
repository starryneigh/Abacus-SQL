import streamlit as st
st.set_page_config(
    page_title="Abacus-SQL",
    page_icon="🦉",
    layout="centered",
    initial_sidebar_state="auto",
    menu_items=None,
)
st.markdown(
    """
    <style>
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    </style>
    """,
    unsafe_allow_html=True
)
import os
import json
import traceback
from dotenv import load_dotenv

from src.webui.login_ui import Login
from src.webui.widgets.show_db import show_database
from src.webui.widgets.main_chat import main_area
from src.webui.widgets.chat import show_chats
from src.webui.widgets.config import show_config
from src.webui.widgets.strategy import show_strategy
from src.webui.utils.my_logger import MyLogger
from src.webui.utils.file import get_save_dir, load_session_json, get_history_chats
from src.text.front import *


lang = st.query_params.get("lang", "zh")
content = text[lang]["app"]
logger = MyLogger(name="Text2SQL Demo", log_file="logs/text2sql_demo.log")
server_thread = None
model_thread = None

def extract_lang():
    if "lang" in st.session_state:
        return
    header = st.context.headers.get("Accept-Language", "en")
    # print(f"Accept-Language: {header}")
    # 解析语言和优先级
    languages = []
    for lang in header.split(","):
        parts = lang.split(";q=")
        language = parts[0].strip()
        q_value = float(parts[1]) if len(parts) > 1 else 1.0  # 默认 q=1.0
        languages.append((language, q_value))

    # 按 q 值排序，降序
    languages.sort(key=lambda x: x[1], reverse=True)

    # 提取最优先语言
    lang = languages[0][0]
    if "zh" in lang:
        lang = "zh"
    else:
        lang = "en"
    st.session_state.lang = lang
    st.query_params["lang"] = lang
    print(f"Extracted language: {st.session_state.lang}")

def verify_env():
    print(os.getenv("COURIER_AUTH_TOKEN", ""))

# 使用 st.cache_resource 确保服务器只启动一次
@st.cache_resource
def start(): 
    load_dotenv()
    verify_env()
    global server_thread
    global model_thread
    try:
        from src.pipeline.server import start_server
        from src.pipeline.model_server import start_model
        # from murre import init
        model_name_or_path = os.getenv("MODEL_NAME_OR_PATH", "./model/Qwen2.5-Coder/7b")
        config_file = os.getenv("CONFIG_FILE", "./config/Qwen2.5-Coder.json")
        server_host = os.getenv("SERVER_HOST", "127.0.0.1")  # 默认端口53683
        server_port = int(os.getenv("SERVER_PORT", 53683))  # 默认端口53683
        model_host = os.getenv("MODEL_HOST", "0.0.0.0")  # 默认端口5000
        model_port = int(os.getenv("MODEL_PORT", 5000))  # 默认端口5000
        demos_file = os.getenv("DEMO_FILE", "./data/dataset/Chase/prepare/demo.json")
        demo_db_path = os.getenv("DEMO_DB_PATH", "./data/dataset/Chase/database")
        demos_file_en = os.getenv("DEMO_FILE_EN", "./data/dataset/Spider/prepare/demo.json")
        demo_db_path_en = os.getenv("DEMO_DB_PATH_EN", "./data/dataset/Spider/database")
        cache_path = os.getenv("CACHE_PATH", "./cache")
        predict_url = f"http://localhost:{model_port}/predict"
        with open(demos_file, "r", encoding="utf-8") as f:
            demos = json.load(f)
            print(f"Loaded {len(demos)} demos from {demos_file}")
        with open(demos_file_en, "r", encoding="utf-8") as f:
            demos_en = json.load(f)
            print(f"Loaded {len(demos_en)} demos from {demos_file_en}")
        print(f"Starting server at {server_port}...")
        server_thread = start_server(
            predict_url,
            demos,
            demos_en,
            host=server_host,
            flask_port=server_port,
            cache_path=cache_path,
            model_name_or_path=model_name_or_path,
            demo_db_path=demo_db_path,
            demo_db_path_en=demo_db_path_en
        )
        print(f"Starting model at {model_port}...")
        model_thread = start_model(
            model_name_or_path=model_name_or_path,
            config_file=config_file,
            port=model_port,
            host=model_host,
        )
    except Exception as e:
        print(f"Failed to start server: {e}")
        traceback.print_exc()
        return False
    print("Server started.")
    return True


def init_sess_state():
    if "initial_settings" not in st.session_state or not st.session_state["initial_settings"]:
        logger.info("Initial settings not found. Initializing...")
        user_cache_path = get_save_dir(user=True, chat=False)
        st.session_state.history_chats = get_history_chats(user_cache_path)
        st.session_state.messages = [
            {
                "role": "ai",
                "content": content["init_msg"],
                "type": "notice",
            }
        ]
        st.session_state.current_chat_index = 0
        # 初始化当前选择的历史聊天窗口（用于检测更改）
        st.session_state.previous_chat = 0
        st.session_state.current_db_map = {}

        # 加载会话状态
        cache_path = get_save_dir(user=True, chat=True)
        load_session_json(st.session_state, cache_path)
        st.session_state["initial_settings"] = True
        logger.info("Initial settings loaded.")


def show_sidebar(login_obj):
    global lang
    global content
    lang = st.query_params.get("lang", "zh")
    content = text[lang]["app"]

    with st.sidebar:
        chat_area, db_area, config_area, strategy_area = st.tabs(
            [content["chat"], content["database"], content["config"], content["strategy"]])

    with chat_area:
        chat_container = st.container()
        show_chats(chat_container)
    with db_area:
        files_map = show_database()
    with config_area:
        show_config(login_obj)
    with strategy_area:
        show_strategy()

    return files_map


def login_sidebar():
    lang = st.query_params.get("lang", "zh")
    # print(f"Current language: {lang}")
    content = text[lang]["app"]
    side_placeholder = st.sidebar.empty()
    with side_placeholder:
        langs = ["中文", "English"]
        if lang == "zh":
            default_lang_index = langs.index("中文")
        else:
            default_lang_index = langs.index("English")
        lang = st.selectbox(content["lang_choose"], langs, index=default_lang_index, key="select_lang", on_change=set_lang)
        # if lang == "English":
        #     st.session_state.lang = "en"
        # else:
        #     st.session_state.lang = "zh"
    return side_placeholder

def set_lang():
    # print(f"session_state: {st.session_state.select_lang}")
    if "select_lang" in st.session_state:
        if st.session_state.select_lang == "中文":
            select_lang = "zh"
        else:
            select_lang = "en"
        # print(f"Language changed: {select_lang}")
        st.query_params["lang"] = select_lang


def main():
    # 启动服务器
    flag = start()
    # print(f"Server started: {flag}")
    extract_lang()

    side_placeholder = login_sidebar()

    login_obj = Login(
        auth_token=os.getenv("COURIER_AUTH_TOKEN", ""),
    )

    LOGGED_IN = login_obj.build_login_ui()

    if LOGGED_IN:
        side_placeholder.empty()
        init_sess_state()
        files_map = show_sidebar(login_obj)
        main_area(files_map)


if __name__ == "__main__":
    main()