import os
import sqlite3
import pandas as pd
import streamlit as st
from ..utils.file import save_session_json, get_save_dir, save_sqlite_from_zip, del_uploadfiles
from ..utils.my_logger import MyLogger
from ...text.front import text

logger = MyLogger(name="show_db", log_file="logs/text2sql_demo.log")

lang = st.query_params.get("lang", "zh")
context = text[lang]["show_db"]

def load_database(file: str) -> tuple[sqlite3.Connection, str]:
    """Load the uploaded SQLite database into a connection."""
    return sqlite3.connect(file)


def get_tables(conn: sqlite3.Connection) -> list[str]:
    """Retrieve all table names from the database."""

    cursor = conn.cursor()
    cursor.execute("SELECT name FROM sqlite_master WHERE type='table';")
    return [row[0] for row in cursor.fetchall()]


def get_table_content(conn: sqlite3.Connection, table_name: str) -> pd.DataFrame:
    """Get content of a specific table."""

    cursor = conn.cursor()
    cursor.execute(f"SELECT * FROM {table_name};")
    return pd.DataFrame(
        cursor.fetchall(), columns=[col[0] for col in cursor.description]
    )


def display_table_content(conn: sqlite3.Connection, tables: list[str]):
    """Display the content of a selected table."""

    if tables:
        selected_table = st.selectbox(context["select_table_text"], tables)
        if selected_table:
            content = get_table_content(conn, selected_table)
            st.write(context["select_table_content_text"].format(selected_table=selected_table))
            st.dataframe(content)
    else:
        st.write(context["select_table_no_table"])


def del_db_expand(files_map: dict):
    # 使用 expander 创建一个可折叠的区域，列出所有数据库文件
    with st.expander(context["uploaded_files_text"]):
        file_names = list(files_map.keys())
        selected_files = []

        # 为每个文件创建一个复选框
        for file_name in file_names:
            if st.checkbox(context["uploaded_files_checkbox"].format(file_name=file_name), key=f"checkbox_{file_name}"):
                selected_files.append(file_name)

        if selected_files:
            st.write(context["uploaded_files_selected"].format(selected_files=', '.join(selected_files)))

        # 显示删除选中的文件按钮
        if st.button(context["uploaded_files_delete"]):
            del_uploadfiles(selected_files)
            save_session_json(st.session_state, get_save_dir(user=True, chat=True))


def show_database():
    global lang
    global context
    lang_chat = st.query_params["lang"] if "lang" in st.query_params else "zh"
    context = text[lang_chat]["show_db"]

    st.header(context["show_db_text"])
    # choose a SQLite file
    uploaded_files = st.file_uploader(
        context["show_db_upload"], type=["sqlite", "zip"], accept_multiple_files=False, key="upload_db",
        help=context["show_db_upload_help"]
    )
    # 打开 ZIP 文件以二进制模式读取
    with open(context["db_example"], "rb") as file:
        # 添加下载按钮
        st.download_button(
            label=context["show_db_download"],
            data=file,
            file_name="example.zip",
            mime="application/zip"
        )
    files_map = {}
    if uploaded_files:
        # print(uploaded_files)
        upload = st.button(context["show_db_upload_button"])
        # st.write(st.session_state.upload_db)
        # 如果上传按钮被点击，保存上传的文件，并更新session_state
        if upload:
            uploaded_files = [uploaded_files]
            files_map = save_sqlite_from_zip(uploaded_files, get_save_dir(user=True, chat=True))
            # st.session_state.upload_db.clear()
            logger.debug(f"已上传 {len(files_map)} 个文件, files_map: {files_map}")
            st.session_state.current_db_map |= files_map
            save_session_json(st.session_state,
                              get_save_dir(user=True, chat=True))

    files_map = st.session_state.current_db_map
    logger.debug(f"当前数据库文件: {files_map}")
    if files_map:
        del_db_expand(files_map)
        # 获取上传文件的名称列表
        file_names = list(files_map.keys())
        # 使用selectbox选择一个数据库文件
        selected_db_name = st.selectbox(context["show_db_select"], file_names)
        # 找到对应的文件对象
        selected_db = files_map.get(selected_db_name)
        if selected_db:
            # 加载并显示选择的数据库内容
            conn = load_database(selected_db)
            tables = get_tables(conn)
            display_table_content(conn, tables)
    else:
        st.write(context["show_db_no_file"])
    return files_map
