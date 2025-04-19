import requests
import json
import sqlite3
import streamlit as st
import pandas as pd
import time
from typing import Optional
from ...text.front import text

class StreamDataHandler:
    def __init__(self, url, lang="zh"):
        self.url = url
        self.prediction = None
        self.rationale = None
        self.lang = lang
        self.context = text[self.lang]["history"]

    def stream_data(self, data, files, message_placeholder):
        """简化后的流式数据处理逻辑"""
        self.prediction = self.rationale = None
        
        try:
            with requests.post(self.url, data=data, files=files, stream=True) as response:
                response.raise_for_status()
                
                with message_placeholder, st.status(self.context["status_text"], expanded=True) as status:
                    for line in self._process_lines(response, status):
                        yield line

        except requests.RequestException as e:
            self._handle_error(message_placeholder, "request_error", str(e))
        except Exception as e:
            self._handle_error(message_placeholder, "other_error", str(e))

    def _process_lines(self, response, status):
        """处理每行流数据"""
        for line in response.iter_lines(decode_unicode=True):
            if not line.strip():
                continue

            if line.startswith("data:"):
                yield from self._process_data_line(line, status)

    def _process_data_line(self, line, status):
        """处理单条数据行"""
        try:
            data = json.loads(line.split(":", 1)[1].strip())
        except (json.JSONDecodeError, IndexError) as e:
            status.text(f"解析错误: {e}")
            return

        # 数据路由处理
        if "prediction" in data:
            self.prediction = data
        elif "generator" in data:
            yield data["generator"]
        elif "notice" in data:
            status.text(data["notice"])
        elif data.get("done"):
            status.update(label="Download complete!", state="complete", expanded=False)
            raise StopIteration  # 终止流
        elif data.get("error"):
            raise Exception(data["error"])
        else:
            yield from self._handle_special_flags(data, status)

    def _handle_special_flags(self, data, status):
        """处理各种调试标志"""
        flag_handlers = {
            "entity_debug_flag": ("status_entity_debug", "status_entity_debug_yield"),
            "skeleton_debug_flag": ("status_skeleton_debug", "status_skeleton_debug_yield"),
            "debug_flag": ("status_execution_debug", "status_execution_debug_yield")
        }

        # print(data)
        for flag, (status_key, yield_key) in flag_handlers.items():
            # print(data.get(flag))
            if data.get(flag) is not None:
                if data.get(flag):
                    status.text(self.context[status_key])
                    yield self.context[yield_key]
                return

        status.text(f"未知数据格式: {data}")

    def _handle_error(self, placeholder, error_key, error_msg):
        """统一错误处理"""
        with placeholder:
            st.error(self.context[error_key].format(
                e=text[self.lang]["error"].get(error_msg, error_msg)
            ))
        st.stop()


    def get_prediction(self):
        """
        返回总结数据。
        """
        self.prediction["rationale"] = self.rationale
        return self.prediction


# chat history
class ChatHistory:
    def __init__(self, lang="zh"):
        self.lang = lang
        self.context = text[self.lang]["history"]
        # 初始化历史记录
        if "messages" not in st.session_state:
            st.session_state.messages = [
                {
                    "role": "ai",
                    "content": self.context["message_init"],
                    "type": "notice",
                }
            ]
        self.history = st.session_state.messages

    def init(self):
        st.session_state.messages = [
            {
                "role": "ai",
                "content": self.context["message_init"],
                "type": "notice",
            }
        ]

    def add_message(self, dict):
        """添加新消息并更新历史记录"""
        st.session_state.messages.append(dict)
        self.update_history()

    # 更新历史记录
    def update_history(self):
        """同步更新历史记录"""
        if "messages" not in st.session_state:
            self.init()
        self.history = st.session_state.messages

    def __iter__(self):
        return iter(self.history)

    def __getitem__(self, idx):
        return self.history[idx]

    def __len__(self):
        return len(self.history)

    def execute_sql(self, idx):
        """执行 SQL 并返回执行结果的消息"""
        msg = self.history[idx]
        db_name = msg["sql"]["database"]
        query = msg["sql"]["query"]
        db_path = msg["db_map"].get(db_name)

        try:
            # 使用 with 语句自动管理资源释放
            with sqlite3.connect(db_path) as conn:
                content = pd.read_sql(query, conn)
        except sqlite3.Error as e:
            # st.error(f"数据库错误: {e}")
            content = f"数据库错误: {e}"
        except Exception as e:
            # st.error(f"SQL 执行错误: {e}")
            content = f"SQL 执行错误: {e}"

        return {
            "role": "ai",
            "content": content,
            "type": "execution",
            "show_flag": True,
        }
    
    def update_messages(self):
        st.session_state.messages = self.history

    def insert_exe_msg(self, idx, msg):
        """在指定位置插入或替换执行结果消息"""
        if idx >= len(self.history):
            self.add_message(msg)
        elif self.history[idx].get("type") == "execution":
            self.history.pop(idx)
            self.history.insert(idx, msg)
        else:
            self.history.insert(idx, msg)
        self.update_messages()
        # print("insert_exe_msg", self.history)

    def show_execution(self, idx):
        """显示执行按钮和结果"""
        placeholder = st.empty()
        columns = st.columns(4)
        with columns[0]:
            exe_button = st.button(self.context["sql_excution"], key=f"execute_{idx}")
        with columns[1]:
            del_placeholder = st.empty()

        if exe_button:
            # print("执行SQL语句")
            placeholder.empty()
            exe_mass = self.execute_sql(idx)
            # print(exe_mass)
            self.insert_exe_msg(idx + 1, exe_mass)

        self.update_history()
        if (
            idx + 1 < len(self.history)
            and self.history[idx + 1].get("type") == "execution"
        ):
            exe_mass = self.history[idx + 1]
            if exe_mass.get("show_flag"):
                with del_placeholder:
                    del_button = st.button(self.context["sql_excution_delete"], key=f"delete_{idx}")
            if exe_mass.get("show_flag") and del_button:
                exe_mass["show_flag"] = False
                placeholder.empty()
                del_placeholder.empty()
                    
            if exe_mass.get("show_flag"):
                with placeholder.container():
                    with st.chat_message("ai"):
                        # content = msg_data_to_write(exe_mass["content"])
                        content = exe_mass["content"]
                        st.write(content)

    def show_message(self, idx):
        """显示聊天消息"""
        msg = self.history[idx]
        # print(msg)
        if msg.get("type") == "execution":
            return
        with st.chat_message(msg["role"]):
            if msg["role"] == "ai":
                if msg.get("type") == "prediction":
                    sql = msg.get("sql")
                    database = sql.get("database") if sql else None
                    query = sql.get("query") if sql else None
                    rationale = msg.get("rationale", "")
                    # print(rationale)
                    if database:
                        st.write(self.context["message_show_db"].format(database=database))
                    # if rationale:
                    #     st.write(f"推理过程: {rationale}")
                    # if query:
                    #     st.write(f"SQL查询语句如下: ")
                    #     st.code(query, language="sql")
                    # return 
            st.write(msg["content"])
        if msg["role"] == "ai" and ("sql" in msg and "db_map" in msg):
            self.show_execution(idx)
