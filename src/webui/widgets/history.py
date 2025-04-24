import requests
import json
import sqlite3
import streamlit as st
import pandas as pd
import time
from typing import Optional, Dict, Any, List, Generator, Union
from ...text.front import text


class StreamDataHandler:
    """
    处理流式数据的类，用于接收和处理服务器返回的流式响应。
    
    主要功能：
    1. 发送POST请求并处理流式响应
    2. 解析响应数据并提取有用信息
    3. 处理各种类型的响应和错误情况
    """
    
    def __init__(self, url: str, lang: str = "zh"):
        """
        初始化StreamDataHandler
        
        Args:
            url: 请求的URL
            lang: 语言设置，默认为中文
        """
        self.url = url
        self.prediction = None
        self.rationale = None
        self.lang = lang
        self.context = text[self.lang]["history"]

    def stream_data(self, data: Dict[str, Any], files: Dict[str, Any], message_placeholder) -> Generator:
        """
        发送POST请求并处理流式响应
        
        Args:
            data: 请求数据
            files: 请求文件
            message_placeholder: Streamlit占位符，用于显示状态
            
        Yields:
            处理后的数据行
        """
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

    def _process_lines(self, response, status) -> Generator:
        """
        处理响应中的每一行数据
        
        Args:
            response: 响应对象
            status: 状态对象
            
        Yields:
            处理后的数据
        """
        for line in response.iter_lines(decode_unicode=True):
            if not line.strip():
                continue

            if line.startswith("data:"):
                yield from self._process_data_line(line, status)

    def _process_data_line(self, line: str, status) -> Generator:
        """
        处理单条数据行
        
        Args:
            line: 数据行
            status: 状态对象
            
        Yields:
            处理后的数据
        """
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

    def _handle_special_flags(self, data: Dict[str, Any], status) -> Generator:
        """
        处理各种调试标志
        
        Args:
            data: 数据字典
            status: 状态对象
            
        Yields:
            处理后的数据
        """
        flag_handlers = {
            "entity_debug_flag": ("status_entity_debug", "status_entity_debug_yield"),
            "skeleton_debug_flag": ("status_skeleton_debug", "status_skeleton_debug_yield"),
            "debug_flag": ("status_execution_debug", "status_execution_debug_yield")
        }

        for flag, (status_key, yield_key) in flag_handlers.items():
            if data.get(flag) is not None:
                if data.get(flag):
                    status.text(self.context[status_key])
                    yield self.context[yield_key]
                return

        status.text(f"未知数据格式: {data}")

    def _handle_error(self, placeholder, error_key: str, error_msg: str) -> None:
        """
        统一错误处理
        
        Args:
            placeholder: Streamlit占位符
            error_key: 错误键
            error_msg: 错误消息
        """
        with placeholder:
            st.error(self.context[error_key].format(
                e=text[self.lang]["error"].get(error_msg, error_msg)
            ))
        st.stop()

    def get_prediction(self) -> Dict[str, Any]:
        """
        返回总结数据
        
        Returns:
            包含预测和推理的字典
        """
        if self.prediction:
            self.prediction["rationale"] = self.rationale
            return self.prediction
        return {}


class ChatHistory(object):
    """
    聊天历史记录管理类，用于存储、显示和操作聊天消息。
    """
    def __init__(self, lang: str = "zh"):
        """初始化聊天历史记录。"""
        self.lang = lang
        self.context = text[self.lang]["history"]
        self._initialize_history()
        self.history: List[Dict[str, Any]] = st.session_state.messages

    def _initialize_history(self) -> None:
        """初始化或重置消息历史到初始状态。"""
        if "messages" not in st.session_state:
            st.session_state.messages = [
                {
                    "role": "ai",
                    "content": self.context["message_init"],
                    "type": "notice",
                }
            ]

    def add_message(self, message: Dict[str, Any]) -> None:
        """添加新消息到历史记录并更新会话状态。"""
        st.session_state.messages.append(message)
        self.history = st.session_state.messages

    def __iter__(self):
        """使类可迭代，直接返回会话状态中的消息迭代器。"""
        return iter(st.session_state.messages)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        """支持索引访问会话状态中的消息。"""
        return st.session_state.messages[idx]

    def __len__(self) -> int:
        """返回会话状态中历史记录的长度。"""
        return len(st.session_state.messages)

    def _execute_sql(self, msg: Dict[str, Any]) -> Any:
        """执行SQL查询并返回结果或错误信息。"""
        db_name = msg["sql"]["database"]
        query = msg["sql"]["query"]
        db_path = msg["db_map"].get(db_name)
        try:
            with sqlite3.connect(db_path) as conn:
                return pd.read_sql(query, conn)
        except sqlite3.Error as e:
            return f"数据库错误: {e}"
        except Exception as e:
            return f"SQL 执行错误: {e}"
    
    def update_messages(self) -> None:
        """更新会话状态中的消息"""
        st.session_state.messages = self.history

    def insert_exe_msg(self, idx: int, msg: Dict[str, Any]) -> None:
        """
        在指定位置插入或替换执行结果消息
        
        Args:
            idx: 插入位置
            msg: 消息字典
        """
        if idx >= len(self.history):
            self.add_message(msg)
        elif self.history[idx].get("type") == "execution":
            self.history.pop(idx)
            self.history.insert(idx, msg)
        else:
            self.history.insert(idx, msg)
        self.update_messages()

    def show_execution(self, idx: int) -> None:
        """显示SQL执行按钮和结果（如果存在）。"""
        placeholder = st.empty()
        columns = st.columns(4)
        with columns[0]:
            if st.button(self.context["sql_excution"], key=f"execute_{idx}"):
                placeholder.empty()
                exe_mass = {
                    "role": "ai",
                    "content": self._execute_sql(self[idx]),  # 执行SQL并获取结果
                    "type": "execution",
                    "show_flag": True,
                }
                self.insert_exe_msg(idx + 1, exe_mass)
        with columns[1]:
            del_placeholder = st.empty()

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
                        content = exe_mass["content"]
                        st.write(content)

    def show_message(self, idx: int) -> None:
        """显示单条聊天消息，并根据消息类型显示额外组件。"""
        msg = self[idx]
        if msg.get("type") == "execution":
            return
            
        with st.chat_message(msg["role"]):
            if msg["role"] == "ai" and msg.get("type") == "prediction" and "sql" in msg:
                if "database" in msg["sql"]:
                    st.write(self.context["message_show_db"].format(database=msg["sql"]["database"]))
            st.write(msg["content"])
            
        if msg["role"] == "ai" and ("sql" in msg and "db_map" in msg):
            self.show_execution(idx)

    def show_history(self) -> None:
        """显示完整的聊天历史记录。"""
        for i in range(len(self)):
            self.show_message(i)
