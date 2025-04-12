import requests
import json
import sqlite3
import streamlit as st
import pandas as pd
import time
from typing import Optional
from ...text.front import text


# lang = st.query_params["lang"] if "lang" in st.query_params else "en"
# context = text[lang]["history"]

def gen_generator(data):
    if isinstance(data, str):
        chunk_size = 10
        for i in range(0, len(data), chunk_size):
            yield data[i : i + chunk_size]
            time.sleep(0.000001)
    else:
        yield data


def get_rationale_sql(prediction):
    """获取解释性 SQL 语句"""
    content = prediction.get("content").lower()
    if "rationale:" in content and "sql:" in content:
        rationale = content.split("rationale:")[1].split("sql:")[0].strip()
    else:
        rationale = content
    if "rationale" not in prediction:
        prediction["rationale"] = rationale
    return prediction

class StreamDataHandler:
    def __init__(self, url, lang="zh"):
        """
        初始化 StreamDataHandler 实例
        :param url: 服务器的 URL，用于发送和接收数据
        """
        self.url = url
        self.prediction: Optional[dict] = None  # 用于存储预测结果
        self.rationale: Optional[str] = None  # 用于存储解释性 SQL 语句
        
        self.lang = lang
        self.context = text[self.lang]["history"]

    def stream_data(self, data, files, message_placeholder):
        """
        从服务器获取流式数据，并以生成器方式传输数据。
        """
        self.prediction = None
        self.rationale = None
        response = None
        try:
            response = requests.post(self.url, data=data, files=files, stream=True)
            response.raise_for_status()  # 添加响应状态码检查

            with message_placeholder, st.status(self.context["status_text"], expanded=True) as status:
                
                flag_i = 0
                for line in response.iter_lines(decode_unicode=True):
                    if not line.strip():  # 处理空行
                        continue

                    # 解析 JSON 数据
                    if line.startswith("data:"):
                        try:
                            json_data = line.split(":", 1)[1].strip()
                            if not json_data:
                                continue
                            data_dict = json.loads(json_data)
                            print(data_dict)
                        except (json.JSONDecodeError, IndexError) as e:
                            status.text(f"解析错误: {str(e)}")  # 捕获解析错误并更新状态
                            continue

                        # 根据不同的键处理不同类型的数据
                        if "prediction" in data_dict:
                            self.prediction = data_dict  # 记录总结数据
                        # elif "rationale" in data_dict:
                        #     self.rationale = data_dict["rationale"]
                        elif "generator" in data_dict:
                            yield data_dict["generator"]  # 流式输出生成器的数据
                        elif "notice" in data_dict:
                            status.text(data_dict["notice"])  # 更新状态消息
                        # elif "debug_flag" in data_dict:
                        #     if not data_dict["debug_flag"]:
                        #         status.text(self.context["status_debug_success"])
                        #     else:
                        #         flag_i += 1
                        #         status.text(self.context["status_debug_fail"])
                        #         yield self.context["status_debug_fail_yield"].format(flag_i=flag_i)
                        elif data_dict.get("entity_debug_flag", False):
                            status.text(self.context["status_entity_debug"])
                            yield self.context["status_entity_debug_yield"]
                        elif data_dict.get("skeleton_debug_flag", False):
                            status.text(self.context["status_skeleton_debug"])
                            yield self.context["status_skeleton_debug_yield"]
                        elif 'debug_flag' in data_dict:
                            if data_dict['debug_flag']:
                                status.text(self.context["status_execution_debug"])
                                yield self.context["status_execution_debug_yield"]
                            else: continue
                        elif "done" in data_dict:
                            status.text(data_dict["done"])  # 处理完成的状态更新
                            status.update(
                                label="Download complete!",
                                state="complete",
                                expanded=False,
                            )
                            break  # 结束流式处理
                        elif "error" in data_dict:
                            raise Exception(data_dict["error"])  # 处理错误消息
                        else:
                            status.text(f"未知数据格式: {data_dict}")  # 处理未预料的数据格式
                    else:
                        continue

        except requests.RequestException as e:
            # 捕获网络请求错误
            with message_placeholder:
                print(e)
                st.error(self.context["request_error"])
            st.stop()
        except Exception as e:
            # 捕获所有其他异常
            err_msgs = text[self.lang]["error"]
            err_type = str(e)
            if err_type in err_msgs:
                e = err_msgs[err_type]
            with message_placeholder:
                st.error(self.context["other_error"].format(e=e))
            st.stop()
        finally:
            if response:
                response.close()  # 确保响应被正确关闭


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
