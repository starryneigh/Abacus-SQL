import os
import json
import requests
import configparser
import streamlit as st
from ..utils.file import save_session_json, get_save_dir, encore_validate_zip
from ...text.front import text

lang = st.query_params.get("lang", "zh")
context = text[lang]["strategy"]

config = configparser.ConfigParser()
config.read('config/config.ini')
if "server_url" in config:
     if "port" in config["server_url"]:
         server_port = config["server_url"]["port"]

def init_strategy():
    if "init_strategy" not in st.session_state or not st.session_state.init_strategy:
        print("init_strategy")
        st.session_state.demonstration = True
        st.session_state.demo_num = 3
        st.session_state.question_num = 0
        st.session_state.pre_generate_sql = True
        st.session_state.self_debug = True
        st.session_state.encore = False
        st.session_state.encore_flag = False
        st.session_state.init_strategy = True
        st.session_state.entity_debug = False
        st.session_state.skeleton_debug = False
        st.session_state.align_flag = False

def show_strategy():
    global lang 
    global context
    lang = st.query_params.get("lang", "zh")
    context = text[lang]["strategy"]

    init_strategy()
    st.write(context["strategy_title"])
    show_encore()
    st.toggle(context["strategy_demo"], key="demonstration", value=st.session_state.demonstration)
    if st.session_state.demonstration:
        if st.session_state.encore:
            st.toggle(context["strategy_encore"], key="encore_flag", value=st.session_state.encore_flag)
        st.session_state.demo_num = st.slider(
            context["strategy_demo_num"], 
            min_value=1, 
            max_value=10, 
            value=st.session_state.demo_num
        )
        # st.session_state.question_num = st.slider(
        #     "question 个数", 
        #     min_value=1, 
        #     max_value=10, 
        #     value=st.session_state.question_num
        # )
    st.toggle(context["strategy_presql"], key="pre_generate_sql", value=st.session_state.pre_generate_sql)
    # st.toggle("实体链接", key="align_flag", value=st.session_state.align_flag)
    st.toggle(context["strategy_self_debug"], key="self_debug", value=st.session_state.self_debug)
    # if st.session_state.self_debug:
    #     st.toggle("entity-debug", key="entity_debug", value=st.session_state.entity_debug)
    #     st.toggle("skeleton-debug", key="skeleton_debug", value=st.session_state.skeleton_debug)
    cache_path = get_save_dir(user=True, chat=True)
    save_session_json(st.session_state, cache_path)

def pro_response(response):
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
            except (json.JSONDecodeError, IndexError) as e:
                continue

            if "notice" in data_dict:
                st.write(data_dict["notice"])
        else:
            continue

def stream_progress_from_server(response):
    # 处理 Server-Sent Events (SSE)
    for line in response.iter_lines():
        if line:
            decoded_line = line.decode('utf-8')
            if decoded_line.startswith("data:"):
                # 提取 process 值
                data = decoded_line.lstrip("data: ")
                info = json.loads(data)
                if "process" in info:
                    process_value = info.get("process", 0)
                    yield process_value
                elif "notice" in info:
                    st.write(info["notice"])


def show_encore():
    with st.expander(context["fused_text"], expanded=True):
        info_list = context["info_list"]
        format_list = context["format_list"]
        uploader = st.file_uploader(context["fused_upload"], type=["zip"], help="\n\n".join(format_list))
        # 打开 ZIP 文件以二进制模式读取
        with open(context["fused_example_file"], "rb") as file:
            # 添加下载按钮
            st.download_button(
                label=context["fused_download"],
                data=file,
                file_name="example_fused.zip",
                mime="application/zip"
            )
        show_info = st.toggle(context["fused_info"], key='show_info', help=context["fused_info_help"])
        if show_info:
            st.info("\n\n".join(info_list))
        encore_botton = st.button(context["fused_button"], help=context["fused_button_help"])

        if st.session_state.encore:
            st.info(context["fused_success"])

        if uploader and encore_botton:
            validate_flag = encore_validate_zip(uploader)
            if not validate_flag:
                return
            # 上传该文件到服务器
            # 读取文件内容
            file_data = uploader.getvalue()
            file_name = uploader.name

            # 定义要上传的服务器 URL
            server_url = f"http://localhost:{server_port}/encore"

            # 准备文件数据（这里的 'file' 是服务器接收的文件字段名称）
            files = {"file": file_data}
            data = {
                "cache_path": get_save_dir(user=True, chat=True).replace("\\", "/") if os.name == 'nt' else get_save_dir(user=True, chat=True),
                "mode": lang,
            }

            with st.spinner(context["fused_spinner"]):
                # 发送 POST 请求到服务器
                response = requests.post(server_url, files=files, data=data, stream=True)
                progress_bar = st.progress(0)
                # 从服务器获取进度更新
                for progress in stream_progress_from_server(response):
                    # 更新进度条，假设最大值为 10
                    progress_bar.progress(progress)
                progress_bar.progress(1.0)
            # st.warning(f"response: {response.text}")

            st.session_state.encore = True
            cache_path = get_save_dir(user=True, chat=True)
            save_session_json(st.session_state, cache_path)

        