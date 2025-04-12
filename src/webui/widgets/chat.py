import streamlit as st
import os
import json
import uuid
from ..icon.custom import ICON_base64
from ..utils.file import save_session_json, filename_correction, get_save_dir, remove_data, load_session_json, rename_dir, rename_curr_db_directory
from ...text.front import text

lang_chat = st.query_params["lang"] if "lang" in st.query_params else "zh"
print(f"lang_chat: {lang_chat}")
context = text[lang_chat]["chat"]

def chat_icon():
    icon_text = f"""
        <div class="icon-text-container">
            <img src='data:image/png;base64,{ICON_base64}' alt='icon' style='width: 60px;'>
            <span style='font-size: 24px;'>{context["icon_text"]}</span>
        </div>
        """
    st.markdown(
        icon_text,
        unsafe_allow_html=True,
    )


def show_chats(container):
    global lang_chat
    global context
    lang_chat = st.query_params["lang"] if "lang" in st.query_params else "zh"
    context = text[lang_chat]["chat"]

    with container:
        chat_icon()

        current_chat = st.radio(
            label=context["current_chat_label"],
            format_func=lambda x: "**"+x.split("_")[0]+"**" if "_" in x else x,
            options=st.session_state["history_chats"],
            label_visibility="collapsed",
            index=st.session_state["current_chat_index"],
            key="current_chat"
            + st.session_state["history_chats"][
                st.session_state["current_chat_index"]
            ],
        )
        st.write("---")

        st.session_state.current_chat_index = st.session_state["history_chats"].index(
            current_chat)
        if st.session_state.previous_chat != current_chat:
            st.session_state.previous_chat = current_chat
            user_cache_path = get_save_dir(user=True, chat=False)
            curr_cache_path = os.path.join(user_cache_path, current_chat)
            st.session_state.pop("messages", None)
            st.session_state.current_db_map = {}
            st.session_state.init_strategy = False
            st.session_state.feedback = {}
            load_session_json(st.session_state, curr_cache_path)
            st.rerun()

    with container:
        c1, c2 = st.columns(2)
        create_chat_button = c1.button(
            context["create_button_text"], use_container_width=True, key="create_chat_button"
        )
        if create_chat_button:
            create_chat_fun()
            st.rerun()

        delete_chat_button = c2.button(
            context["delete_button_text"], use_container_width=True, key="delete_chat_button"
        )
        if delete_chat_button:
            delete_chat_fun(current_chat)
            st.rerun()

        if ("set_chat_name" in st.session_state) and st.session_state[
            "set_chat_name"
        ] != "":
            reset_chat_name_fun(
                st.session_state["set_chat_name"], current_chat)
            st.session_state["set_chat_name"] = ""
            st.rerun()

        st.write("\n")
        chat_name = st.text_input(context["set_name_text"], key="set_chat_name", placeholder=context["set_name_placeholder"])
        # if chat_name:
        #     if "_" in chat_name:
        #         st.error("请不要使用下划线")
        #         st.session_state["set_chat_name"] = ""
        st.caption(context["set_name_caption"])

        model_display_options = context["model_display_options"]
        model_actual_values = context["model_actual_values"]
        model_api_bases = context["model_default_url"]
        # 创建下拉框
        option = st.selectbox(context["model_text"], model_display_options, help=context["model_help"])
        model_index = model_display_options.index(option)
        actual_value = model_actual_values[model_index]
        st.session_state["model_name"] = actual_value

        if model_index != 0:
            st.text_input(
                context["model_api_key_text"], 
                type="password", 
                key="api_key", 
                placeholder=context["model_api_key_placeholder"]
            )
            st.text_input(
                context["model_api_base_text"],
                key="api_base",
                placeholder=context["model_api_base_placeholder"],
                value=model_api_bases[model_index],
            )


def reset_chat_name_fun(chat_name, current_chat):
    # print("current_chat", current_chat)
    chat_name = chat_name + "_" + str(uuid.uuid4())
    new_name = filename_correction(chat_name)
    rename_curr_db_directory(new_name)

    current_chat_index = st.session_state.current_chat_index
    # print(f"current_chat_index: {current_chat_index}")
    st.session_state["history_chats"][current_chat_index] = new_name
    user_cache_path = get_save_dir(user=True, chat=False)
    curr_cache_path = os.path.join(user_cache_path, current_chat)
    rename_dir(curr_cache_path, new_name)
    cache_path = get_save_dir(user=True, chat=True)
    save_session_json(st.session_state, cache_path)


def create_chat_fun():
    st.session_state["history_chats"] = [
        "New Chat_" + str(uuid.uuid4())
    ] + st.session_state["history_chats"]
    st.session_state["current_chat_index"] = 0


def delete_chat_fun(current_chat):
    # 如果只剩下一个窗口，再新增一个新窗口
    if len(st.session_state["history_chats"]) == 1:
        chat_init = "New Chat_" + str(uuid.uuid4())
        st.session_state["history_chats"].append(chat_init)

    pre_chat_index = st.session_state["history_chats"].index(current_chat)
    if pre_chat_index > 0:
        st.session_state["current_chat_index"] = (
            st.session_state["history_chats"].index(current_chat) - 1
        )
    else:
        st.session_state["current_chat_index"] = 0

    st.session_state["history_chats"].pop(pre_chat_index)
    user_cache_path = get_save_dir(user=True, chat=False)
    curr_cache_path = os.path.join(user_cache_path, current_chat)
    remove_data(curr_cache_path)
