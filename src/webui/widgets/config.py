import streamlit as st
from ...text.front import text
from ..icon.custom import avatar_path

def split_line():
    st.markdown("""
        <div style="
            border: 1px solid #ccc;  /* 边框颜色 */
            padding: 15px;            /* 内边距 */
            border-radius: 10px;      /* 圆角边框 */
        ">
    """, unsafe_allow_html=True)

def change_lang():
    lang = st.query_params.get("lang", "zh")
    content = text[lang]["app"]
    langs = ["中文", "English"]
    if lang == "zh":
        default_lang_index = langs.index("中文")
    else:
        default_lang_index = langs.index("English")
    lang = st.selectbox(content["lang_choose"], langs, index=default_lang_index, key="main_select_lang", on_change=set_lang)
    return "zh" if lang == "中文" else "en"

def set_lang():
    # print(f"session_state: {st.session_state.select_lang}")
    if "main_select_lang" in st.session_state:
        if st.session_state.main_select_lang == "中文":
            select_lang = "zh"
        else:
            select_lang = "en"
        # print(f"Language changed: {select_lang}")
        st.session_state.lang = select_lang
        st.query_params["lang"] = select_lang
        # st.rerun()

def show_config(login_obj):
    lang = st.query_params.get("lang", "zh")
    context = text[lang]["config"]

    st.write(context["config_title"])
    split_line()
    user_container = st.container()
    with user_container:
        img, name, logout = st.columns([2, 2, 2], vertical_alignment="center")
        img.image(avatar_path, width=60)
        slogan = context["user_slogan"]
        name.write(f"### **{st.session_state.username}**  \n{slogan}")
        with logout:
            login_obj.logout_widget()
    split_line()

    change_lang()
    # context = text[lang]["config"]

    with st.expander(context["config_resetpwd"]):
        login_obj.reset_password("config Forgot Password Form", border=False)