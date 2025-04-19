import streamlit as st
import json
import os
import uuid
from streamlit_lottie import st_lottie
from streamlit_option_menu import option_menu
from streamlit_cookies_manager import EncryptedCookieManager
from .utils import (
    check_usr_pass,
    load_lottieurl,
    check_valid_name,
    check_valid_email,
    check_unique_email,
    check_unique_usr,
    register_new_usr,
    check_email_exists,
    generate_random_passwd,
    send_passwd_in_email,
    change_passwd,
    check_current_passwd,
    check_huozi_pass,
)
from ...text.front import text


class __login__:
    """
    Builds the UI for the Login/ Sign Up page.
    """

    def __init__(
        self,
        auth_token: str,
        company_name: str = "Text2sql Demo",
        width=200,
        height=250,
        logout_button_name: str = "登出",
        hide_menu_bool: bool = False,
        hide_footer_bool: bool = False,
        lottie_url: str = "https://assets8.lottiefiles.com/packages/lf20_ktwnwv5m.json",
        prefix: str = "streamlit_login_ui_yummy_cookies",
        password: str = "9d68d6f2-4258-45c9-96eb-2d6bc74ddbb5-d8f49cab-edbb-404a-94d0-b25b1d4a564b",
    ):
        """
        Arguments:
        -----------
        1. self
        2. auth_token : The unique authorization token received from - https://www.courier.com/email-api/
        3. company_name : This is the name of the person/ organization which will send the password reset email.
        4. width : Width of the animation on the login page.
        5. height : Height of the animation on the login page.
        6. logout_button_name : The logout button name.
        7. hide_menu_bool : Pass True if the streamlit menu should be hidden.
        8. hide_footer_bool : Pass True if the 'made with streamlit' footer should be hidden.
        9. lottie_url : The lottie animation you would like to use on the login page. Explore animations at - https://lottiefiles.com/featured
        """
        self.auth_token = auth_token
        self.company_name = company_name
        self.width = width
        self.height = height
        self.logout_button_name = logout_button_name
        self.hide_menu_bool = hide_menu_bool
        self.hide_footer_bool = hide_footer_bool
        self.lottie_url = lottie_url
        self.lang = st.query_params["lang"] if "lang" in st.query_params else "en"
        self.context = text[self.lang]["login"]
        # print(f"当前语言：{self.lang}")

        self.place = st.empty()
        with self.place:
            self.cookies = EncryptedCookieManager(
                prefix=prefix,
                password=password,
            )
            if not self.cookies.ready():
                st.stop()

    def check_auth_json_file_exists(self, auth_filename: str) -> bool:
        """
        Checks if the auth file (where the user info is stored) already exists.
        """
        file_names = []
        for path in os.listdir("./"):
            if os.path.isfile(os.path.join("./", path)):
                file_names.append(path)

        present_files = []
        for file_name in file_names:
            if auth_filename in file_name:
                present_files.append(file_name)

            present_files = sorted(present_files)
            if len(present_files) > 0:
                return True
        return False

    def get_username(self):
        if st.session_state["LOGOUT_BUTTON_HIT"] == False:
            fetched_cookies = self.cookies
            if "__login_signup_ui_username__" in fetched_cookies.keys():
                username = fetched_cookies["__login_signup_ui_username__"]
                return username

    def login_widget(self, login_type="") -> None:
        """
        Creates the login widget, checks and sets cookies, authenticates the users.
        Supports both default and huozi login.
        """
        # 检查用户是否已经登录
        if not st.session_state.get("LOGGED_IN", False):
            fetched_cookies = self.cookies
            # 如果 cookies 中存在用户名且合法，则自动登录
            username_cookie = fetched_cookies.get("__login_signup_ui_username__")
            if (
                username_cookie
                and username_cookie != "1c9a923f-fb21-4a91-b3f3-5f18e3f01182"
            ):
                st.session_state["LOGGED_IN"] = True
                st.session_state["username"] = self.get_username()

        # 获取不同的登录 URL 和验证函数
        if login_type == "huozi":
            check_credentials = check_huozi_pass
        else:
            check_credentials = check_usr_pass

        # 如果用户未登录，显示登录表单
        if not st.session_state.get("LOGGED_IN", False):
            st.session_state["LOGOUT_BUTTON_HIT"] = False

            del_login = st.empty()
            with del_login.form(f"{login_type.capitalize()} Login Form"):
                username = st.text_input(
                    self.context["username_text"],
                    placeholder=self.context["username_placeholder"],
                    help=(
                        self.context["username_help_huozi"]
                        if login_type == "huozi"
                        else self.context["username_help"]
                    ),
                )
                password = st.text_input(
                    self.context["password_text"],
                    placeholder=self.context["password_placeholder"],
                    type="password",
                    help=(
                        self.context["password_help_huozi"]
                        if login_type == "huozi"
                        else self.context["password_help"]
                    ),
                )
                login_submit_button = st.form_submit_button(
                    label=self.context["login_button_name"],
                    help=self.context["login_button_help"],
                )

            if login_submit_button:
                if check_credentials(username, password):
                    if login_type == "huozi":
                        username = username + " #huozi"
                        st.success(self.context["login_success_huozi"])
                    st.success(self.context["login_success"])
                    st.session_state["LOGGED_IN"] = True
                    st.session_state["username"] = username
                    self.cookies["__login_signup_ui_username__"] = username
                    self.cookies.save()
                    # print(self.cookies)
                    del_login.empty()  # 清除登录表单
                    st.rerun()
                else:
                    st.error(self.context["login_fail"])

    def animation(self) -> None:
        """
        Renders the lottie animation.
        """
        try:
            lottie_json = load_lottieurl(self.lottie_url)
            st_lottie(lottie_json, width=self.width, height=self.height)
        except Exception as e:
            return

    def sign_up_widget(self) -> None:
        """
        Creates the sign-up widget and stores the user info in a secure way in the _secret_auth_.json file.
        """
        with st.form("Sign Up Form"):
            # 获取用户输入
            name_sign_up = st.text_input(
                self.context["name_signup_text"],
                placeholder=self.context["name_signup_placeholder"],
                help=self.context["name_signup_help"],
            )
            email_sign_up = st.text_input(
                self.context["email_signup_text"],
                placeholder=self.context["email_signup_placeholder"],
                help=self.context["email_signup_help"],
            )
            username_sign_up = st.text_input(
                self.context["username_signup_text"],
                placeholder=self.context["username_signup_placeholder"],
                help=self.context["username_signup_help"],
            )
            password_sign_up = st.text_input(
                self.context["password_signup_text"],
                placeholder=self.context["password_signup_placeholder"],
                type="password",
                help=self.context["password_signup_help"],
            )
            sign_up_submit_button = st.form_submit_button(
                label=self.context["signup_button_name"],
                help=self.context["signup_button_help"],
            )

            if sign_up_submit_button:
                # 校验输入
                valid_name_check = check_valid_name(name_sign_up)
                valid_email_check = check_valid_email(email_sign_up)
                unique_email_check = check_unique_email(email_sign_up)
                unique_username_check = check_unique_usr(username_sign_up)

                # 错误提示逻辑合并
                if not valid_name_check:
                    st.error(self.context["signup_name_error"])
                elif not valid_email_check:
                    st.error(self.context["signup_email_error"])
                elif not unique_email_check:
                    st.error(
                        self.context["signup_email_error_unique"].format(
                            email_sign_up=email_sign_up
                        )
                    )
                elif not unique_username_check:
                    st.error(
                        self.context["signup_username_error"].format(
                            username_sign_up=username_sign_up
                        )
                    )
                elif not username_sign_up:
                    st.error(self.context["signup_username_none"])
                else:
                    # 所有检查通过后，注册新用户
                    register_new_usr(
                        name_sign_up, email_sign_up, username_sign_up, password_sign_up
                    )
                    st.success(self.context["signup_success_text"])

    def forgot_password(self) -> None:
        """
        Creates the forgot password widget and after user authentication (email),
        triggers an email to the user containing a random password.
        """
        with st.form("Forgot Password Form"):
            email_forgot_passwd = st.text_input(
                self.context["forgotpwd_email_text"],
                placeholder=self.context["forgotpwd_email_placeholder"],
                help=self.context["forgotpwd_email_help"],
            )
            forgot_passwd_submit_button = st.form_submit_button(
                label=self.context["forgotpwd_button_text"],
                help=self.context["forgotpwd_button_help"],
            )

        if forgot_passwd_submit_button:
            # 检查邮箱是否注册
            email_exists_check, username_forgot_passwd = check_email_exists(
                email_forgot_passwd
            )

            if not email_exists_check:
                st.error(self.context["forgotpwd_email_error_exist"])
            else:
                # 生成随机密码并发送邮件
                random_password = generate_random_passwd()
                send_passwd_in_email(
                    self.auth_token,
                    username_forgot_passwd,
                    email_forgot_passwd,
                    self.company_name,
                    random_password,
                )
                # 更新用户的密码
                change_passwd(email_forgot_passwd, random_password)
                st.success(self.context["forgotpwd_success_text"])

    def reset_password(self, form_name, border=True) -> None:
        """
        Creates the reset password widget and after user authentication (email and temporary password),
        resets the password and updates the same in the _secret_auth_.json file.
        """
        form_name += "123"
        with st.form(form_name, border=border):
            email_reset_passwd = st.text_input(
                self.context["resetpwd_email_text"],
                placeholder=self.context["resetpwd_email_placeholder"],
                help=self.context["resetpwd_email_help"],
            )
            current_passwd = st.text_input(
                self.context["resetpwd_currentpwd_text"],
                placeholder=self.context["resetpwd_currentpwd_placeholder"],
                help=self.context["resetpwd_currentpwd_help"],
            )
            new_passwd = st.text_input(
                self.context["resetpwd_newpwd_text"],
                placeholder=self.context["resetpwd_newpwd_placeholder"],
                type="password",
                help=self.context["resetpwd_newpwd_help"],
            )
            new_passwd_1 = st.text_input(
                self.context["resetpwd_newpwd1_text"],
                placeholder=self.context["resetpwd_newpwd1_placeholder"],
                type="password",
                help=self.context["resetpwd_newpwd1_help"],
            )
            reset_passwd_submit_button = st.form_submit_button(
                label=self.context["resetpwd_button_text"],
                help=self.context["resetpwd_button_help"],
            )

        if reset_passwd_submit_button:
            # 验证邮箱是否存在
            email_exists_check, username_reset_passwd = check_email_exists(
                email_reset_passwd
            )

            if not email_exists_check:
                st.error(self.context["resetpwd_email_error_exist"])
                return

            # 检查临时密码是否正确
            current_passwd_check = check_current_passwd(
                email_reset_passwd, current_passwd
            )
            if not current_passwd_check:
                st.error(self.context["resetpwd_currentpwd_error"])
                return

            # 检查两次输入的新密码是否匹配
            if new_passwd != new_passwd_1:
                st.error(self.context["resetpwd_newpwd_error"])
                return

            # 更新密码
            change_passwd(email_reset_passwd, new_passwd)
            st.success(self.context["resetpwd_success_text"])

    def logout_widget(self) -> None:
        """
        Creates the logout widget in the sidebar only if the user is logged in.
        """
        if st.session_state["LOGGED_IN"] == True:
            del_logout = st.empty()
            del_logout.markdown("#")
            logout_click_check = del_logout.button(
                self.context["logout_button_text"],
                key="logout",
                help=self.context["logout_button_help"],
            )

            if logout_click_check == True:
                st.session_state["LOGOUT_BUTTON_HIT"] = True
                st.session_state["LOGGED_IN"] = False
                st.session_state.initial_settings = False
                self.cookies["__login_signup_ui_username__"] = (
                    "1c9a923f-fb21-4a91-b3f3-5f18e3f01182"
                )
                self.cookies.save()
                # print(self.cookies)
                del_logout.empty()
                st.rerun()

    def hide_menu(self) -> None:
        """
        Hides the streamlit menu situated in the top right.
        """
        st.markdown(
            """ <style>
        #MainMenu {visibility: hidden;}
        </style> """,
            unsafe_allow_html=True,
        )

    def hide_footer(self) -> None:
        """
        Hides the 'made with streamlit' footer.
        """
        st.markdown(
            """ <style>
        footer {visibility: hidden;}
        </style> """,
            unsafe_allow_html=True,
        )

    def tabs_nav(self, main_page):
        """
        Creates the tabs navigation bar.
        """
        with main_page:
            login, sign_up, login_huozi, forgot_passwd, reset_passwd = st.tabs(
                [
                    self.context["tab_login"],
                    self.context["tab_signup"],
                    self.context["tab_login_huozi"],
                    self.context["tab_forgotpwd"],
                    self.context["tab_resetpwd"],
                ]
            )
        return login, sign_up, login_huozi, forgot_passwd, reset_passwd

    def build_login_ui(self):
        """
        Brings everything together, calls important functions.
        """
        if "LOGGED_IN" not in st.session_state:
            st.session_state["LOGGED_IN"] = False

        if "LOGOUT_BUTTON_HIT" not in st.session_state:
            st.session_state["LOGOUT_BUTTON_HIT"] = False

        auth_json_exists_bool = self.check_auth_json_file_exists("_secret_auth_.json")

        if auth_json_exists_bool == False:
            with open("_secret_auth_.json", "w") as auth_json:
                json.dump([], auth_json)

        title = st.empty()
        main_page = st.empty()

        title.write(self.context["login_title"])

        login, sign_up, login_huozi, forgot_passwd, reset_passwd = self.tabs_nav(
            main_page
        )

        with login:
            self.display_login_form()
        with sign_up:
            self.sign_up_widget()
        with login_huozi:
            self.display_login_form(login_type="huozi")
        with forgot_passwd:
            self.forgot_password()
        with reset_passwd:
            self.reset_password("Forgot Password Form")

        if st.session_state["LOGGED_IN"] == True:
            title.empty()
            main_page.empty()
            self.place.empty()

        if self.hide_menu_bool == True:
            self.hide_menu()

        if self.hide_footer_bool == True:
            self.hide_footer()

        return st.session_state["LOGGED_IN"]

    def display_login_form(self, login_type="default"):
        """
        显示登录表单并处理动画。
        login_type: 默认为 "default"，也可以为 "huozi"。
        """
        c1, c2 = st.columns([7, 3])
        with c1:
            self.login_widget(login_type=login_type)
        with c2:
            if not st.session_state["LOGGED_IN"]:
                self.animation()
