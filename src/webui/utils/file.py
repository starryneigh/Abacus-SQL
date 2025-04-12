import json
import os
import uuid
import re
import streamlit as st
import shutil
import zipfile
from io import BytesIO

def rename_curr_db_directory(new_parent_dir_name):
    # 获取 current_db_map（假设它存储的是文件的绝对路径）
    current_db_map = st.session_state.get("current_db_map", {})
    # print(f"current_db_map: {current_db_map}")

    if not current_db_map:
        st.error("当前没有文件。")
        return

    # 获取当前父目录（假设所有文件都在同一个父目录下）
    file_paths = list(current_db_map.values())
    # print(f"file_paths: {file_paths}")
    
    if file_paths:
        current_parent_dir = os.path.dirname(file_paths[0])  # 获取父目录路径
        new_parent_dir = os.path.join(os.path.dirname(current_parent_dir), new_parent_dir_name)  # 新的父目录路径

        # 重命名父目录
        try:
            # os.rename(current_parent_dir, new_parent_dir)
            # print(f"目录已成功重命名为: {new_parent_dir}")

            # 更新 current_db_map 中的路径
            updated_db_map = {}
            for file_name, file_path in current_db_map.items():
                # 替换旧的父目录路径为新的路径
                new_file_path = file_path.replace(current_parent_dir, new_parent_dir)
                updated_db_map[file_name] = new_file_path

            # 更新 session_state 中的 current_db_map
            st.session_state.current_db_map = updated_db_map


        except Exception as e:
            st.error(f"重命名目录时出错: {e}")
    else:
        st.error("当前目录为空或无法获取路径信息。")
    

def rename_dir(curr_cache_path: str, new_name: str):
    print(f"当前文件夹路径: {curr_cache_path}")
    print(f"新文件夹名称: {new_name}")
    # 获取当前文件夹的父路径和最后一个文件夹名
    parent_dir, old_folder_name = os.path.split(curr_cache_path)

    # 新的文件夹路径 (仅修改最后一个文件夹的名称)
    new_cache_path = os.path.join(parent_dir, new_name)

    # 检查文件夹是否存在，并重命名最后一个文件夹
    if os.path.exists(curr_cache_path):
        try:
            os.rename(curr_cache_path, new_cache_path)
            print(f"文件夹 '{old_folder_name}' 已重命名为 '{new_name}'")
        except OSError as e:
            print(f"重命名文件夹时发生错误: {e}")
    else:
        print(f"文件夹 '{curr_cache_path}' 不存在，无法重命名。")

def get_history_chats(path: str) -> list:
    os.makedirs(path, exist_ok=True)
    chat_names = sort_folders_by_creation_time(path)
    if len(chat_names) == 0:
        chat_names.append('New Chat_' + str(uuid.uuid4()))
    return chat_names

def sort_folders_by_creation_time(directory):
    # 获取目录下的所有文件和文件夹
    items = os.listdir(directory)
    # 获取每个文件或文件夹的完整路径，并获取其创建时间
    items_with_time = [(item, os.path.getctime(os.path.join(directory, item))) for item in items]
    # 按创建时间排序 (从旧到新)
    sorted_items = sorted(items_with_time, key=lambda x: x[1], reverse=True)
    # 只返回文件夹名称
    return [item[0] for item in sorted_items]

def encore_validate_zip(uploaded_file):
    if uploaded_file.name.endswith(".zip"):
        try:
            # 读取 zip 文件内容
            with zipfile.ZipFile(BytesIO(uploaded_file.read()), 'r') as z:
                # 列出所有文件
                file_list = z.namelist()
                print(f"Files in zip: {file_list}")
                
                # Step 1: 检查是否存在 JSON 文件
                json_file = []
                for file_name in file_list:
                    if file_name.endswith(".json") and "/" not in file_name and "\\" not in file_name:
                        json_file.append(file_name)
                        print(f"JSON 文件: {json_file}")
                
                if not json_file:
                    st.error("错误：ZIP 文件中未包含顶层 JSON 文件。")
                    return False
                
                if len(json_file) > 1:
                    st.error("错误：ZIP 文件中包含多个文件。请确保只包含一个 JSON 文件。")
                    return False

                # Step 2: 读取并验证 JSON 文件内容
                with z.open(json_file[0]) as f:
                    try:
                        json_data = json.load(f)
                        if not isinstance(json_data, list) or len(json_data) < 2:
                            st.error("错误：JSON 文件内容必须是包含至少两个条目的列表。")
                            return False
                        # 检查 JSON 中的每个条目是否符合格式要求
                        for item in json_data:
                            if not all(key in item for key in ["question", "query", "db_id"]):
                                st.error("错误：JSON 文件中每个条目必须包含 'question'、'query' 和 'db_id' 字段。")
                                return False
                    except json.JSONDecodeError:
                        st.error("错误：无法解析 JSON 文件内容。")
                        return False
                
                # Step 3: 检查 database 文件夹是否存在
                if "database/" not in file_list:
                    st.error("错误：ZIP 文件中未包含 database 文件夹。")
                    return False
                
                # Step 4: 检查每个 db_id 文件夹中是否包含对应的 .sqlite 文件
                db_ids = {item["db_id"] for item in json_data}
                for db_id in db_ids:
                    db_folder = f"database/{db_id}/"
                    db_file = f"{db_folder}{db_id}.sqlite"
                    if db_folder not in file_list or db_file not in file_list:
                        st.error(f"错误：database 文件夹中缺少 '{db_id}' 文件夹或 '{db_id}.sqlite' 文件。")
                        return False
                
                # 所有验证通过
                return True
                
        except zipfile.BadZipFile:
            st.error("错误：无法读取 ZIP 文件内容。")
            return False
    else:
        st.error("错误：上传的文件不是 ZIP 文件。")
        return False

def save_sqlite_from_zip(uploaded_files, save_directory):
    """
    从上传的 zip 文件中解压 sqlite 文件并保存到指定目录。

    Arguments:
    - uploaded_file: 用户上传的文件对象（通过 st.file_uploader 获取的 UploadedFile）
    - save_directory: 要保存 sqlite 文件的目标路径

    Returns:
    - 保存文件的路径，如果没有找到 sqlite 文件，返回 None。
    """
    # 确保保存目录存在
    if not os.path.exists(save_directory):
        os.makedirs(save_directory)

    file_map = {}
    for uploaded_file in uploaded_files:
        # 检查文件是否是 zip 文件
        if uploaded_file.name.endswith(".zip"):
            try:
                # 读取 zip 文件内容
                with zipfile.ZipFile(BytesIO(uploaded_file.read()), 'r') as z:
                    # 列出所有文件
                    file_list = z.namelist()
                    # st.write("Files in zip:", file_list)
                    # 尝试用 GBK 解码文件名，以便于读取文件名
                    try:
                        decoded_file_list = [file_name.encode('cp437').decode('gbk') for file_name in file_list]
                    except UnicodeDecodeError:
                        decoded_file_list = file_list  # 如果解码失败，使用原始的文件名

                    # 遍历文件，寻找 .sqlite 文件
                    for original_file_name, decoded_file_name in zip(file_list, decoded_file_list):
                        if decoded_file_name.endswith(".sqlite") and "/" not in decoded_file_name and "\\" not in decoded_file_name:
                            # 解压并保存 .sqlite 文件
                            z.extract(original_file_name, save_directory)
                            # 将文件重命名为解码后的名称（如果解码后有变化）
                            saved_path = os.path.join(save_directory, decoded_file_name)
                            original_path = os.path.join(save_directory, original_file_name)
                            if original_path != saved_path:
                                if os.path.exists(saved_path):
                                    os.remove(saved_path)
                                os.rename(original_path, saved_path)
                            file_map[decoded_file_name.split(".")[0]] = saved_path
                            # st.success(f"SQLite file extracted and saved to {saved_path}")
                            st.success(f"{os.path.basename(decoded_file_name)} 已上传")

            except zipfile.BadZipFile:
                st.error("上传的文件格式错误，请查看help信息并重新上传。")
                return None
        else:
            if uploaded_file.name.endswith(".sqlite"):
                # 保存 sqlite 文件
                saved_path = os.path.join(save_directory, uploaded_file.name)
                with open(saved_path, "wb") as f:
                    f.write(uploaded_file.getbuffer())
                file_map[uploaded_file.name.split(".")[0]] = saved_path
                st.success(f"{os.path.basename(uploaded_file.name)} 已上传")
    return file_map

def del_uploadfiles(selected_files):
    files_map = st.session_state.current_db_map
    for file_name in selected_files:
        file_path = files_map.get(file_name)
        
        # 删除文件
        if os.path.exists(file_path):
            os.remove(file_path)
            st.success(f"{file_name} 已删除")
            
            # 更新 current_db_map
            del st.session_state.current_db_map[file_name]

def save_uploadfiles(uploaded_files, save_dir):
    """
    保存上传的文件到指定目录，避免文件名冲突。

    参数:
    - uploaded_files: 用户通过 st.file_uploader 上传的文件对象（可以是单个文件或多个文件）。
    - save_directory: 保存文件的目录，默认是 'uploaded_files'。
    
    返回:
    - 保存文件的完整路径列表。
    """
    # 创建保存目录（如果不存在）
    os.makedirs(save_dir, exist_ok=True)

    saved_files = []  # 用于存储已保存的文件路径

    for uploaded_file in uploaded_files:
        save_path = os.path.join(save_dir, uploaded_file.name)
        # 将文件保存到指定路径
        with open(save_path, "wb") as f:
            f.write(uploaded_file.getbuffer())

        # 添加保存的文件路径到列表中
        saved_files.append(save_path)

        # 打印文件保存的成功信息
        print(f"文件 '{uploaded_file.name}' 已成功保存为 '{save_path}'")

    return saved_files

def save_session_json(session_state, filedir: str):
    """保存 session_state 到 JSON 文件"""
    os.makedirs(filedir, exist_ok=True)
    file_path = os.path.join(filedir, "session_state.json")
    # 过滤可序列化的 session_state
    serializable_data = {}
    
    for key, value in session_state.items():
        try:
            if key == "messages":
                # 不保留executions
                value = [msg for msg in value if msg.get("type") != "execution"]
            # 检查数据是否可以序列化为 JSON
            json.dumps(value)
            serializable_data[key] = value
        except (TypeError, OverflowError):
            # 如果数据不可序列化，将其转换为字符串保存
            serializable_data[key] = str(value)
    
    # 将可序列化的数据保存为 JSON 文件
    with open(file_path, "w") as f:
        json.dump(serializable_data, f)


load_list = ["messages", "current_db_map", "init_strategy", "demonstration", "demo_num", "question_num", "pre_generate_sql", "self_debug", "encore", "encore_flag"]

def load_session_json(session_state, filedir: str):
    """从 JSON 文件加载数据到 session_state"""
    file_path = os.path.join(filedir, "session_state.json")
    try:
        # 从文件加载 JSON 数据
        with open(file_path, "r") as f:
            session_data = json.load(f)

        # 更新 session_state
        for key, value in session_data.items():
            if key in load_list:
                session_state[key] = value

    except FileNotFoundError:
        save_session_json(session_state, filedir)
    except json.JSONDecodeError:
        print(f"Error decoding JSON from {file_path}.")

def remove_data(folder_path: str):
    if "cache" not in folder_path:
        print("You can only remove data from cache path.")
        return
    try:
        shutil.rmtree(folder_path)
        print(f"文件夹 {folder_path} 及其内容已成功删除")
    except OSError as e:
        print(f"删除文件夹 {folder_path} 失败: {e}")

cache_path = "cache"
def get_save_dir(user=True, chat=True):
    if user:
        if chat:
            current_chat = st.session_state["history_chats"][st.session_state["current_chat_index"]]
            save_dir = os.path.join(cache_path, st.session_state["username"], current_chat)
        else:
            save_dir = os.path.join(cache_path, st.session_state["username"])
    else:
        save_dir = cache_path
    return save_dir

def filename_correction(filename: str) -> str:
    pattern = r'[^\w\.-]'
    filename = re.sub(pattern, '', filename)
    return filename

def delete_unused_sqlite_files(cache_folder, db_map):
    """
    删除缓存文件夹中不在 db_map 中的 .sqlite 文件。
    
    参数:
    - cache_folder: 缓存文件夹路径。
    - db_map: 包含文件路径的字典。
    """
    # 获取缓存文件夹中的所有 .sqlite 文件
    cached_files = [os.path.join(cache_folder, f) for f in os.listdir(cache_folder) if f.endswith('.sqlite')]
    
    # 获取 db_map 中的 .sqlite 文件路径
    db_map_files = [os.path.join(cache_folder, f"{key}.sqlite") for key in db_map.keys()]
    
    # 删除不在 db_map 中的 .sqlite 文件
    for cached_file in cached_files:
        if cached_file not in db_map_files:
            try:
                os.remove(cached_file)
                print(f"删除了缓存中的 .sqlite 文件: {cached_file}")
            except Exception as e:
                print(f"删除文件 {cached_file} 时发生错误: {e}")

def store_missing_sqlite_files(cache_folder, db_map):
    """
    将 db_map 中不存在于缓存文件夹的文件以 key + '.sqlite' 格式保存到缓存中。
    
    参数:
    - cache_folder: 缓存文件夹路径。
    - db_map: 包含文件路径的字典。
    """
    # 获取缓存文件夹中的所有 .sqlite 文件
    cached_files = [os.path.join(cache_folder, f) for f in os.listdir(cache_folder) if f.endswith('.sqlite')]
    
    # 将 db_map 中未存储的文件复制到缓存文件夹，以 key + '.sqlite' 命名
    for key, db_path in db_map.items():
        target_file = os.path.join(cache_folder, f"{key}.sqlite")
        
        if target_file not in cached_files:
            try:
                # 复制文件并重命名为 key + '.sqlite'
                shutil.copy(db_path, target_file)
                print(f"将 {db_path} 复制到缓存，并命名为: {target_file}")
            except Exception as e:
                print(f"保存文件 {db_path} 时发生错误: {e}")

def ds_sqlite_files(cache_folder, db_map):
    delete_unused_sqlite_files(cache_folder, db_map)
    store_missing_sqlite_files(cache_folder, db_map)
    # 构建一个新的 db_map，反映当前缓存中的 .sqlite 文件
    current_db_map = {}
    for f in os.listdir(cache_folder):
        if f.endswith('.sqlite'):
            key = os.path.splitext(f)[0]  # 获取不带扩展名的文件名作为 key
            current_db_map[key] = os.path.join(cache_folder, f)
            # print(f"当前数据库映射: {key} -> {current_db_map[key]}")
    # print(f"current_db_map: {current_db_map}")
    
    return current_db_map
