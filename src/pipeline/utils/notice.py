import json
from .my_logger import MyLogger

logger = MyLogger("sserver", "logs/text2sql.log")

def send_notice(message, type="notice"):
    """
    发送进度消息，通知用户当前操作状态。
    """
    dict_data = {type: message}
    logger.info(f"Sending {type}: {message}")
    yield f"data: {json.dumps(dict_data)}\n\n"