from .my_logger import MyLogger
import requests
import json
from tqdm import tqdm
from .extract_from_sql import extract_sql_and_db
from .stream_generator import GPTOpenAIChatGenerator

logger = MyLogger("stream_handler", "logs/text2sql.log")

class StreamDataHandler:
    def __init__(self, url: str):
        """
        初始化 StreamDataHandler 实例
        :param url: 服务器的 URL，用于发送和接收数据
        """
        self.url = url
        self.prediction = ""  # 用于存储生成的预测结果

    def generate_with_llm(self, source: list[str], config: dict=None, api_data:dict = None) -> list[list[tuple[str, float]]]:
        """
        从服务器获取预测结果。
        :param source: 生成预测结果的源数据
        :param config: 配置信息
        :return: 生成的预测结果
        """
        logger.info(f"Generating data with LLM: {source}")
        if api_data:
            model_name = api_data.get("model_name", "Qwen")
        else:
            model_name = "Qwen"
        print(f"model_name: {model_name}")
        predictions = []
        # for prompt in tqdm(source):
        for prompt in source:
            prediction_list = []
            data = {"prompt": [prompt]}
            try:
                if "Qwen" not in model_name:
                    packed_data = (api_data["messages"], model_name, {"temperature": 0})
                    openai_generator = GPTOpenAIChatGenerator(model_name)
                    response = openai_generator.generate_single(packed_data, api_key=api_data.get("api_key"), api_base=api_data.get("api_base"))
                else:
                    response = requests.post(self.url, json=data)
                    response.raise_for_status()  # 捕获请求失败的状态码

                if hasattr(response, 'iter_lines'):  # requests 响应对象
                    # 处理 requests 响应
                    iterator = response.iter_lines(chunk_size=8192, decode_unicode=False, delimiter=b"\0")
                 # 判断 response 类型并处理
                elif hasattr(response, '__iter__'):  # 生成器对象
                    # 处理生成器
                    iterator = response
                else:
                    raise ValueError("Unknown response type")

                prediction = ""
                for chunk in iterator:
                    if not chunk:
                        continue

                    try:
                        data_chunk = json.loads(chunk.decode("utf-8")) if isinstance(chunk, bytes) else chunk
                        if "error" in data_chunk:
                            raise ConnectionError(f"Error from server: {data_chunk['error']}")
                    except json.JSONDecodeError:
                        logger.error("Failed to decode JSON chunk")
                        continue
                    
                    logger.debug(f"Received data from server: {data_chunk}")
                    data_gen = data_chunk.get("generator")
                    if data_gen:
                        prediction += data_gen

            except requests.RequestException as e:
                logger.error(f"Error connecting to server: {e}")
                raise ConnectionError(f"Error connecting to server: {e}")

            logger.info("Connection to server established, generating data...")
            prediction_list.append((prediction, 1.0))
            predictions.append(prediction_list)
        return predictions


    def stream_data(self, data: dict, db_infos: list[dict], cho_db=None, files=None):
        """
        从服务器获取流式数据，并以生成器方式传输数据。
        :param data: 发送给服务器的 JSON 数据
        :param db_infos: 数据库信息的列表，用于修正 SQL 预测
        :param files: 可选，上传的文件信息
        :yield: 服务器返回的流数据，格式为 JSON
        """
        logger.info(f"Sending data to server: {data}")
        model_name = data.get("model_name", "Qwen")
        print(f"model_name: {model_name}")
        try:
            if "Qwen" not in model_name:
                packed_data = (data["messages"], model_name, {"temperature": 0})
                openai_generator = GPTOpenAIChatGenerator(model_name)
                response = openai_generator.generate_single(packed_data, api_key=data.get("api_key"), api_base=data.get("api_base"))
            else:
                response = requests.post(self.url, json=data, files=files, stream=True)
                response.raise_for_status()  # 捕获请求失败的状态码
        except requests.RequestException as e:
            logger.error(f"Error connecting to server: {e}")
            raise ConnectionError(f"Error connecting to server: {e}")

        logger.info("Connection to server established, streaming data...")
        if hasattr(response, 'iter_lines'):  # requests 响应对象
            # 处理 requests 响应
            iterator = response.iter_lines(chunk_size=8192, decode_unicode=False, delimiter=b"\0")
            # 判断 response 类型并处理
        elif hasattr(response, '__iter__'):  # 生成器对象
            # 处理生成器
            iterator = response
        else:
            raise ValueError("Unknown response type")
        self.prediction = ""  # 重置 prediction

        for chunk in iterator:
            if not chunk:
                continue

            try:
                data_chunk = json.loads(chunk.decode("utf-8")) if isinstance(chunk, bytes) else chunk
                if "error" in data_chunk:
                    raise ConnectionError(f"Error from server: {data_chunk['error']}")
            except json.JSONDecodeError:
                logger.error("Failed to decode JSON chunk")
                continue
            
            logger.debug(f"Received data from server: {data_chunk}")
            self._handle_chunk(data_chunk)
            yield f"data: {json.dumps(data_chunk)}\n\n"

        yield from self._generate_final_output(db_infos, cho_db)

    def _handle_chunk(self, data_chunk):
        """
        处理每个收到的数据块，更新预测结果。
        """
        data_gen = data_chunk.get("generator")
        if data_gen:
            self.prediction += data_gen

    def _generate_final_output(self, db_infos, cho_db):
        """
        生成并发送修正后的 SQL 预测结果。
        """
        prediction = self.get_prediction()
        sql, db_id = self.fix_pred(db_infos)
        logger.info(f"Final SQL: {sql}, DB ID: {db_id}")
        
        send_data = {
            "prediction": prediction,
            "query": sql,
            "database": db_id if cho_db is None else cho_db,
        }
        # print(f"send_data: {send_data}")
        yield f"data: {json.dumps(send_data)}\n\n"

    def fix_pred(self, db_infos: list[dict]):
        """
        根据预测结果和数据库信息修正 SQL 语句和数据库 ID。
        """
        sql, db_id = extract_sql_and_db(self.prediction, db_infos)
        return sql, db_id

    def get_prediction(self):
        """
        返回当前生成的预测结果。
        """
        return self.prediction
