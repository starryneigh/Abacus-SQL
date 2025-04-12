import asyncio
import time
import torch
from tqdm import tqdm
import json
import os
import sys
import openai
from openai import OpenAI
from flask import Flask, Response, request
from vllm import AsyncLLMEngine, AsyncEngineArgs, SamplingParams
from typing import List, Dict, Any, Tuple



def handle_openai_exception(e):
    """
    处理 OpenAI 异常的函数，并抛出包含错误类型字符串的异常。
    
    :param e: 捕获的异常对象
    :return: 无
    """
    if isinstance(e, openai.RateLimitError):
        # 对于速率限制错误，重试请求
        time.sleep(10)  # 等待10秒后再重试
        raise Exception('RateLimitError')
    elif isinstance(e, openai.AuthenticationError):
        # 对于身份验证错误
        raise Exception('AuthenticationError')
    elif isinstance(e, openai.APIConnectionError):
        # API连接错误
        raise Exception('APIConnectionError')
    elif isinstance(e, openai.Timeout):
        # 请求超时
        raise Exception('Timeout')
    elif isinstance(e, openai.InternalServerError):
        # 内部服务器错误
        raise Exception('InternalServerError')
    else:
        # 其他未知错误
        print(type(e))
        raise Exception('UnknownError')
    
def merge_messages(messages):
    # 新的消息列表
    merged_messages = []
    
    # 当前拼接的消息内容
    current_message = None

    for i, message in enumerate(messages):
        # 如果当前消息为空，初始化为第一个消息
        if current_message is None:
            current_message = message
        else:
            # 如果连续消息是同一角色（user 或 assistant），就拼接内容
            if current_message['role'] == message['role']:
                current_message['content'] += "\n" + message['content']
            else:
                # 如果角色不同，将当前拼接的消息加入新列表，并开始新的拼接
                merged_messages.append(current_message)
                current_message = message
    
    # 添加最后一个消息
    if current_message is not None:
        merged_messages.append(current_message)
    
    return merged_messages


class GPTAzureChatGenerator(object):
    def __init__(self, model_name: str):
        self.model_name = model_name
        self.error_types = {
            "continue_error": [
                "timed out",
                "Connection error",
                "Connection reset by peer",
                "Remote end closed connection without response",
                "occurred in violation of protocol",
                "Failed to resolve",
                "TLSV1_ALERT_INTERNAL_ERROR",
                "Error communicating",
                "The server is overloaded or not ready yet",
                "upstream_error",
                "new_api_error",
                "当前分组上游负载已饱和",
                "Lock wait timeout exceeded"
            ],
            "sleep_error": [
                "call rate limit",
                "token rate limit"
            ],
            "ignore_error": [
                "content",
                "reduce the length"
            ]
        }

    def generate_single(self, packed_data: List[tuple]) -> List[Tuple[str, float]]:
        from openai import AzureOpenAI
        from openai.types.chat import ChatCompletion

        sentence, engine, config = packed_data
        client = AzureOpenAI(
            api_version="2023-07-01-preview",
            azure_endpoint="https://yfllm01.openai.azure.com/",
            api_key=API_KEY,
        )

        while True:
            try:
                completion: ChatCompletion = client.chat.completions.create(
                    model=engine,
                    messages=[{"role": "user", "content": sentence}],
                    **config)
                return [(x.message.content, 1.0) for x in completion.choices]
            except Exception as e:
                continue_flag = False
                sleep_flag = False
                ignore_flag = False
                for x in self.error_types['continue_error']:
                    if x in str(e):
                        continue_flag = True
                for x in self.error_types['sleep_error']:
                    if x in str(e):
                        sleep_flag = True
                        continue_flag = True
                for x in self.error_types['ignore_error']:
                    if x in str(e):
                        ignore_flag = True
                if sleep_flag:
                    time.sleep(5)
                if continue_flag:
                    continue
                if not ignore_flag:
                    print(e)
                return [""]

    def generate(self, source: List[str], config: Dict[str, Any]) -> List[List[Tuple[str, float]]]:
        config = deepcopy(config)
        if config['parallel']:
            config.pop('parallel')
            if 'batch_size' in config:
                config.pop('batch_size')
            packed_data = [(x, self.model_name, config) for x in source]
            with ProcessPoolExecutor(max_workers=os.cpu_count() // 2) as _:
                result: List[List[str]] = list(process_map(
                    self.generate_single, packed_data, max_workers=os.cpu_count() // 2, chunksize=1))
        else:
            config.pop('parallel')
            result: List[List[str]] = [self.generate_single(
                (x, self.model_name, config)) for x in tqdm(source)]
        return result


class GPTOpenAIChatGenerator(GPTAzureChatGenerator):
    def generate_single(self, packed_data: List[tuple], api_key, api_base):
        client = OpenAI(api_key=api_key, base_url=api_base)

        messages, engine, config = packed_data
        print(f"messages: {messages}")
        messages = merge_messages(messages)
        try:
            # 记录请求开始时间
            start_time = time.time()
            completion = client.chat.completions.create(
                model=self.model_name,
                messages=messages,
                stream=True,
                **config
            )
            # 逐个处理流式返回的分块
            for chunk in completion:
                # 计算每个分块的延迟
                chunk_time = time.time() - start_time

                # 提取每个分块的消息内容
                # print(f"chunk: {chunk}")
                if hasattr(chunk, 'choices') and len(chunk.choices) > 0:
                    chunk_message = chunk.choices[0].delta.content
                    if not chunk_message:
                        continue
                    # print(chunk_message)

                # 将文本内容逐步返回给调用者
                yield {"generator": chunk_message}

                # 打印消息和延迟
                print(f"Message received {chunk_time:.2f} seconds after request: {chunk_message}")
        except Exception as e:
            handle_openai_exception(e)
            # except Exception as e:
            #     if isinstance(e, openai.InternalServerError):
            #         print("Skipping InternalServerError in generic Exception handler")
            #         raise e
            #     continue_flag = False
            #     sleep_flag = False
            #     ignore_flag = False
            #     err_msg = []
            #     for x in self.error_types['continue_error']:
            #         if x in str(e):
            #             continue_flag = True
            #             err_msg.append(x)
            #     for x in self.error_types['sleep_error']:
            #         if x in str(e):
            #             sleep_flag = True
            #             continue_flag = True
            #             err_msg.append(x)
            #     for x in self.error_types['ignore_error']:
            #         if x in str(e):
            #             ignore_flag = True
            #             err_msg.append(x)
            #     if sleep_flag:
            #         time.sleep(5)
            #     if continue_flag:
            #         continue
            #     if not ignore_flag:
            #         print(e)
            #     err_str = "\n".join(err_msg)
            #     raise Exception(err_str)


class LlamaGenerator(object):
    def __init__(self, model_name_or_path: str):
        def check_cuda_gt_8() -> bool:
            device_count = torch.cuda.device_count()
            for i in range(device_count):
                device_properties = torch.cuda.get_device_properties(i)
                compute_capability = float(
                    f"{device_properties.major}.{device_properties.minor}"
                )
                if compute_capability < 8.0:
                    return False
            return True

        self.model_name_or_path = model_name_or_path
        self.engine_args = AsyncEngineArgs(
            model=model_name_or_path,
            dtype="auto" if check_cuda_gt_8() else "float",
            enforce_eager=True,
            max_model_len=4000
        )
        self.llm_engine = AsyncLLMEngine.from_engine_args(self.engine_args)
        self.tokenizer = None

    async def init_tokenizer(self):
        self.tokenizer = await self.llm_engine.get_tokenizer()

    async def generate(self, source: List[str], config: Dict[str, Any], request_id: int = None):

        assert len(source) == 1, "Streaming generation only supports a single prompt"
        # print(f"source: {source[0]}")

        # 确保tokenizer已初始化
        if self.tokenizer is None:
            await self.init_tokenizer()

        # source_filtered = []
        # for i, x in tqdm(enumerate(source), total=len(source), desc="filtering too long input"):
        #     if len(self.tokenizer(x)['input_ids']) > self.llm_engine.llm_engine.model_config.max_model_len:
        #         source[i] = "TL;NR"
        #         too_long_data_count += 1
        #     else:
        #         source_filtered.append(x)
        # print(f"too long input count: {too_long_data_count}")
        # if config["ignore_too_long"]:
        #     source = source_filtered

        sampling_params = SamplingParams(
            temperature=config["temperature"],
            top_p=config["top_p"],
            max_tokens=config["max_tokens"],
            n=config["n"],
            logprobs=1,
            stop=config["stop"],
        )

        prompt = source[0]
        # print(f"prompt: {prompt}")
        sys.stdout.flush()

        # # 在调用generate之前，确保之前的任务已经完成或被取消
        # if hasattr(self, "results_generator") and not self.results_generator.aclose():
        #     print("Closing previous results generator")
        #     sys.stdout.flush()
        #     await self.results_generator.aclose()  # 关闭之前的生成器

        request_id = time.monotonic()
        results_generator = self.llm_engine.generate(
            prompt, sampling_params, request_id=request_id
        )
        return results_generator
        # print("Generation started")
        # print("results_generator: ", results_generator)
        # print(await results_generator.asend(None))
        # print(await results_generator.__anext__())

        # previous_text = ""
        # try:
        #     async for request_output in results_generator:
        #         if not request_output.outputs:
        #             print("No outputs found in request_output, skipping.")
        #             continue
        #         text = request_output.outputs[0].text
        #         # print(f"request_output: {text}")
        #         stream_text = text[len(previous_text) :]
        #         yield stream_text
        #         # print(text[len(previous_text):])
        #         previous_text = text
        #     await self.llm_engine.abort(request_id)
        #     print(f"request_id: {request_id} is aborted")
        # except Exception as e:
        #     print(f"Error: {e}")
        # print("Generation finished")
        # print(f"result: {text}")

MODEL_MAP: Dict[str, object] = {
    "llama": {
        'text': LlamaGenerator,
    },
    "qwen": {
        'text': LlamaGenerator,
    }
}


def generate_with_llm(model_name_or_path: str, source: List[str], config: Dict[str, Any], mode: str = 'text') -> List[List[Tuple[str, float]]]:
    generator = detect_generator(model_name_or_path, mode)
    results = generator.generate(source, config)
    del generator
    return results


def detect_generator(model_name_or_path: str, mode: str = 'text') -> object:
    for token in MODEL_MAP:
        if token in model_name_or_path.lower():
            return MODEL_MAP[token][mode](model_name_or_path)
        
def consistency(answers: List[Tuple[str, Any, float]]) -> Tuple[str, Any]:
    count: Dict[str, float] = {}
    record: Dict[str, Tuple[str, str]] = {}
    for a, b, c in answers:
        x = str(b)
        if "error" in x.lower():
            continue
        if x not in count:
            count[x] = 0
            record[x] = (a, b)
        count[x] += c
    if not count:
        return "", ""
    return record[max(count, key=lambda x: count[x])]