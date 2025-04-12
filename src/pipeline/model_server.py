import time
import json
import sys
import threading
import uvicorn
import argparse
from typing import List, Dict, Any, Tuple
from fastapi.responses import StreamingResponse
from fastapi import FastAPI, Request
from .utils.stream_generator import detect_generator
from .utils.my_logger import MyLogger
from .utils.server_thread import UvicornServerThread

logger = MyLogger("smodel_load", "logs/model.log")


def load_model(model_name_or_path: str) -> object:
    logger.info(f"Loading model from {model_name_or_path}")
    generator = detect_generator(model_name_or_path)
    logger.info("Model loaded")
    return generator


def create_model_app(
    model_name_or_path: str,
    config_file: str,
):
    app = FastAPI()
    global generator
    generator = load_model(model_name_or_path)
    with open(config_file, "r", encoding="utf-8") as f:
        config = json.load(f)

    @app.post("/predict")
    async def predict(request: Request):
        logger.info("Request received")
        data = await request.json()
        logger.debug(data)
        prompt: List[str] = data.get("prompt")
        if not prompt:
            return {"error": "prompt is required"}
        logger.info(prompt)
        # if "config" in data:
        #     config = data["config"]
        request_id = time.monotonic()
        stream_gen = await generator.generate(prompt, config, request_id)

        async def streaming_resp():
            pre_res = ""
            async for item in stream_gen:
                res = item.outputs[0].text
                logger.debug(res)
                yld_res = res[len(pre_res) :]
                yield (json.dumps({"generator": yld_res}) + "\0").encode("utf-8")
                pre_res = res
            await generator.llm_engine.abort(request_id)

        return StreamingResponse(streaming_resp())
    
    return app


def start_model(
    model_name_or_path: str,
    config_file: str,
    port: int = 5000,
    host: str = "0.0.0.0",
):
    app = create_model_app(model_name_or_path, config_file)
    server_thread = UvicornServerThread(app, host, port)
    server_thread.start()
    return server_thread


def parser():
    args = argparse.ArgumentParser()
    args.add_argument("--model_name_or_path", type=str, default="./model/Qwen/7b")
    args.add_argument("--config_file", type=str, default="./config/Qwen.json")
    args.add_argument("--port", type=int, default=5000)
    return args.parse_args()


if __name__ == "__main__":
    args = parser()
    port = args.port
    model_name_or_path = args.model_name_or_path
    config_file = args.config_file

    start_model(model_name_or_path, config_file, port)
    while True:
        print("Server is running")
        time.sleep(20)
