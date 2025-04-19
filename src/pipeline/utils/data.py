from pydantic import BaseModel, BeforeValidator
from typing import Dict, Any
import json
from typing_extensions import Annotated

def parse_json_string(v: str) -> Any:
    if isinstance(v, str):
        try:
            return json.loads(v)
        except json.JSONDecodeError:
            raise ValueError('Input is not a valid JSON string')
    return v

class GenerateSQLRequest(BaseModel):
    question: str
    history: Annotated[list[dict], BeforeValidator(parse_json_string)] = []
    prompts_user: Annotated[list[dict], BeforeValidator(parse_json_string)] = []
    db_infos: Annotated[list[dict], BeforeValidator(parse_json_string)] = []
    cache_path: str = "./cache"
    demonstration: bool = False
    demo_num: int = 5
    question_num: int = 5
    pre_generate_sql: bool = True
    self_debug: bool = True
    encore: bool = False
    entity_debug: bool = False
    skeleton_debug: bool = False
    align_flag: bool = False
    mode: str = "ch"
    model_name: str = "Qwen2.5-Coder_7b"
    api_key: str = "None"
    api_base: str = "None"
    api_data: dict = {}
    db_id_path_map: Dict[str, str] = None # 添加 db_id_path_map 字段