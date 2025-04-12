import sys
import json
import torch
import os
import math
import requests

from typing import List, Dict, Any, Tuple
from transformers import AutoModel, AutoTokenizer
from copy import deepcopy
from .utils.my_logger import MyLogger
from collections import Counter

logger = MyLogger("murre", "logs/text2sql.log")

model_path = os.getenv("SGPT_MODEL_NAME_OR_PATH", "./model/SGPT/125m")
logger.info(f"Loading model from {model_path}")
model = AutoModel.from_pretrained(model_path)
tokenizer = AutoTokenizer.from_pretrained(model_path)
logger.info("Model loaded")


def init(model_path: str):
    global model, tokenizer
    if model is None or tokenizer is None:
        logger.info(f"Loading model from {model_path}")
        model = AutoModel.from_pretrained(model_path)
        tokenizer = AutoTokenizer.from_pretrained(model_path)
        logger.info("Model loaded")
    return model, tokenizer


def embd_tables(docs_file, embd_doc_path: str) -> List[Dict[str, Any]]:
    from .Murre.embd import open_docs_file, embd_docs

    logger.info("\nEmbedding tables")

    doc_embeddings_file = embd_doc_path
    logger.info(f"Embedding docs in {docs_file} to {doc_embeddings_file}")
    docs = open_docs_file(docs_file)
    doc_embeddings_json = embd_docs(model, tokenizer, docs)
    # print_debug(doc_embeddings_json=doc_embeddings_json[0])

    logger.info(f"Saving embeddings to {doc_embeddings_file}")
    with open(doc_embeddings_file, "w", encoding="utf-8") as f:
        json.dump(doc_embeddings_json, f, ensure_ascii=False, indent=4)

    logger.info("Embedding done\n")
    return doc_embeddings_json


def retrieve(
    doc_embeddings: str,
    queries_file: str,
    retri_dump_path: str,
    top_k: List[int] = [5],
    last_retrieved_file: str = None,
):
    from .Murre.embd import (
        embd_query,
        calculate_single_similarity,
        construct_retrieved_data,
    )

    logger.info("\nRetrieving")

    with open(doc_embeddings, "r", encoding="utf-8") as f:
        doc_embeddings_json = json.load(f)
    # queries_file = args.queries_file
    # last_retrieved_file = args.last_retrieved_file
    retrieved_file = retri_dump_path

    logger.info(f"Retrieving from {queries_file} to {retrieved_file}")
    with open(queries_file, "r", encoding="utf-8") as f:
        data = json.load(f)
    queries = [d["utterance"].split("\n") for d in data]
    original_docs = doc_embeddings_json

    idx_map = [(qi, di) for qi, q in enumerate(queries) for di in q]
    query_embeddings = embd_query(model, tokenizer, idx_map, queries)

    doc_embeddings = []
    docs = [x["pred_schema"] for x in original_docs]
    for ed in original_docs:
        tensor_data = ed.get("embedding")
        if tensor_data is not None:
            tensor = torch.tensor(tensor_data)
            doc_embeddings.append(tensor)

    if last_retrieved_file:
        schema_map = {}
        with open(last_retrieved_file, "r", encoding="utf-8") as f:
            last_retrieved_data = json.load(f)
        for di, d in enumerate(original_docs):
            schema_map[d["pred_schema"]] = int(di)

    sim = []
    sort_sim = []
    if not last_retrieved_file:
        for qi, query in enumerate(query_embeddings):
            sim, sort_sim = calculate_single_similarity(
                query, doc_embeddings, top_k, sim, sort_sim
            )
    else:
        retrieved_embeddings_map = [
            [doc_embeddings[schema_map[ret["schema"]]] for ret in data["retrieved"]]
            for data in last_retrieved_data
        ]
        for qi, query in enumerate(query_embeddings):
            retrieved_embeddings = retrieved_embeddings_map[qi]
            sim, sort_sim = calculate_single_similarity(
                query, retrieved_embeddings, top_k, sim, sort_sim
            )

    # 构造检索结果
    retrieved_data = construct_retrieved_data(
        queries=queries,
        data=data,
        sort_sim=sort_sim,
        docs=docs,
        original_docs=original_docs,
        last_retrieved_data=last_retrieved_data if last_retrieved_file else None,
        last_retrieved_file=last_retrieved_file,
        schema_map=schema_map if last_retrieved_file else None,
    )

    logger.info(f"Saving retrieved data to {retrieved_file}")
    with open(retrieved_file, "w", encoding="utf-8") as f:
        json.dump(retrieved_data, f, indent=4, ensure_ascii=False)
    logger.info("Retrieval done\n")


def score_data(
    retrieved_paths: List[str],
    dump_path: str,
    top_k: List[int] = [5],
):
    from .Murre.rewrite_fuc import find_json_files, pos, judge_stop

    logger.info(f"Scoring from {retrieved_paths} to {dump_path}")

    num_turns = len(retrieved_paths)
    # print(num_turns)
    retrieved_data = [
        [json.load(open(file, "r", encoding="utf-8")) for file in find_json_files(path)]
        for path in retrieved_paths
    ]
    logger.debug(f"retrieved_data: {retrieved_data}")

    num_data = len(retrieved_data[0][0][0]['retrieved'])
    top_k[0] = min(top_k[0], num_data)
    logger.debug(f"top_k: {top_k}") 

    # the similarity of each schema under the query
    schema_score: List[Dict[str, float]] = []
    # schema_score_t0: List[Dict[str, float]] = []

    # num_turns = len(retrieved_data[0][0]["selected_database"])

    logger.debug(f"retrieve_data[0][0]: {retrieved_data[0][0]}")
    for di, data in enumerate(retrieved_data[0][0]):
        schema_score.append({})
        for x in data["retrieved"]:
            schema_score[di].setdefault(x["schema"], 0)
    logger.debug(f"schema_score: {schema_score}")

    end_turn = [num_turns - 1] * len(retrieved_data[0][0])
    logger.debug(f"end_turn: {end_turn}")

    for ti in range(1, num_turns):
        for di, _ in enumerate(retrieved_data[1][0]):
            for fi, fdata in enumerate(retrieved_data[-1]):
                input_data = retrieved_data[ti][fi][di].get("input", [])
                if input_data and judge_stop(input_data[0]) and end_turn[di] > ti - 1:
                    end_turn[di] = ti - 1
    logger.debug(f"end_turn: {end_turn}")


    count_end_turn = Counter(end_turn)
    total_length = len(retrieved_data[0][0])
    count_end_turn = {key: val / total_length for key, val in count_end_turn.items()}
    logger.debug(f"count_end_turn: {count_end_turn}")

    for di, d0 in enumerate(retrieved_data[0][0]):
        i_turn = end_turn[di]
        for fi, fdata in enumerate(retrieved_data[i_turn]):
            d = fdata[di]
            if "question" in d and not "utterance" in d:
                d["utterance"] = d["question"]
            for x in d["retrieved"]:
                if d.get("selected_database"):
                    summ = math.log(pos(x["similarity"])) + sum(
                        math.log(pos(d["selected_database"][si][1]))
                        for si, _ in enumerate(d["selected_database"])
                    )
                else:
                    summ = pos(x["similarity"])
                schema_score[di][x["schema"]] = max(summ, schema_score[di][x["schema"]])
                if d.get("selected_database"):
                    for si, _ in enumerate(d["selected_database"]):
                        schema_score[di][d["selected_database"][si][0]] = max(
                            summ, schema_score[di][d["selected_database"][si][0]]
                        )
    logger.debug(f"schema_score: {schema_score}")
    # print(end_turn)
    logger.debug([i for i, et in enumerate(end_turn) if et == 2])

    results: List[List[Tuple[str, float]]] = []
    for si, s in enumerate(schema_score):
        results.append([])
        results[-1] = sorted(s.items(), key=lambda x: x[1], reverse=True)[: max(top_k)]
        retrieved = [{} for _ in range(max(top_k))]
        for k in range(max(top_k)):
            retrieved[k]["rank"] = k
            retrieved[k]["schema"] = results[si][k][0]
            retrieved[k]["similarity"] = results[si][k][1]
        utterance_org = retrieved_data[0][0][si].pop("question", None)
        retrieved_data[0][0][si].pop("pred_schema", None)
        retrieved_data[0][0][si]["utterance_org"] = utterance_org
        retrieved_data[0][0][si]["utterance"] = [
            [
                retrieved_data[i][fi][si]["utterance"]
                for fi in range(len(retrieved_data[i]))
            ]
            for i in range(1, len(retrieved_data))
        ]
        retrieved_data[0][0][si]["turn0_selected_database"] = [
            t["schema"] for t in retrieved_data[0][0][si]["retrieved"][:5]
        ]
        retrieved_data[0][0][si]["retrieved"] = retrieved
        # retrieved_data[0][0][si]["recall"] = compute_recall_multiple(args.top_k, [x['schema'] for x in retrieved], retrieved_data[0][0][si]['gold'])

    logger.info(f"Saving scored data to {dump_path}\n")
    os.makedirs(os.path.dirname(dump_path), exist_ok=True)
    with open(dump_path, "w", encoding="utf-8") as f:
        json.dump(retrieved_data[0][0], f, ensure_ascii=False, indent=4)


def infer_sql(query_path: str, tables_file: str, top_k: int = 5):
    from .Murre.rewrite_fuc import construct_tables_input

    print("\nInfering")

    # with open(config_file, 'r', encoding='utf-8') as f:
    #     config = json.load(f)
    with open(query_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    with open(tables_file, "r", encoding="utf-8") as f:
        org_dbs_dict = json.load(f)
    dbs_dict = {d["db_id"]: d for d in org_dbs_dict}
    tables = []
    for d in data:
        table = construct_tables_input(
            [r["schema"] for r in d["retrieved"][:top_k]], dbs_dict
        )
        # print(table)
        tables.append(table)
    return tables


def construct_db_input(query_path: str, question, db_map, top_k: int = 5):
    from .Murre.rewrite_fuc import extract_db_names
    from .utils.database import database_to_string

    with open(query_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    

    cho_db = extract_db_names([r["schema"] for r in data[0]["retrieved"][:top_k]])
    logger.debug(f'cho_db: {cho_db}')
    logger.debug(f'db_map: {db_map}')
    db_file = db_map[cho_db]
    logger.debug(f'db_file: {db_file}')
    schema = database_to_string(db_file, question=question)
    # print(data[0]["input"])
    return schema, cho_db


if __name__ == "__main__":
    model_path = "./model/SGPT/125m"
    init(model_path)
    embd_tables("cache/tables.json", "cache/embedding.json")
    retrieve_dir = "cache/retrieve/turn0"
    os.makedirs(retrieve_dir, exist_ok=True)
    retrieve(
        "cache/embedding.json",
        "cache/question.json",
        "cache/retrieve/turn0/retrieved.json",
    )
    score_data([retrieve_dir], "cache")
    tables = construct_db_input("cache/dev.json", "cache/tables.json", "cache")
    print(tables)
