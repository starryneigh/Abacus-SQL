import sys
import json
import torch
import argparse

from typing import List, Dict, Any
from transformers import AutoModel, AutoTokenizer
from scipy.spatial.distance import cosine
import math


def compute_recall(pred_list: List, gold_list: List):
    correct_count = 0
    for p in pred_list:
        if p in gold_list:
            correct_count += 1
    rec = 1.0 * correct_count / len(gold_list)
    return rec


# tokenize schema with special brackets
def tokenize_with_specb(tokenizer, texts, is_query):
    SPECB_QUE_BOS = tokenizer.encode("[", add_special_tokens=False)[0]
    SPECB_QUE_EOS = tokenizer.encode("]", add_special_tokens=False)[0]

    SPECB_DOC_BOS = tokenizer.encode("{", add_special_tokens=False)[0]
    SPECB_DOC_EOS = tokenizer.encode("}", add_special_tokens=False)[0]
    # Tokenize without padding
    batch_tokens = tokenizer(texts, padding=False, truncation=True)
    # Add special brackets & pay attention to them
    for seq, att in zip(batch_tokens["input_ids"], batch_tokens["attention_mask"]):
        if is_query:
            seq.insert(0, SPECB_QUE_BOS)
            seq.append(SPECB_QUE_EOS)
        else:
            seq.insert(0, SPECB_DOC_BOS)
            seq.append(SPECB_DOC_EOS)
        att.insert(0, 1)
        att.append(1)
    # Add padding
    batch_tokens = tokenizer.pad(batch_tokens, padding=True, return_tensors="pt")
    return batch_tokens


# Get weighted mean embedding
def get_weightedmean_embedding(batch_tokens, model):
    # Get the embeddings
    with torch.no_grad():
        # Get hidden state of shape [bs, seq_len, hid_dim]
        last_hidden_state = model(
            **batch_tokens, output_hidden_states=True, return_dict=True
        ).last_hidden_state
        # 新版的transformers可能不支持las_hidden_state
        # 如果29行代码报错，可以用下面的代码代替
        # last_hidden_state = model(**batch_tokens, output_hidden_states=True, return_dict=True).hidden_states[-1]

    # Get weights of shape [bs, seq_len, hid_dim]
    weights = (
        torch.arange(start=1, end=last_hidden_state.shape[1] + 1)
        .unsqueeze(0)
        .unsqueeze(-1)
        .expand(last_hidden_state.size())
        .float()
        .to(last_hidden_state.device)
    )

    # Get attn mask of shape [bs, seq_len, hid_dim]
    input_mask_expanded = (
        batch_tokens["attention_mask"]
        .unsqueeze(-1)
        .expand(last_hidden_state.size())
        .float()
    )

    # Perform weighted mean pooling across seq_len: bs, seq_len, hidden_dim -> bs, hidden_dim
    sum_embeddings = torch.sum(last_hidden_state * input_mask_expanded * weights, dim=1)
    sum_mask = torch.sum(input_mask_expanded * weights, dim=1)

    embeddings = sum_embeddings / sum_mask

    return embeddings


def open_docs_file(docs_file: str) -> List[str]:
    docs = []
    docs_type = docs_file.split("/")[-1].split(".")[-1]

    if docs_type == "json":
        with open(docs_file, "r", encoding="utf-8") as f:
            orginal_tables = json.load(f)
        for d in orginal_tables:
            tables_schema = d["schema"]
            docs += tables_schema
    elif docs_type == "txt":
        with open(docs_file, "r", encoding="utf-8") as f:
            schema_string = f.read()
        docs = schema_string.split("\n")
    return docs


def embd_docs(model, tokenizer, docs):
    doc_embeddings_json = []
    doc_embeddings = get_weightedmean_embedding(
        tokenize_with_specb(tokenizer, docs, is_query=False), model
    )
    for di, d in enumerate(doc_embeddings):
        doc_data = {}
        doc_data.setdefault("pred_schema", docs[di])
        doc_data.setdefault("embedding", d.tolist())
        doc_embeddings_json.append(doc_data)
    return doc_embeddings_json


def embd_query(model, tokenizer, idx_map, queries, batch_size=4):
    batch_num = math.ceil(len(idx_map) / batch_size)
    embeddings_list = []
    for b in range(batch_num):
        # 计算当前批次的起始和结束索引
        start_index = b * batch_size
        end_index = min((b + 1) * batch_size, len(idx_map))
        # 提取当前批次的文本
        batch_texts = [x[1] for x in idx_map[start_index:end_index]]
        # 对当前批次的文本进行分词和嵌入计算
        batch_tokens = tokenize_with_specb(tokenizer, batch_texts, is_query=True)
        query_embeddings0 = get_weightedmean_embedding(batch_tokens, model)
        # print(len(query_embeddings0))
        embeddings_list.append(query_embeddings0)

    query_embeddings = torch.cat(embeddings_list, dim=0)
    # print(query_embeddings.size())
    # print(query_embeddings[-2:])
    # print(doc_embeddings[:5])

    query_embeddings = [
        [x for i, x in enumerate(query_embeddings) if idx_map[i][0] == qi]
        for qi in range(len(queries))
    ]
    # print(len(query_embeddings))
    return query_embeddings


# Calculate cosine similarities
# Cosine similarities are in [-1, 1]. Higher means more similar
def calculate_single_similarity(qe, de, top_k, sim, sort_sim):
    sim.append([])
    # print(len(qe))
    for qi, q in enumerate(qe):
        # sort_sim.append([])
        for di, d in enumerate(de):
            cosine_sim = 1 - cosine(q, d)
            sim[-1].append((di, cosine_sim))
        # assert len(doc_embeddings) == len(sim[qi])
    sort_qi = sorted(sim[-1], key=lambda x: (x[1]), reverse=True)
    qi_sort_sim = []
    record_doc = []
    for sqi in sort_qi:
        if len(qi_sort_sim) == max(top_k):
            break
        if sqi[0] not in record_doc:
            qi_sort_sim.append(sqi)
            record_doc.append(sqi[0])
        # sort_sim[-1].append(sqi)
    sort_sim.append(qi_sort_sim)
    return sim, sort_sim


def construct_retrieved_data(
    queries: List[str],
    data: List[Dict[str, Any]],
    sort_sim: List[List[List[int]]],
    docs: List[str],
    original_docs: List[Dict[str, Any]],
    last_retrieved_data: List[Dict[str, Any]],
    last_retrieved_file: str,
    schema_map: Dict[str, int],
) -> List[Dict[str, Any]]:
    retrieved_data = []
    for qi, query in enumerate(queries):
        user_question = data[qi]["utterance"]
        retrieved_docs = []
        if last_retrieved_file:
            org_docs = [x["pred_schema"] for x in original_docs]
            docs = [
                org_docs[schema_map[x]]
                for x in [ret["schema"] for ret in last_retrieved_data[qi]["retrieved"]]
            ]
        if data[qi].get("utterance_org") is not None:
            if isinstance(data[qi]["utterance_org"], str):
                utterance_org = [data[qi]["utterance_org"]]
            elif isinstance(data[qi]["utterance_org"], list):
                utterance_org = data[qi]["utterance_org"]
        else:
            utterance_org = []

        # 处理排序后的相似度列表
        for rank, (doc_index, similarity) in enumerate(sort_sim[qi]):
            qi_ret = {"rank": rank, "schema": docs[doc_index], "similarity": float(similarity)}
            retrieved_docs.append(qi_ret)
        selected_database = data[qi].get("selected_database", [])
        retrieved_data.append(
            {
                "utterance": user_question,
                "input": query,
                "utterance_org": utterance_org,
                "retrieved": retrieved_docs,
                "selected_database": selected_database,
            }
        )
    return retrieved_data
