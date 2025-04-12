import sys
import json
import torch
import spacy
import random
import sqlparse
import os
import networkx as nx

from tqdm import tqdm
from copy import deepcopy
from functools import partial
from typing import List, Dict, Any, Tuple
from tqdm.contrib.concurrent import process_map

from rank_bm25 import BM25Okapi
from sqlparse.tokens import Token
from nltk.tokenize import word_tokenize
import jieba
from transformers import AutoModel, AutoTokenizer
from concurrent.futures import ProcessPoolExecutor


random.seed(42)
sys.path.append('.')

# # 设置 jieba 缓存目录
# os.environ["JIEBA_DATA_DIR"] = os.path.expanduser("~/.cache/jieba")

# # 初始化 jieba
# jieba.initialize()


class BaseSelector():
    def __init__(self, demonstrations: List[Dict[str, Any]]):
        assert 'question' in demonstrations[0]
        self.demonstrations = deepcopy(demonstrations)
        random.shuffle(self.demonstrations)

    def select(self, number: int, example: Dict[str, Any]) -> List[Tuple[Dict[str, Any], float]]:
        return [(d, 0) for d in self.demonstrations[:number]]


class BM25Selector(BaseSelector):
    def __init__(self, demonstrations: List[Dict[str, Any]]):
        super().__init__(demonstrations)
        self.bm25 = BM25Okapi([word_tokenize(d['question'])
                              for d in demonstrations])
        # tokens = [list(jieba.tokenize(d['question'])) for d in demonstrations]
        # tokens = [[t[0] for t in token] for token in tokens]
        # self.bm25 = BM25Okapi(tokens)
        # print(tokens)


    def select(self, number: int, example: Dict[str, Any]) -> List[Tuple[Dict[str, Any], float]]:
        question = example['question']
        question_tokenized = word_tokenize(question)
        # question_tokenized = jieba.tokenize(question)
        # tokens = list(question_tokenized)
        # question_tokenized = [t[0] for t in tokens]
        # print(tokens)
        scores = self.bm25.get_scores(question_tokenized)
        sorted_demos = [(demo, score) for score, demo in sorted(
            zip(scores, self.demonstrations), key=lambda x: x[0], reverse=True)[:number]]
        # print([(demo["table_comment"], score) for demo, score in sorted_demos])
        return sorted_demos


class EncoderSelector(BaseSelector):
    def _get_weightedmean_embedding(self, texts: List[str], is_query: bool, batch_size: int = 512) -> torch.Tensor:
        def tokenize_with_specb(texts: List[str], is_query: bool, tokenizer: AutoTokenizer):
            # Tokenize without padding
            batch_tokens = tokenizer(
                texts, padding=False, truncation=True)
            # Add special brackets & pay attention to them
            for seq, att in zip(batch_tokens["input_ids"], batch_tokens["attention_mask"]):
                if is_query:
                    seq.insert(0, self.special_tokens['specb_que_bos'])
                    seq.append(self.special_tokens['specb_que_eos'])
                else:
                    seq.insert(0, self.special_tokens['specb_doc_bos'])
                    seq.append(self.special_tokens['specb_doc_eos'])
                att.insert(0, 1)
                att.append(1)
            # Add padding
            batch_tokens = tokenizer.pad(
                batch_tokens, padding=True, return_tensors="pt")
            return batch_tokens

        def batch_encode(model: AutoModel, batch_tokens, batch_size: int):
            n = batch_tokens['input_ids'].size(0)
            all_last_hidden_states = []
            for start_index in tqdm(range(0, n, batch_size), desc="Processing batches"):
                end_index = min(start_index + batch_size, n)
                batch = {key: val[start_index:end_index]
                         for key, val in batch_tokens.items()}
                with torch.no_grad():
                    outputs = model(
                        **batch, output_hidden_states=True, return_dict=True)
                all_last_hidden_states.append(outputs.last_hidden_state)
            last_hidden_state = torch.cat(all_last_hidden_states, dim=0)
            return last_hidden_state

        batch_tokens = tokenize_with_specb(texts, is_query, self.tokenizer)
        last_hidden_state = batch_encode(self.model, batch_tokens, batch_size)
        # Get weights of shape [bs, seq_len, hid_dim]
        weights = (
            torch.arange(start=1, end=last_hidden_state.shape[1] + 1)
            .unsqueeze(0)
            .unsqueeze(-1)
            .expand(last_hidden_state.size())
            .float().to(last_hidden_state.device)
        )
        # Get attn mask of shape [bs, seq_len, hid_dim]
        input_mask_expanded = (
            batch_tokens["attention_mask"]
            .unsqueeze(-1)
            .expand(last_hidden_state.size())
            .float()
        )
        # Perform weighted mean pooling across seq_len: bs, seq_len, hidden_dim -> bs, hidden_dim
        sum_embeddings = torch.sum(
            last_hidden_state * input_mask_expanded * weights, dim=1)
        sum_mask = torch.sum(input_mask_expanded * weights, dim=1)
        embeddings = sum_embeddings / sum_mask
        return embeddings

    def __init__(self, demonstrations: List[Dict[str, Any]], model_type: str = "./model/SGPT/125m"):
        super().__init__(demonstrations)
        self.model = AutoModel.from_pretrained(model_type)
        self.tokenizer = AutoTokenizer.from_pretrained(model_type)

        self.special_tokens = {
            "specb_que_bos": self.tokenizer.encode("[", add_special_tokens=False)[0],
            "specb_que_eos": self.tokenizer.encode("]", add_special_tokens=False)[0],
            "specb_doc_bos": self.tokenizer.encode("{", add_special_tokens=False)[0],
            "specb_doc_eos": self.tokenizer.encode("}", add_special_tokens=False)[0]
        }
        self.demonstrations_embedding = self._get_weightedmean_embedding(
            [d['question'] for d in demonstrations], False)

    def select(self, number: int, example: Dict[str, Any]) -> List[Tuple[Dict[str, Any], float]]:
        return self.select_multiple(number, [example])[0]

    def select_multiple(self, number: int, examples: List[Dict[str, Any]]) -> List[List[Tuple[Dict[str, Any], float]]]:
        questions_embedding = self._get_weightedmean_embedding(
            [e['question'] for e in examples], is_query=True)
        dot_product = torch.matmul(
            questions_embedding, self.demonstrations_embedding.T)
        norm_a = torch.norm(questions_embedding, dim=1, keepdim=True)
        norm_b = torch.norm(self.demonstrations_embedding, dim=1)
        similarity = dot_product / torch.matmul(norm_a, norm_b.unsqueeze(0))

        top_values, top_indices_torch = torch.topk(similarity, number, dim=1)
        top_indices = top_indices_torch.tolist()
        top_scores = top_values.tolist()

        return [[(self.demonstrations[i], top_scores[example_idx][demo_idx]) for demo_idx, i in enumerate(top_indices[example_idx])] for example_idx in range(len(top_indices))]


class SDPSelector(BaseSelector):
    def __init__(self, demonstrations: List[Dict[str, Any]]):
        super().__init__(demonstrations)
        self.nlp = spacy.load("en_core_web_sm")

    def select(self, number: int, example: Dict[str, Any]) -> List[Dict[str, Any]]:
        def sentence_similarity(text1, text2):
            def build_dependency_graph(doc):
                G = nx.DiGraph()
                for token in doc:
                    G.add_node(token.i, text=token.text, pos=token.pos_)
                    G.add_edge(token.head.i, token.i, dep=token.dep_)
                return G

            def graph_similarity(graph1, graph2):
                nodes_intersection = set(nx.get_node_attributes(graph1, 'text').values()) & set(
                    nx.get_node_attributes(graph2, 'text').values())
                edges1 = {(u, v): data['dep']
                          for u, v, data in graph1.edges(data=True)}
                edges2 = {(u, v): data['dep']
                          for u, v, data in graph2.edges(data=True)}
                edges_intersection = set(
                    edges1.values()) & set(edges2.values())
                similarity = (len(nodes_intersection) + len(edges_intersection)) / (
                    0.5 * (len(graph1) + len(graph2) + len(graph1.edges()) + len(graph2.edges())))
                return similarity

            doc1 = self.nlp(text1)
            doc2 = self.nlp(text2)
            graph1 = build_dependency_graph(doc1)
            graph2 = build_dependency_graph(doc2)
            return graph_similarity(graph1, graph2)

        question = example['question']
        similarity_scores = [
            (demo, sentence_similarity(question, demo["question"]))
            for demo in self.demonstrations
        ]
        selected_demonstrations = sorted(
            similarity_scores, key=lambda x: x[1], reverse=True)[:number]
        return [demo[0] for demo in selected_demonstrations]


class SkeletonSelector(EncoderSelector):
    def _skeleton_mask(self, question: str, schema: Dict[str, Any]) -> str:
        spans = [c[1] for c in schema['column_names']] + schema['table_names']
        for s in spans:
            question = question.replace(s, "<mask>")
        return question

    def __init__(self, demonstrations: List[Dict[str, Any]]):
        super().__init__(demonstrations)
        for d in self.demonstrations:
            d['question'] = self._skeleton_mask(d['question'], d['schema'])

    def select(self, number: int, example: Dict[str, Any]) -> List[Tuple[Dict[str, Any], float]]:
        example = deepcopy(example)
        example['question'] = self._skeleton_mask(
            example['question'], example['schema'])
        return super().select(number, example)

    def select_multiple(self, number: int, examples: List[Dict[str, Any]]) -> List[List[Tuple[Dict[str, Any], float]]]:
        examples = deepcopy(examples)
        for e in examples:
            e['question'] = self._skeleton_mask(e['question'], e['schema'])
        return super().select_multiple(number, examples)


class SQLSelector(BaseSelector):
    def _extract_skeleton(self, sql: str) -> str:
        def should_replace(token):
            """Determine if the token should be replaced with '*'."""
            if token.value.upper() in ['MIN', 'MAX', 'SUM', 'AVG', 'COUNT', 'LENGTH', 'CAST']:
                return False
            if token.ttype in [Token.Name, Token.String, Token.Number] or token.ttype in Token.Literal:
                return True
            return False

        def process_token(token):
            """Process each token and decide whether to replace it or keep it."""
            if token.is_group:
                # If the token is a group (like a subquery or a function), process its tokens
                return token.__class__([process_token(t) for t in token.tokens])
            elif should_replace(token):
                # Replace the token with a '*' token
                return sqlparse.sql.Token(Token.Wildcard, '*')
            else:
                return token

        def print_tokens(tokens):
            print(json.dumps(
                [f"{token.ttype} {token.value} {token.is_group}" for token in tokens], ensure_ascii=False, indent=4))
            for token in tokens:
                if token.is_group:
                    print_tokens(token.tokens)

        """Extract the skeleton from an SQL query by replacing certain tokens with '*'."""
        # Parse the SQL statement
        parsed = sqlparse.parse(sql)[0]
        # Process and replace tokens as needed
        modified_tokens = [process_token(token) for token in parsed.tokens]
        # Rebuild and return the modified SQL statement
        return ''.join(str(token) for token in modified_tokens)

    def __init__(self, demonstrations: List[Dict[str, Any]], llm_name_or_path: str = './model/DeepSeek-Coder-Instruct/33b', config_file: str = './config/DeepSeek-Coder-Instruct.json', mode: str = 'chat'):
        super().__init__(demonstrations)
        for d in self.demonstrations:
            d['skeleton'] = self._extract_skeleton(d['query'])
        self.llm_name_or_path = llm_name_or_path
        self.config = json.load(open(config_file, 'r', encoding='utf-8'))
        self.mode = mode

    def select(self, number: int, example: Dict[str, Any]) -> List[Tuple[Dict[str, Any], float]]:
        return self.select_multiple(number, [example])[0]

    def select_multiple(self, number: int, examples: List[Dict[str, Any]]) -> List[List[Tuple[Dict[str, Any], float]]]:
        from utils.database import fix_sql
        from utils.generator import generate_with_llm

        prompts = [e['prompt'] for e in examples]
        predictions = generate_with_llm(
            self.llm_name_or_path, prompts, self.config, self.mode)
        skeletons = []
        for p in predictions:
            try:
                sql = fix_sql(p[0][0])
                skeletons.append(self._extract_skeleton(sql))
            except:
                skeletons.append('')

        examples_temp = [{"question": s} for s in skeletons]
        demonstrations_temp = [
            {"question": d['skeleton'], "origin": d} for d in self.demonstrations]
        results = select_multiple(
            number, examples_temp, demonstrations_temp, "bm25")
        return [[(r[0]['origin'], r[1]) for r in result] for result in results]


class SchemaSelector(BaseSelector):
    def __init__(self, demonstrations: List[Dict[str, Any]]):
        pass

    def select(self, number: int, example: Dict[str, Any]) -> List[Tuple[Dict[str, Any], float]]:
        pass


SELECTOR_MAP: Dict[str, BaseSelector] = {
    "base": BaseSelector,
    "bm25": BM25Selector,
    "encoder": EncoderSelector,
    "sdp": SDPSelector,
    "skeleton": SkeletonSelector,
    "sql": SQLSelector
}


def select_single_pair(example: Dict[str, Any], number: int, selector: BaseSelector) -> Dict[str, Any]:
    return selector.select(number, example)


def select_multiple(examples: List[Dict[str, Any]], demonstrations: List[Dict[str, str]], args=None) -> List[List[Dict[str, Any]]]:
    selector = SELECTOR_MAP[args.selector_type if args else 'bm25'](
        demonstrations)
    if hasattr(selector, "select_multiple"):
        return selector.select_multiple(args.demonstration_number if args else 3, examples)
    partial_func = partial(
        select_single_pair, number=(args.demonstration_number if args else 3), selector=selector)
    results = process_map(partial_func, examples, chunksize=1,
                          max_workers=None, total=len(examples))
    # 顺序处理每个 example
    # results = [partial_func(example) for example in examples]
    return [[x[0] for x in r] for r in results]

