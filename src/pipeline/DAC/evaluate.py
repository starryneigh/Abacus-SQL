import os
import json
import random
import sqlparse

from tqdm import tqdm
from ..utils.database import extract_schema
from typing import List, Dict, Any, Union

random.seed(42)
os.environ["TOKENIZERS_PARALLELISM"] = "true"


def extract_sql_entities(sql: str, schema: Dict[str, Any]) -> Dict[str, List[str]]:
    from utils.database import remove_on_clause

    def align_schema_entities(sql: str, entities: List[str], schema: Dict[str, Any]) -> Dict[str, List[str]]:
        from utils.database import extract_table_aliases

        result: Dict[str, List[str]] = {
            "table": [],
            "column": []
        }
        aliases = extract_table_aliases(sql)

        result['table'] = [x.lower() for x in aliases.values()]
        table_names = [name.lower() for name in schema['table_names_original']]
        for e in entities:
            if e.lower() in table_names:
                result['table'].append(e)
        result['table'] = list(set([t.lower() for t in result['table']]))

        columns: List[str] = []
        for e in entities:
            used_flag = False
            if e.lower() in result['table']:
                continue
            for t in aliases.keys():
                if f" {t}.{e}".lower() in sql.lower():
                    result['column'].append(f"{aliases[t]}.{e}")
                    used_flag = True
                    break
            if not used_flag:
                for t in result['table']:
                    if f" {t}.{e}".lower() in sql.lower():
                        result['column'].append(f"{t}.{e}")
                        used_flag = True
                        break
            if not used_flag:
                columns.append(e)

        for e in columns:
            for c in schema['column_names_original']:
                if e.lower() == c[1].lower() and schema['table_names_original'][c[0]].lower() in result['table']:
                    result['column'].append(
                        f"{schema['table_names_original'][c[0]]}.{c[1]}")
                    break
        return result

    sql_removed_on_clause = remove_on_clause(
        list(sqlparse.parse(sql)[0]))[0]
    schema_entities = extract_schema(sql_removed_on_clause, schema)
    return align_schema_entities(sql_removed_on_clause, schema_entities, schema)


def compare_sql_entities(pred: str, gold: str, schema: Dict[str, Any]) -> Dict[str, List[str]]:
    pred_entities = extract_sql_entities(pred, schema)
    gold_entities = extract_sql_entities(gold, schema)

    pred_entities['table'] = sorted([x.lower()
                                    for x in pred_entities['table']])
    gold_entities['table'] = sorted([x.lower()
                                    for x in gold_entities['table']])
    pred_entities['column'] = sorted(
        [x.lower() for x in pred_entities['column']])
    gold_entities['column'] = sorted(
        [x.lower() for x in gold_entities['column']])

    result = {
        "table": [x for x in gold_entities['table'] if x not in pred_entities['table']],
        "column": [x for x in gold_entities['column'] if x not in pred_entities['column']]
    }
    return result


def extract_alignment_entities(alignment: List[Dict[str, str]]) -> Dict[str, List[str]]:
    result = {
        "table": [],
        "column": []
    }
    for a in alignment:
        if not a['type'] or not a['schema'] or '*' in a['schema']:
            continue
        if a['type'] in ['col', 'val']:
            result['column'].append(a['schema'])
        if a['type'] == 'tbl':
            result['table'].append(a['schema'])
    result['table'] = list(set(result['table']))
    result['column'] = list(set(result['column']))
    return result


def is_fully_contain(sql: str, schema: Dict[str, Any], alignment: Union[List[Dict[str, str]], str]) -> bool:
    sql_entities = extract_sql_entities(sql, schema)
    alignment_entities = extract_sql_entities(alignment, schema) if isinstance(
        alignment, str) else extract_alignment_entities(alignment)

    sql_entities['table'] = [x.lower() for x in sql_entities['table']]
    sql_entities['column'] = [x.lower() for x in sql_entities['column']]
    alignment_entities['table'] = [x.lower()
                                   for x in alignment_entities['table']]
    alignment_entities['column'] = [x.lower()
                                    for x in alignment_entities['column']]

    return set(sql_entities['table']).issubset(set(alignment_entities['table'])) and set(sql_entities['column']).issubset(set(alignment_entities['column']))


if __name__ == '__main__':
    # with open('./result/Bird/8b/dev.generate.eval.json', 'r', encoding='utf-8') as f:
    #     data_generated = json.load(f)
    # with open('./result/Bird/8b/dev.debug.eval.json', 'r', encoding='utf-8') as f:
    #     data_debug = json.load(f)
    # with open('./result/Bird/8b/dev.debug.json', 'r', encoding='utf-8') as f:
    #     data = json.load(f)

    # result = []
    # for o, g, d in zip(data, data_generated, data_debug):
    #     if not(g['eval'] and not d['eval']):
    #         continue
    #     result.append({
    #         "db_id": o['db_id'],
    #         "question": o['question'],
    #         "query": {
    #             "gold": o['query'],
    #             "pred": d['pred']
    #         },
    #         "alignment": extract_alignment_entities(o['alignment']['pred'])
    #     })

    # with open('./result/Bird/8b/dev.debug.compare.json', 'w', encoding='utf-8') as f:
    #     json.dump(result, f, ensure_ascii=False, indent=4)

    # =================================================================================

    with open('./dataset/Spider/tables.json', 'r', encoding='utf-8') as f:
        schemas = {schema['db_id']: schema for schema in json.load(f)}
    with open('./result/Spider/dev.align.json', 'r', encoding='utf-8') as f:
        data_pred = json.load(f)
    with open('./result/Spider/8b/dev.debug.eval.json', 'r', encoding='utf-8') as f:
        data_eval = json.load(f)
        for p, e in zip(data_pred, data_eval):
            p['eval'] = e['eval'] if 'eval' in e else e['correct']
    with open('./dataset/Spider/dev.json', 'r', encoding='utf-8') as f:
        data_gold = json.load(f)

    result = []
    result_eval = {}
    fully_contain = 0
    for pred, gold in tqdm(zip(data_pred, data_gold), total=len(data_gold)):
        if pred['eval']:
            continue

        pred['query_pred'] = pred['prediction']['query']
        pred['mismatch_sql_entities'] = compare_sql_entities(
            pred['prediction']['query'], gold['query'], schemas[gold['db_id']])
        if not pred['mismatch_sql_entities']['table'] and not pred['mismatch_sql_entities']['column']:
            continue
        pred['alignment_entities'] = extract_alignment_entities(
            pred['alignment']['pred'])
        for part in ['mismatch_sql_entities', 'alignment_entities']:
            for evidence in ['table', 'column']:
                for i, x in enumerate(pred[part][evidence]):
                    pred[part][evidence][i] = x.lower()

        pred['error'] = []
        for t in pred['mismatch_sql_entities']['table']:
            if t not in pred['alignment_entities']['table']:
                pred['error'].append('table alignment mismatch')
        for c in pred['mismatch_sql_entities']['column']:
            if c not in pred['alignment_entities']['column']:
                pred['error'].append('column alignment mismatch')
        pred['error'] = list(set(pred['error']))
        pred_error = ' | '.join(pred['error'])
        result_eval[pred_error] = result_eval.get(pred_error, 0) + 1

        if is_fully_contain(gold['query'], schemas[gold['db_id']], pred['alignment']['pred']):
            fully_contain += 1

        pred.pop('prediction')
        result.append(pred)

    print(
        f"Fully contain: {fully_contain}/{len([d for d in data_pred if not d['eval']])}")
    print(json.dumps(result_eval, ensure_ascii=False, indent=4))
    random.shuffle(result)
    with open('./result/Spider/8b/dev.debug.compare.json', 'w', encoding='utf-8') as f:
        json.dump(result, f, ensure_ascii=False, indent=4)
