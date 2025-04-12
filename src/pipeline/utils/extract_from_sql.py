import re
import sqlparse
from sqlparse.sql import IdentifierList, Identifier, Comparison
from sqlparse.tokens import Keyword, DML, Name
from .my_logger import MyLogger

logger = MyLogger("extract_from_sql", "logs/text2sql.log")

def is_subselect(parsed):
    if not parsed.is_group:
        return False
    for item in parsed.tokens:
        if item.ttype is DML and item.value.upper() == 'SELECT':
            return True
    return False

def extract_tables(parsed):
    from_seen = False
    tables = []
    for item in parsed.tokens:
        # 如果是子查询，递归提取表名
        # print(from_seen)
        if is_subselect(item):
            tables += extract_tables(item)
        if from_seen:
            if isinstance(item, IdentifierList):
                for identifier in item.get_identifiers():
                    tables.append(identifier.get_real_name())
            elif isinstance(item, Identifier):
                tables.append(item.get_real_name())
            elif item.ttype is Keyword:
                from_seen = False
        # 检测到 FROM 或 JOIN 关键字
        if item.ttype is Keyword and item.value.upper() in ('FROM', 'JOIN'):
            from_seen = True
    return tables

def extract_columns(parsed):
    columns = []
    select_seen = False
    for item in parsed.tokens:
        # 确保只处理SELECT之后的部分，直到遇到FROM
        if select_seen:
            # 如果遇到 FROM 或其他关键字，停止列名提取
            if isinstance(item, sqlparse.sql.Token) and item.ttype is Keyword and item.value.upper() == 'FROM':
                break

            # 如果是 IdentifierList (多个列)
            if isinstance(item, IdentifierList):
                for identifier in item.get_identifiers():
                    col_name = identifier.get_real_name()
                    if col_name and col_name.upper() not in ['DISTINCT']:
                        columns.append(col_name)
            # 如果是 Identifier (单个列)
            elif isinstance(item, Identifier):
                col_name = item.get_real_name()
                if col_name and col_name.upper() not in ['DISTINCT']:
                    columns.append(col_name)
            # 处理直接列出的字段 (如 COUNT(*))
            elif item.ttype is Name:
                col_name = item.value
                if col_name and col_name.upper() not in ['DISTINCT']:
                    columns.append(col_name)
                    
        # 如果遇到 SELECT 关键字，开始提取列名
        if isinstance(item, sqlparse.sql.Token) and item.ttype is DML and item.value.upper() == 'SELECT':
            select_seen = True

    return columns

def extract_conditions_columns(parsed):
    """提取 ON 和 WHERE 子句中的列名"""
    condition_columns = []
    
    for item in parsed.tokens:
        # 检测 WHERE 或 JOIN ON 子句
        if isinstance(item, sqlparse.sql.Where):  # 处理 WHERE 子句
            for subitem in item.tokens:
                if isinstance(subitem, Comparison):  # Comparison 表示比较操作（如 =、> 等）
                    for identifier in subitem.get_sublists():
                        if isinstance(identifier, Identifier):
                            condition_columns.append(identifier.get_real_name())
                elif isinstance(subitem, Identifier):
                    condition_columns.append(subitem.get_real_name())
        # 处理 ON 子句的列
        elif isinstance(item, IdentifierList) or isinstance(item, Comparison):
            for identifier in item.get_sublists():
                if isinstance(identifier, Identifier):
                    condition_columns.append(identifier.get_real_name())

    return condition_columns

def extract_columns_and_tables(sql_query):
    # 使用 sqlparse 解析 SQL
    parsed = sqlparse.parse(sql_query)
    if parsed:
        parsed = parsed[0]
    else:
        return [], []
    
    # 提取 SELECT 中的列名
    columns = extract_columns(parsed)
    # print(columns)
    
    # 提取 WHERE 和 ON 中的列名
    condition_columns = extract_conditions_columns(parsed)
    # print(condition_columns)
    
    # 提取 FROM 和 JOIN 后的表名
    tables = extract_tables(parsed)
    
    # 合并 SELECT 列名和条件中的列名
    all_columns = list(set(columns + condition_columns))
    
    return all_columns, list(set(tables))


def analyze_sql_syntax(sql_statement: str) -> None:
    """
    分析并验证SQL语句的语法是否正确。

    参数:
        sql_statement (str): 要分析的SQL查询语句。

    返回:
        None: 函数将打印SQL语句的有效性或错误信息。
    """
    try:
        parsed = sqlparse.parse(sql_statement)
        if len(parsed) == 0:
            logger.debug(f"Invalid SQL: {sql_statement}")
            return
        # 检查是否是 SELECT 语句
        if parsed[0].get_type() == "SELECT":
            logger.debug(f"Valid SQL: {sql_statement}")
        else:
            logger.debug(f"Invalid SQL (Not a SELECT statement): {sql_statement}")
    except Exception as e:
        logger.debug(f"Syntax error in SQL: {sql_statement}\nError: {e}")

def get_rationale(prediction):
    """获取解释性 SQL 语句"""
    content = prediction.lower()
    if "rationale:" in content and "sql:" in content:
        rationale = content.split("rationale:")[1].split("sql:")[0].strip()
    elif "推理过程:" in content and "sql:" in content:
        rationale = content.split("推理过程:")[1].split("sql:")[0].strip()
    else:
        rationale = content
    return rationale

def extract_sql_from_text(text: str) -> list[str]:
    """
    从给定文本中提取所有以SELECT开头的SQL查询语句。

    参数:
        text (str): 包含自然语言和SQL语句的文本。

    返回:
        list[str]: 提取出的SQL语句列表。
    """
    # 正则表达式：匹配以SELECT开头，并以分号结尾的SQL语句
    pattern = r'(select.*?;)'  # 非贪婪匹配以分号结尾的SQL语句
    sql_statements = re.findall(pattern, text, re.IGNORECASE | re.DOTALL)  # 忽略大小写并匹配换行
    # print(sql_statements)
    if '```sql' in text:
        sql = text.split('```sql')[-1].split('```')[0].strip()
    elif sql_statements:
        sql = sql_statements[0]
    elif "SQL:" in text:
        sql = text.split("SQL:")[1].strip()
    elif "\n\n" in text:
        sql = text.split("\n\n")[0]
    else:
        sql = text
    # print(sql)
    return sql

# 弃用
def extract_table_and_columns(sql: str) -> tuple[list[str], list[str]]:
    """
    提取SQL语句中的表名和列名。

    参数:
        sql (str): 要分析的SQL查询语句。

    返回:
        tuple[list[str], list[str]]: 返回一个元组，包含表名列表和列名列表。
    """
    # 正则表达式提取表名和列名
    table_pattern = re.compile(r"from\s+([a-zA-Z_][a-zA-Z0-9_ ]*)", re.IGNORECASE)
    column_pattern = re.compile(r"select\s+(.*?)\s+from", re.IGNORECASE)
    where_pattern = re.compile(r"where\s+(.*?)\s*;", re.IGNORECASE)

    # 提取表名
    tables = table_pattern.findall(sql)
    tables = [table.strip().lower() for table in tables]
    logger.debug(f"Extracted Tables: {tables}")

    # 提取列名
    columns = column_pattern.findall(sql)
    columns = columns[0].split(",") if columns else []
    columns = [col.split("as")[0].strip().lower() for col in columns]
    logger.debug(f"Extracted Columns: {columns}")

    # 提取WHERE条件中的列
    where_conditions = where_pattern.findall(sql)
    if where_conditions:
        where_columns = re.findall(r"([a-zA-Z_][a-zA-Z0-9_]*)\s*=", where_conditions[0])
        logger.debug(f"Extracted Where columns: {where_columns}")
        columns += [col.strip().lower() for col in where_columns]

    return tables, columns


def match_database(sql: str, databases: list[dict]) -> str | None:
    """
    根据SQL语句中的表名和列名匹配对应的数据库ID。

    参数:
        sql (str): 要匹配的SQL查询语句。
        databases (list[dict]): 包含数据库模式的列表，每个数据库包含表名、列名等信息。

    返回:
        str | None: 匹配到的数据库ID，如果未匹配到则返回None。
    """
    columns_in_query, tables_in_query = extract_columns_and_tables(sql)
    if not columns_in_query or not tables_in_query:
        return databases[0]["db_id"]
    # 将所有列名表名都小写
    try:
        columns_in_query = [col.lower() for col in columns_in_query]
        tables_in_query = [table.lower() for table in tables_in_query]
        logger.debug(f"Columns in Query: {columns_in_query}")
        logger.debug(f"Tables in Query: {tables_in_query}")
    except Exception as e:
        logger.debug(f"Error in SQL: {sql}\nError: {e}")
        return databases[0]["db_id"]

    max_match_score = 0
    for db in databases:
        matched_tables = []
        matched_columns = []

        # 如果查询使用了通配符 (*)，直接匹配所有列
        if "*" in columns_in_query:
            matched_columns.append("*")

        for table in tables_in_query:
            if table in [t.lower() for t in db["table_names"]]:
                matched_tables.append(table)

                # 找到表对应的所有列
                table_index = db["table_names"].index(
                    next(t for t in db["table_names"] if t.lower() == table)
                )
                columns_in_db = [
                    col[1].lower()
                    for col in db["column_names"]
                    if col[0] == table_index
                ]

                # 检查查询语句中的列是否在数据库列中
                for col in columns_in_query:
                    if col in columns_in_db and col not in matched_columns:
                        matched_columns.append(col)

        logger.debug(f"Matched Tables: {matched_tables}")
        logger.debug(f"Matched Columns: {matched_columns}")
        # 如果表和列完全匹配，返回数据库ID
        if matched_tables == tables_in_query and matched_columns == columns_in_query:
            db_id = db["db_id"]
            logger.debug(f"The SQL query belongs to database: {db_id}")
            return db_id
        # 如果部分匹配，计算匹配分数
        else:
            match_score = len(matched_tables) + len(matched_columns)
            if match_score >= max_match_score:
                max_match_score = match_score
                best_match = db["db_id"]

    logger.debug(f"The best match is: {best_match}")
    return best_match


def extract_sql_and_db(text: str, databases: list[dict]) -> str | None:
    """
    从给定文本中提取SQL语句，并匹配对应的数据库ID。

    参数:
        text (str): 包含自然语言和SQL语句的文本。
        databases (list[dict]): 包含数据库模式的列表，每个数据库包含表名、列名等信息。

    返回:
        str | None: 匹配到的数据库ID，如果未匹配到则返回None。
    """
    extracted_sql = extract_sql_from_text(text)
    analyze_sql_syntax(extracted_sql)
    db_id = match_database(extracted_sql, databases)
    return extracted_sql, db_id

if __name__ == "__main__":
    # 测试数据
    databases = [
        {
            "column_names": [
                [-1, "*"],
                [0, "actid"],
                [0, "activity name"],
                [1, "stuid"],
                [1, "actid"],
                [2, "facid"],
                [2, "actid"],
                [3, "stuid"],
                [3, "lname"],
                [3, "fname"],
                [3, "age"],
                [3, "sex"],
                [3, "major"],
                [3, "advisor"],
                [3, "city code"],
                [4, "facid"],
                [4, "lname"],
                [4, "fname"],
                [4, "rank"],
                [4, "sex"],
                [4, "phone"],
                [4, "room"],
                [4, "building"],
            ],
            "column_names_original": [
                [-1, "*"],
                [0, "actid"],
                [0, "activity_name"],
                [1, "stuid"],
                [1, "actid"],
                [2, "FacID"],
                [2, "actid"],
                [3, "StuID"],
                [3, "LName"],
                [3, "Fname"],
                [3, "Age"],
                [3, "Sex"],
                [3, "Major"],
                [3, "Advisor"],
                [3, "city_code"],
                [4, "FacID"],
                [4, "Lname"],
                [4, "Fname"],
                [4, "Rank"],
                [4, "Sex"],
                [4, "Phone"],
                [4, "Room"],
                [4, "Building"],
            ],
            "column_types": [
                "text",
                "number",
                "text",
                "number",
                "number",
                "number",
                "number",
                "number",
                "text",
                "text",
                "number",
                "text",
                "number",
                "number",
                "text",
                "number",
                "text",
                "text",
                "text",
                "text",
                "number",
                "text",
                "text",
            ],
            "foreign_keys": [[4, 1], [3, 7], [6, 1], [5, 15]],
            "primary_keys": [1, 7, 15],
            "table_names": [
                "activity",
                "participates in",
                "faculty participates in",
                "student",
                "faculty",
            ],
            "table_names_original": [
                "Activity",
                "Participates_in",
                "Faculty_Participates_in",
                "Student",
                "Faculty",
            ],
            "db_id": "activity_1",
            "schema": [
                "activity_1.activity(actid, activity name)",
                "activity_1.participates in(stuid, actid)",
                "activity_1.faculty participates in(facid, actid)",
                "activity_1.student(stuid, lname, fname, age, sex, major, advisor, city code)",
                "activity_1.faculty(facid, lname, fname, rank, sex, phone, room, building)",
            ],
        },
        {
            "column_names": [
                [-1, "*"],
                [0, "perpetrator id"],
                [0, "people id"],
                [0, "date"],
                [0, "year"],
                [0, "location"],
                [0, "country"],
                [0, "killed"],
                [0, "injured"],
                [1, "people id"],
                [1, "name"],
                [1, "height"],
                [1, "weight"],
                [1, "home town"],
            ],
            "column_names_original": [
                [-1, "*"],
                [0, "Perpetrator_ID"],
                [0, "People_ID"],
                [0, "Date"],
                [0, "Year"],
                [0, "Location"],
                [0, "Country"],
                [0, "Killed"],
                [0, "Injured"],
                [1, "People_ID"],
                [1, "Name"],
                [1, "Height"],
                [1, "Weight"],
                [1, "Home Town"],
            ],
            "column_types": [
                "text",
                "number",
                "number",
                "text",
                "number",
                "text",
                "text",
                "number",
                "number",
                "number",
                "text",
                "number",
                "number",
                "text",
            ],
            "foreign_keys": [[2, 9]],
            "primary_keys": [1, 9],
            "table_names": ["perpetrator", "people"],
            "table_names_original": ["perpetrator", "people"],
            "db_id": "perpetrator",
            "schema": [
                "perpetrator.perpetrator(perpetrator id, people id, date, year, location, country, killed, injured)",
                "perpetrator.people(people id, name, height, weight, home town)",
            ],
        },
    ]
    # 示例文本
    text = "Question: please give me the name and phone number of the faculty member who is in charge of the computer science department\nSELECT abc from people;"
    # 提取 SQL 语句
    extracted_sql = extract_sql_from_text(text)
    # 分析提取的 SQL 语句
    analyze_sql_syntax(extracted_sql)
    # 匹配数据库
    db_id = match_database(extracted_sql, databases)
