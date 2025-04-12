import os
import re
import sqlite3

from typing import Dict, List, Any
from func_timeout import func_timeout


DATABASE_TEMPLATE = """
CREATE TABLE {table_name} (
{table_info}
);
/*
Columns in {table_name} and 3 examples in each column:
{values};
*/
""".strip()


def format_db(db: Dict[str, Any], db_path: str, sample_number: int = 3) -> str:
    def sample_values(db_path: str, table_name: str, column_names: List[str], number: int) -> Dict[str, List[str]]:
        """
        Load a database from the given path, sample a number of examples from the specified table,
        and return the data in a dictionary format where the order of keys follows the order of column names.
        This function also handles column names with spaces by enclosing them in double quotes.

        :param db_path: Path to the SQLite database.
        :param table_name: Name of the table to sample from.
        :param column_names: List of column names to include in the output.
        :param number: Number of samples to retrieve.
        :return: A dictionary with keys as column names and values as lists of column data.
        """
        try:
            # Connect to the SQLite database
            conn = sqlite3.connect(db_path)
            cursor = conn.cursor()
            # Prepare the query to fetch the data, enclosing column names in double quotes if they contain spaces
            columns = ", ".join(
                [f'"{name}"' if ' ' in name else name for name in column_names])
            query = f"SELECT {columns} FROM {table_name} ORDER BY RANDOM() LIMIT {number}"
            # Execute the query and fetch the data
            cursor.execute(query)
            rows = cursor.fetchall()
            # Organize the data in the desired format
            data = {column: [] for column in column_names}
            for row in rows:
                for i, column in enumerate(column_names):
                    data[column].append(row[i])
            # Close the database connection
            conn.close()
            return data
        except Exception as e:
            return {}

    db_path = os.path.join(db_path, db["db_id"], f"{db['db_id']}.sqlite")
    databases: List[str] = []
    for table_idx, table_name in enumerate(db['table_names_original']):
        columns: List[str] = [c[1] + ' ' + ('int' if t == 'number' else t) for c, t in zip(
            db['column_names_original'], db['column_types']) if c[0] == table_idx]

        primary_keys: List[str] = [
            f"PRIMARY KEY ({db['column_names_original'][p][1]})" for p in db['primary_keys'] if db['column_names_original'][p][0] == table_idx]

        foreign_keys: List[str] = []
        for f in db['foreign_keys']:
            if db['column_names'][f[0]][0] == table_idx:
                column_0 = db['column_names_original'][f[0]]
                column_1 = db['column_names_original'][f[1]]
                foreign_keys.append(
                    f"FOREIGN KEY ({column_0[1]}) REFERENCES {db['table_names_original'][column_1[0]]}({column_1[1]})")

        corresponding_values = sample_values(db_path, table_name, [
                                             c[1] for c in db['column_names_original'] if c[0] == table_idx], sample_number)
        corresponding_values = {k: [f'"{x}"' if isinstance(x, str) else str(
            x) for x in v] for k, v in corresponding_values.items()}
        values: List[str] = [
            f"{k}: {', '.join(v)}" for k, v in corresponding_values.items()]

        databases.append(DATABASE_TEMPLATE.format(
            table_name=table_name,
            table_info=' ,\n'.join(columns + primary_keys + foreign_keys),
            values=';\n'.join(values)
        ))
        if sample_number == 0:
            databases[-1] = databases[-1].split('/*')[0].strip()
    return '\n'.join(databases).lower()


def parse_sql_safe(db_path: str, sql: str, timeout: float = 5) -> List[Any]:
    def parse_sql(db_path: str, sql: str) -> List[Any]:
        # 连接到 SQLite 数据库
        conn = sqlite3.connect(db_path)
        cursor = conn.cursor()
        try:
            # 执行第一个 SQL 查询
            cursor.execute(sql)
            result = cursor.fetchall()
            # 检查两个查询结果是否相同且不为空
            return result
        except sqlite3.Error as e:
            return []
        finally:
            # 关闭数据库连接
            conn.close()

    try:
        return func_timeout(timeout, parse_sql, args=(db_path, sql))
    except:
        return []


def check_sql_equivalence_safe(db_path: str, p_str: str, g_str: str, timeout: float = 5) -> bool:
    def check_sql_equivalence(db_path: str, p_str: str, g_str: str) -> bool:
        p_str = p_str.lower()
        g_str = g_str.lower()
        try:
            conn = sqlite3.connect(db_path)
            cursor = conn.cursor()
            cursor.execute(p_str)
            p_results = cursor.fetchall()
            p_columns = [desc[0] for desc in cursor.description]
            cursor.execute(g_str)
            g_results = cursor.fetchall()
            g_columns = [desc[0] for desc in cursor.description]
            cursor.close()
            conn.close()
        except:
            return False

        if len(p_results) != len(g_results) or len(p_columns) != len(g_columns):
            return False
        if set(p_columns) != set(g_columns):
            return False
        for p_row, g_row in zip(p_results, g_results):
            p_dict = dict(zip(p_columns, p_row))
            g_dict = dict(zip(g_columns, g_row))
            if p_dict != g_dict:
                return False
        return True

    try:
        return func_timeout(timeout, check_sql_equivalence, args=(db_path, p_str, g_str))
    except:
        return False


def fix_sql(sql: str) -> str:
    sql = sql.lower().strip()
    if not sql.startswith('select'):
        sql = f"SELECT {sql}"
    sql = sql.split(';')[0].split('/*')[0].strip().replace('\n', ' ')
    sql = re.sub(' +', ' ', sql)
    return sql
