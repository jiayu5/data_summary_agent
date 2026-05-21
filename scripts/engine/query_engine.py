import json
import os
import re
import base64

import numpy as np

from dotenv import load_dotenv
from openai import OpenAI
from neo4j import GraphDatabase

from build.vector_index import search_similar

load_dotenv()

NEO4J_URI = os.getenv("NEO4J_URI", "bolt://localhost:7687")
NEO4J_USER = os.getenv("NEO4J_USER", "neo4j")
NEO4J_PASSWORD = os.getenv("NEO4J_PASSWORD", "password123")
XIAOMI_API_KEY = os.getenv("XIAOMI_API_KEY")
XIAOMI_API_BASE = os.getenv("XIAOMI_API_BASE")
XIAOMI_MODEL = os.getenv("XIAOMI_MODEL", "mimo-v2.5-pro")

SCHEMA_PATH = os.path.join(os.path.dirname(__file__), "..", "..", "data", "schema.json")


def load_schema() -> str:
    with open(SCHEMA_PATH, "r", encoding="utf-8") as f:
        return json.dumps(json.load(f), ensure_ascii=False, indent=2)


def build_system_prompt(schema_json: str) -> str:
    return (
        "你是一个 Neo4j Cypher 查询专家。根据以下知识图谱 schema 和示例，"
        "将用户的自然语言问题转换为 Cypher 查询语句。\n\n"
        f"Schema:\n{schema_json}\n\n"
        "规则：\n"
        "1. 只返回 Cypher 语句，不要解释，不要 markdown 代码块\n"
        "2. 不要使用不存在的属性或关系\n"
        "3. 对数值做聚合时注意处理 NULL 值（用 WHERE x IS NOT NULL 过滤）\n"
        "4. 计算百分比/比率时，先用 WITH 分步计算总数\n"
        "5. 结果列使用有意义的别名"
    )


def build_user_prompt(question: str, examples: list[dict]) -> str:
    parts = []
    for i, ex in enumerate(examples, 1):
        parts.append(f"示例{i}:\n问题：{ex['question']}\nCypher：{ex['cypher']}")
    parts.append(f"问题：{question}\nCypher：")
    return "\n\n".join(parts)


def extract_cypher(text: str) -> str:
    """Extract Cypher query from LLM response, stripping markdown fences and prose."""
    # Remove markdown code blocks
    text = re.sub(r"```(?:cypher|sql)?\s*\n?", "", text).strip()
    # Remove common prefixes the LLM might add
    text = re.sub(r"^(Cypher[:：]?\s*|查询[:：]?\s*|答案[:：]?\s*)", "", text, flags=re.IGNORECASE).strip()
    # Find the first statement starting with a Cypher keyword
    match = re.search(r"(MATCH|CALL|CREATE|WITH|OPTIONAL)\b.*", text, re.DOTALL | re.IGNORECASE)
    if match:
        return match.group(0).strip().rstrip(";")
    # If no keyword found, try returning the whole cleaned text
    return text.strip().rstrip(";")


def call_llm(system_prompt: str, user_prompt: str) -> str:
    client = OpenAI(api_key=XIAOMI_API_KEY, base_url=XIAOMI_API_BASE)
    response = client.chat.completions.create(
        model=XIAOMI_MODEL,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ],
        temperature=0,
        max_tokens=4096,
    )
    return response.choices[0].message.content


def execute_cypher(cypher: str) -> list[dict]:
    driver = GraphDatabase.driver(NEO4J_URI, auth=(NEO4J_USER, NEO4J_PASSWORD))
    try:
        with driver.session() as session:
            result = session.run(cypher)
            keys = result.keys()
            records = []
            for record in result:
                records.append({k: record[k] for k in keys})
            return records
    finally:
        driver.close()


def classify_result(question: str, results: list[dict]) -> bool:
    """Ask LLM whether the query results should be visualized."""
    system = (
        "你是数据分析助手。根据用户的问题和查询结果，判断是否适合用图表展示。\n"
        "只回答 yes 或 no。"
    )
    user = f"用户问题：{question}\n查询结果：{json.dumps(results, ensure_ascii=False, default=str)}\n\n是否适合画图？"
    response = call_llm(system, user)
    return response.strip().lower().startswith("yes")


def build_viz_prompt(question: str, results: list[dict], cypher: str) -> str:
    return (
        "你是 Plotly 可视化专家。根据查询结果生成 Python Plotly 代码。\n\n"
        f"用户问题：{question}\n"
        f"Cypher：{cypher}\n"
        f"查询结果：{json.dumps(results, ensure_ascii=False, default=str)}\n\n"
        "可用变量：\n"
        "- data: 查询结果 (list[dict])\n"
        "- pd: pandas\n"
        "- go: plotly.graph_objects\n"
        "- px: plotly.express\n\n"
        "要求：\n"
        "1. 直接使用 data 变量，不需要再查 Neo4j\n"
        "2. 用 Plotly 生成图表，赋值给 fig 变量\n"
        "3. 设置图表标题和轴标签\n"
        "4. 只返回 Python 代码，不要解释，不要 markdown 代码块"
    )


def extract_code(text: str) -> str:
    """Extract Python code from LLM response."""
    text = re.sub(r"```(?:python)?\s*\n?", "", text).strip()
    return text


def generate_viz_code(question: str, results: list[dict], cypher: str) -> str:
    """Generate Plotly visualization code from LLM."""
    prompt = build_viz_prompt(question, results, cypher)
    raw = call_llm(
        "你只返回可执行的 Python 代码，不要任何解释文字。",
        prompt,
    )
    return extract_code(raw)


def execute_viz_code(code: str, results: list[dict]) -> str:
    """Execute Plotly code and return the figure as standard JSON."""
    import pandas as pd
    import plotly.graph_objects as go
    import plotly.express as px

    exec_globals = {
        "data": results,
        "pd": pd,
        "go": go,
        "px": px,
        "fig": None,
        "__builtins__": __builtins__,
    }
    exec(code, exec_globals)
    fig = exec_globals.get("fig")
    if fig is None:
        raise ValueError("代码未生成 fig 变量")

    return _fig_to_standard_json(fig)


def _decode_plotly_binary(obj):
    """Recursively decode Plotly binary format and numpy arrays to standard Python types."""
    if isinstance(obj, dict):
        if "dtype" in obj and "bdata" in obj:
            raw = base64.b64decode(obj["bdata"])
            dtype_map = {"f8": "<f8", "f4": "<f4", "i4": "<i4", "i8": "<i8", "u4": "<u4"}
            np_dtype = dtype_map.get(obj["dtype"], f"<{obj['dtype']}")
            return np.frombuffer(raw, dtype=np_dtype).tolist()
        return {k: _decode_plotly_binary(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [_decode_plotly_binary(item) for item in obj]
    elif isinstance(obj, np.ndarray):
        return obj.tolist()
    elif isinstance(obj, (np.integer,)):
        return int(obj)
    elif isinstance(obj, (np.floating,)):
        return float(obj)
    return obj


def _fig_to_standard_json(fig) -> str:
    """Convert a Plotly figure to standard JSON, decoding any binary format."""
    d = fig.to_plotly_json()
    decoded = _decode_plotly_binary(d)
    return json.dumps(decoded, ensure_ascii=False)


def query(question: str, top_k: int = 3) -> dict:
    """
    End-to-end natural language to Cypher query.

    Returns:
        {
            "question": str,
            "cypher": str,
            "results": list[dict] or None,
            "error": str or None
        }
    """
    output = {"question": question, "cypher": None, "results": None, "error": None}

    try:
        # Step 1: Retrieve similar examples
        examples = search_similar(question, top_k=top_k)

        # Step 2: Build prompts
        schema_json = load_schema()
        system_prompt = build_system_prompt(schema_json)
        user_prompt = build_user_prompt(question, examples)

        # Step 3: Generate Cypher
        raw_response = call_llm(system_prompt, user_prompt)
        cypher = extract_cypher(raw_response)
        output["cypher"] = cypher

        # Step 4: Execute
        output["results"] = execute_cypher(cypher)

    except Exception as e:
        output["error"] = str(e)

    return output


if __name__ == "__main__":
    test_questions = [
        "各舱位等级的平均票价是多少？",
        "女性乘客的存活率是多少？",
        "票价最高的乘客是谁？",
        "各登船港口的乘客数量是多少？",
    ]
    for q in test_questions:
        print(f"\n{'='*60}")
        print(f"Question: {q}")
        result = query(q)
        print(f"Cypher: {result['cypher']}")
        if result["error"]:
            print(f"Error: {result['error']}")
        else:
            print(f"Results: {result['results']}")
