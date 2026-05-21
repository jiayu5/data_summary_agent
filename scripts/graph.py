from typing import TypedDict

from langgraph.graph import END, StateGraph

from engine.query_engine import (
    load_schema,
    build_system_prompt,
    build_user_prompt,
    extract_cypher,
    call_llm,
    execute_cypher,
    classify_result,
    generate_viz_code,
    execute_viz_code,
)
from build.vector_index import search_similar
from engine.ml_engine import (
    load_model_and_preprocessor,
    extract_features_from_text,
    predict_single,
)


# --- State Definition ---

class QueryState(TypedDict):
    question: str
    examples: list[dict]
    schema: str
    system_prompt: str
    cypher: str
    results: list[dict]
    error: str
    retry_count: int
    need_viz: bool
    viz_code: str
    chart_json: str
    intent: str
    prediction: dict


# --- Nodes ---

def retrieve(state: QueryState) -> dict:
    """Retrieve similar few-shot examples and load schema."""
    examples = search_similar(state["question"], top_k=3)
    schema = load_schema()
    system_prompt = build_system_prompt(schema)
    return {
        "examples": examples,
        "schema": schema,
        "system_prompt": system_prompt,
        "retry_count": 0,
        "error": "",
        "prediction": {},
    }


def intent_router(state: QueryState) -> dict:
    """Classify user intent: data query or ML prediction."""
    system = (
        "判断用户意图。只回答一个词：\n"
        "- 如果用户想查询/统计数据或画图，回答 query\n"
        "- 如果用户想预测乘客是否存活，回答 predict"
    )
    response = call_llm(system, state["question"]).strip().lower()
    if "predict" in response:
        return {"intent": "predict"}
    return {"intent": "query"}


def predict_node(state: QueryState) -> dict:
    """Extract features from natural language and predict survival."""
    # Extract passenger features from question
    features = extract_features_from_text(state["question"])
    if not features:
        return {"error": "无法从描述中提取乘客特征", "prediction": {}}

    # Load model and predict
    try:
        model, preprocessor, feature_names = load_model_and_preprocessor()
        result = predict_single(model, preprocessor, feature_names, features)
        return {"prediction": result}
    except FileNotFoundError:
        return {"error": "模型未训练，请先运行 train_model.py", "prediction": {}}
    except Exception as e:
        return {"error": f"预测失败: {e}", "prediction": {}}


def generate(state: QueryState) -> dict:
    """Generate Cypher from natural language question."""
    user_prompt = build_user_prompt(state["question"], state["examples"])

    # If retrying, append error context so LLM can fix the query
    if state.get("error"):
        user_prompt += (
            f"\n\n上一次生成的 Cypher 执行出错：\n"
            f"Cypher: {state['cypher']}\n"
            f"错误: {state['error']}\n"
            f"请修正后重新生成。"
        )

    raw = call_llm(state["system_prompt"], user_prompt)
    cypher = extract_cypher(raw)
    return {"cypher": cypher, "error": ""}


def execute(state: QueryState) -> dict:
    """Execute generated Cypher against Neo4j."""
    cypher = state["cypher"]
    if not cypher:
        return {"results": None, "error": "LLM 未返回有效的 Cypher 语句"}
    try:
        results = execute_cypher(cypher)
        return {"results": results, "error": ""}
    except Exception as e:
        return {"results": None, "error": str(e)}


def increment_retry(state: QueryState) -> dict:
    """Increment retry counter before regenerating."""
    return {"retry_count": state.get("retry_count", 0) + 1}


def classify_result_node(state: QueryState) -> dict:
    """Decide whether to visualize the query results."""
    if state.get("error") or not state.get("results"):
        return {"need_viz": False}
    need_viz = classify_result(state["question"], state["results"])
    return {"need_viz": need_viz}


def generate_viz(state: QueryState) -> dict:
    """Generate Plotly visualization code."""
    code = generate_viz_code(state["question"], state["results"], state["cypher"])
    return {"viz_code": code}


def execute_viz(state: QueryState) -> dict:
    """Execute the visualization code and capture the chart JSON."""
    try:
        chart_json = execute_viz_code(state["viz_code"], state["results"])
        return {"chart_json": chart_json}
    except Exception as e:
        return {"chart_json": "", "error": f"可视化执行失败: {e}"}


# --- Conditional Edge ---

def route_after_execute(state: QueryState) -> str:
    """Decide whether to retry or continue to classification."""
    if state.get("error") and state.get("retry_count", 0) < 2:
        return "retry"
    return "classify"


def route_after_intent(state: QueryState) -> str:
    """Route to query pipeline or prediction pipeline."""
    if state.get("intent") == "predict":
        return "predict"
    return "query"


def route_after_classify(state: QueryState) -> str:
    """Decide whether to visualize or end."""
    if state.get("need_viz"):
        return "visualize"
    return "end"


# --- Build Graph ---

def build_graph() -> StateGraph:
    graph = StateGraph(QueryState)

    # Add nodes
    graph.add_node("retrieve", retrieve)
    graph.add_node("intent_router", intent_router)
    graph.add_node("generate", generate)
    graph.add_node("execute", execute)
    graph.add_node("increment_retry", increment_retry)
    graph.add_node("classify_result_node", classify_result_node)
    graph.add_node("generate_viz", generate_viz)
    graph.add_node("execute_viz", execute_viz)
    graph.add_node("predict_node", predict_node)

    # Edges
    graph.set_entry_point("intent_router")
    graph.add_edge("retrieve", "generate")
    graph.add_edge("generate", "execute")
    graph.add_edge("increment_retry", "generate")
    graph.add_edge("generate_viz", "execute_viz")
    graph.add_edge("execute_viz", END)
    graph.add_edge("predict_node", END)

    # Conditional: intent_router → retrieve (query) or predict
    graph.add_conditional_edges(
        "intent_router",
        route_after_intent,
        {"query": "retrieve", "predict": "predict_node"},
    )

    # Conditional: execute → retry or classify
    graph.add_conditional_edges(
        "execute",
        route_after_execute,
        {"retry": "increment_retry", "classify": "classify_result_node"},
    )

    # Conditional: classify → visualize or end
    graph.add_conditional_edges(
        "classify_result_node",
        route_after_classify,
        {"visualize": "generate_viz", "end": END},
    )

    return graph.compile()


# Compiled graph instance
_app = build_graph()


# --- Public API ---

def query(question: str) -> dict:
    """
    Run a natural language question through the LangGraph pipeline.

    Returns:
        {
            "question": str,
            "cypher": str,
            "results": list[dict] or None,
            "error": str or None,
            "retry_count": int
        }
    """
    initial_state: QueryState = {
        "question": question,
        "examples": [],
        "schema": "",
        "system_prompt": "",
        "cypher": "",
        "results": None,
        "error": "",
        "retry_count": 0,
        "need_viz": False,
        "viz_code": "",
        "chart_json": "",
        "intent": "",
        "prediction": {},
    }

    final_state = _app.invoke(initial_state)

    return {
        "question": final_state["question"],
        "intent": final_state.get("intent", ""),
        "cypher": final_state.get("cypher", ""),
        "results": final_state.get("results"),
        "error": final_state.get("error") or None,
        "retry_count": final_state.get("retry_count", 0),
        "prediction": final_state.get("prediction") or None,
        "need_viz": final_state.get("need_viz", False),
        "chart_json": final_state.get("chart_json") or None,
    }


if __name__ == "__main__":
    import plotly.io as pio

    test_questions = [
        "各舱位等级的平均票价是多少？",
        "女性乘客的存活率是多少？",
        "画出survived乘客在不同年龄的分布",
        "各登船港口的乘客数量柱状图",
        "预测一个30岁女性头等舱乘客能否存活",
    ]
    for q in test_questions:
        print(f"\n{'='*60}")
        print(f"Question: {q}")
        result = query(q)
        print(f"Intent: {result['intent']}")
        if result["intent"] == "predict":
            if result.get("prediction"):
                print(f"Prediction: {result['prediction']}")
            if result.get("error"):
                print(f"Error: {result['error']}")
        else:
            print(f"Cypher: {result['cypher']}")
            if result["error"]:
                print(f"Error: {result['error']}")
                print(f"Retries: {result['retry_count']}")
            else:
                print(f"Results: {result['results'][:2]}...")
                print(f"Need Viz: {result['need_viz']}")
                if result["chart_json"]:
                    fig = pio.from_json(result["chart_json"])
                    fig.show()
                    print("Chart opened in browser.")
