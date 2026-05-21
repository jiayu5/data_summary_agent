# Data Summary Agent

基于 RAG + LangGraph 的数据分析系统，支持通过自然语言进行查询、统计分析、数据可视化和 ML 预测。

## 技术栈

| 组件 | 技术 |
|------|------|
| 知识图谱 | Neo4j Desktop |
| 向量数据库 | ChromaDB |
| Embedding | BAAI/bge-base-en-v1.5 |
| LLM | 小米 mimo-v2.5-pro（OpenAI 兼容接口） |
| 编排框架 | LangGraph |
| ML | scikit-learn (RandomForest) |
| 可视化 | Plotly |
| 语言 | Python 3 + conda (rag) |

## 架构设计

### RAG Pipeline 总览

```
数据准备           索引构建            检索+生成+可视化       ML 预测
─────────        ──────────         ─────────────────     ──────────
CSV → Neo4j      Schema 提取         LangGraph 管道        训练脚本
知识图谱构建      Text2Cypher 示例     intent → retrieve     train_model.py
                  向量索引            → generate → viz      自然语言预测
```

### LangGraph 管道流程

```
START → intent_router
            │
  ┌─────────┴─────────┐
  ▼                   ▼
retrieve         predict_node → END
  │
  ▼
generate
  │
  ▼
execute ─── error + retry < 2 ──→ increment_retry → generate
  │
  ▼ success
classify_result
  │
  ├── no → END
  │
  └── yes → generate_viz → execute_viz → END
```

**节点说明：**

| 节点 | 职责 |
|------|------|
| `intent_router` | 入口。LLM 判断意图：数据查询 (query) 或 ML 预测 (predict) |
| `retrieve` | query 分支：从 ChromaDB 检索相似的 few-shot 示例，加载 Schema |
| `generate` | 拼装 Prompt（Schema + 示例 + 问题），调 LLM 生成 Cypher |
| `execute` | 在 Neo4j 执行生成的 Cypher |
| `increment_retry` | 递增重试计数，将错误信息追加到 prompt 让 LLM 修正 |
| `classify_result` | LLM 判断查询结果是否适合可视化 |
| `generate_viz` / `execute_viz` | 生成并执行 Plotly 可视化代码 |
| `predict_node` | predict 分支：从自然语言提取乘客特征，加载模型进行预测 |

**状态定义：**

```python
class QueryState(TypedDict):
    question: str        # 用户原始问题
    examples: list[dict] # 检索到的 few-shot 示例
    schema: str          # 图谱 schema JSON
    system_prompt: str   # 系统 prompt
    cypher: str          # 生成的 Cypher
    results: list[dict]  # Neo4j 执行结果
    error: str           # 错误信息
    retry_count: int     # 重试次数
    need_viz: bool       # 是否需要可视化
    viz_code: str        # 生成的 Plotly 代码
    chart_json: str      # Plotly Figure JSON
    intent: str          # "query" 或 "predict"
    prediction: dict     # ML 预测结果
```

## 项目结构

```
data_summary_agent/
├── .env                           # 环境变量（不提交）
├── CLAUDE.md                      # 项目约定
├── README.md
├── requirements.txt               # Python 依赖
├── data/
│   ├── titanic_cleaned.csv        # 原始数据
│   ├── schema.json                # 图谱 Schema（自动生成）
│   ├── text2cypher_examples.json  # Text2Cypher few-shot 示例库
│   ├── chroma_db/                 # ChromaDB 持久化存储
│   └── models/                    # sklearn 模型存储
│       ├── model.joblib
│       └── meta.json
└── scripts/
    ├── build/                     # 数据准备 + 索引构建
    │   ├── kg_builder.py          # CSV → Neo4j 知识图谱
    │   ├── schema_extractor.py    # Neo4j 动态提取 Schema
    │   └── vector_index.py        # ChromaDB 向量索引构建 + 检索
    ├── engine/                    # 底层工具函数
    │   ├── query_engine.py        # LLM 调用、Cypher 提取、可视化执行
    │   └── ml_engine.py           # sklearn 数据预处理、训练、预测
    ├── graph.py                   # LangGraph 管道定义 + 对外入口
    └── train_model.py             # 独立训练脚本
```

### 模块职责

| 模块 | 职责 |
|------|------|
| `build/kg_builder.py` | 从 CSV 构建 Neo4j 知识图谱（Passenger、Pclass、Embarked 节点） |
| `build/schema_extractor.py` | 从 Neo4j 动态提取 schema.json |
| `build/vector_index.py` | BGE 嵌入 + ChromaDB，提供语义检索接口 |
| `engine/query_engine.py` | 底层工具：LLM 调用、Cypher 提取、Neo4j 执行、Plotly 可视化 |
| `engine/ml_engine.py` | ML 工具：数据预处理、模型训练/保存/加载、特征提取、预测 |
| `graph.py` | LangGraph 管道，对外只暴露 `query()` 函数 |
| `train_model.py` | 独立脚本，训练 RandomForest 并保存到 data/models/ |

### 模块依赖关系

```
graph.py
  ├── engine.query_engine  (call_llm, execute_cypher, ...)
  ├── build.vector_index   (search_similar)
  └── engine.ml_engine     (load_model, extract_features, predict)

engine.query_engine
  └── build.vector_index   (search_similar)

engine.ml_engine
  └── engine.query_engine  (call_llm)

train_model.py
  └── engine.ml_engine     (load_and_preprocess, train_model, save_model)
```

## 快速开始

### 1. 环境准备

```bash
# 安装 Neo4j Desktop 并启动数据库
# 下载: https://neo4j.com/download/

# 安装依赖
conda activate rag
pip install -r requirements.txt
```

### 2. 构建知识图谱

```bash
cd scripts
python build/kg_builder.py
```

### 3. 构建索引

```bash
cd scripts

# 提取 Schema
python build/schema_extractor.py

# 构建向量索引（首次会下载 BGE 模型，约 400MB）
python build/vector_index.py
```

### 4. 训练 ML 模型

```bash
cd scripts
python train_model.py
# → Accuracy: 0.81, AUC: 0.89
# → 保存到 data/models/
```

### 5. 查询和预测

```python
# 需从 scripts/ 目录运行
from graph import query

# 数据查询
result = query("各舱位等级的平均票价是多少？")
print(result["cypher"])   # 生成的 Cypher
print(result["results"])  # Neo4j 查询结果

# 数据查询 + 可视化
result = query("各登船港口的乘客数量柱状图")
# → result["chart_json"] 包含 Plotly 图表 JSON

# ML 预测
result = query("预测一个30岁女性头等舱乘客能否存活")
print(result["prediction"])  # {"survived": 1, "probability": 0.87}
```

## 数据流示例

**数据查询：**

```
用户: "女性乘客的存活率是多少？"
  → intent_router: "query"
  → retrieve: 检索到相似示例（存活率查询）
  → generate: MATCH (p:Passenger {sex: 'female'}) ...
  → execute: [{"female_survival_rate": 74.2}]
  → classify: 不适合画图 → END
```

**ML 预测：**

```
用户: "预测一个30岁女性头等舱乘客能否存活"
  → intent_router: "predict"
  → predict_node: 提取特征 {age: 30, sex: "female", pclass: 1}
                 → 加载模型 → 预测
  → {"survived": 1, "probability": 0.87, "label": "存活"}
```
