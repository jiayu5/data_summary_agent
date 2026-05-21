# Data Summary Agent

基于 RAG + LangGraph 的数据分析系统，支持通过自然语言进行统计分析、数据可视化和 ML 预测。

## 技术栈
- Python + conda (rag)
- Neo4j Desktop — 知识图谱存储
- 小米大模型 mimo-v2.5-pro (OpenAI 兼容接口) — LLM
- ChromaDB + BGE — 向量索引
- LangGraph — 管道编排
- sklearn — ML 预测
- Plotly — 可视化

## 项目结构
```
data_summary_agent/
├── .env                           # 环境变量（不提交）
├── CLAUDE.md
├── README.md
├── requirements.txt
├── data/
│   ├── titanic_cleaned.csv        # 原始数据
│   ├── schema.json                # 图谱 Schema（自动生成）
│   ├── text2cypher_examples.json  # Text2Cypher 示例库
│   ├── chroma_db/                 # 向量索引存储
│   └── models/                    # ML 模型存储
└── scripts/
    ├── build/                     # 数据准备 + 索引构建
    │   ├── kg_builder.py          # CSV → Neo4j 知识图谱
    │   ├── schema_extractor.py    # Neo4j → schema.json
    │   └── vector_index.py        # 向量索引构建 + 检索
    ├── engine/                    # 生成 + ML 引擎
    │   ├── query_engine.py        # LLM 调用、Cypher 提取、可视化
    │   └── ml_engine.py           # sklearn 训练/预测工具
    ├── graph.py                   # LangGraph 管道入口
    └── train_model.py             # 独立训练脚本
```

## 常用命令
```bash
# 1. 构建知识图谱（需先启动 Neo4j）
cd scripts && python build/kg_builder.py

# 2. 提取 Schema
cd scripts && python build/schema_extractor.py

# 3. 构建向量索引
cd scripts && python build/vector_index.py

# 4. 训练 ML 模型
cd scripts && python train_model.py

# 5. 测试查询
cd scripts && python graph.py
```

所有脚本需从 `scripts/` 目录运行（因为使用相对 import）。

## Neo4j Desktop 设置
1. 下载安装 Neo4j Desktop (https://neo4j.com/download/)
2. 创建本地项目 → Add Database → 创建数据库
3. 设置密码为 `password123`（或同步修改 .env 中的 NEO4J_PASSWORD）
4. 启动数据库，状态变绿后即可运行脚本

## 环境变量 (.env)
- `NEO4J_URI`: bolt://localhost:7687
- `NEO4J_USER` / `NEO4J_PASSWORD`: Neo4j 认证
- `XIAOMI_API_KEY`: 小米大模型 API Key
- `XIAOMI_API_BASE`: API 端点
- `XIAOMI_MODEL`: 模型 ID (默认 mimo-v2.5-pro)

## 约定
- 数据文件放在 `data/` 目录
- `scripts/build/` — 数据准备和索引构建脚本
- `scripts/engine/` — 可复用的底层工具函数
- `scripts/graph.py` — LangGraph 管道，对外入口
- `.env` 不提交到 git
- 所有脚本从 `scripts/` 目录运行（相对 import）
