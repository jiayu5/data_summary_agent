import json
import os

import chromadb
from sentence_transformers import SentenceTransformer

DATA_DIR = os.path.join(os.path.dirname(__file__), "..", "..", "data")
EXAMPLES_PATH = os.path.join(DATA_DIR, "text2cypher_examples.json")
CHROMA_DIR = os.path.join(DATA_DIR, "chroma_db")
COLLECTION_NAME = "text2cypher"
MODEL_NAME = "BAAI/bge-base-en-v1.5"


def build_index():
    with open(EXAMPLES_PATH, "r", encoding="utf-8") as f:
        examples = json.load(f)
    print(f"Loaded {len(examples)} examples.")

    model = SentenceTransformer(MODEL_NAME)
    print(f"Model {MODEL_NAME} loaded.")

    questions = [ex["question"] for ex in examples]
    embeddings = model.encode(questions, show_progress_bar=True).tolist()

    client = chromadb.PersistentClient(path=CHROMA_DIR)
    # Delete existing collection if re-building
    try:
        client.delete_collection(COLLECTION_NAME)
    except Exception:
        pass

    collection = client.create_collection(
        name=COLLECTION_NAME,
        metadata={"hnsw:space": "cosine"},
    )

    ids = [f"ex_{i}" for i in range(len(examples))]
    documents = questions
    metadatas = [
        {"cypher": ex["cypher"], "intent": ex["intent"], "question": ex["question"]}
        for ex in examples
    ]

    collection.add(ids=ids, embeddings=embeddings, documents=documents, metadatas=metadatas)
    print(f"Indexed {len(examples)} examples into ChromaDB at {CHROMA_DIR}")


def search_similar(question: str, top_k: int = 3):
    """Search for similar Text2Cypher examples. Use this for retrieval at query time."""
    model = SentenceTransformer(MODEL_NAME)
    embedding = model.encode([question]).tolist()

    client = chromadb.PersistentClient(path=CHROMA_DIR)
    collection = client.get_collection(COLLECTION_NAME)

    results = collection.query(query_embeddings=embedding, n_results=top_k)

    hits = []
    for i in range(len(results["documents"][0])):
        hits.append({
            "question": results["metadatas"][0][i]["question"],
            "cypher": results["metadatas"][0][i]["cypher"],
            "intent": results["metadatas"][0][i]["intent"],
            "distance": results["distances"][0][i],
        })
    return hits


if __name__ == "__main__":
    build_index()

    print("\n--- Test retrieval ---")
    test_queries = [
        "女性乘客的存活率是多少？",
        "头等舱有多少人？",
        "最贵的船票多少钱？",
    ]
    for q in test_queries:
        print(f"\nQuery: {q}")
        results = search_similar(q, top_k=2)
        for r in results:
            print(f"  [{r['distance']:.3f}] {r['question']}")
            print(f"         -> {r['cypher']}")
