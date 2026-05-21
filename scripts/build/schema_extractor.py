import json
import os
from dotenv import load_dotenv
from neo4j import GraphDatabase

load_dotenv()

NEO4J_URI = os.getenv("NEO4J_URI", "bolt://localhost:7687")
NEO4J_USER = os.getenv("NEO4J_USER", "neo4j")
NEO4J_PASSWORD = os.getenv("NEO4J_PASSWORD", "password123")
OUTPUT_PATH = os.path.join(os.path.dirname(__file__), "..", "..", "data", "schema.json")

# Neo4j type mapping
TYPE_MAP = {
    "String": "STRING",
    "Long": "INTEGER",
    "Integer": "INTEGER",
    "Float": "FLOAT",
    "Double": "FLOAT",
    "Boolean": "BOOLEAN",
}


def extract_node_properties(driver):
    """Extract labels and property types by sampling nodes."""
    nodes = {}
    with driver.session() as session:
        result = session.run("CALL db.labels()")
        labels = [record["label"] for record in result]

        for label in labels:
            result = session.run(
                f"MATCH (n:{label}) RETURN properties(n) AS props LIMIT 100"
            )
            prop_types = {}
            for record in result:
                props = record["props"]
                for key, value in props.items():
                    neo4j_type = type(value).__name__
                    mapped = TYPE_MAP.get(neo4j_type, neo4j_type.upper())
                    if key not in prop_types:
                        prop_types[key] = mapped

            nodes[label] = {"properties": prop_types}
    return nodes


def extract_relationships(driver):
    """Extract relationship types and their start/end node labels."""
    relationships = []
    with driver.session() as session:
        result = session.run("CALL db.relationshipTypes()")
        rel_types = [record["relationshipType"] for record in result]

        for rel_type in rel_types:
            result = session.run(
                f"MATCH (a)-[r:{rel_type}]->(b) "
                "RETURN DISTINCT labels(a)[0] AS from, labels(b)[0] AS to LIMIT 1"
            )
            for record in result:
                relationships.append({
                    "type": rel_type,
                    "from": record["from"],
                    "to": record["to"],
                })
    return relationships


def main():
    driver = GraphDatabase.driver(NEO4J_URI, auth=(NEO4J_USER, NEO4J_PASSWORD))
    try:
        driver.verify_connectivity()
        print("Connected to Neo4j.")

        nodes = extract_node_properties(driver)
        relationships = extract_relationships(driver)

        schema = {
            "description": "Titanic 乘客知识图谱，包含乘客、舱位等级、登船港口三类节点",
            "nodes": nodes,
            "relationships": relationships,
        }

        os.makedirs(os.path.dirname(OUTPUT_PATH), exist_ok=True)
        with open(OUTPUT_PATH, "w", encoding="utf-8") as f:
            json.dump(schema, f, ensure_ascii=False, indent=2)

        print(f"Schema saved to {OUTPUT_PATH}")
        print(f"  Nodes: {list(nodes.keys())}")
        print(f"  Relationships: {[r['type'] for r in relationships]}")

        # Print summary
        for label, info in nodes.items():
            print(f"\n  {label}:")
            for prop, ptype in info["properties"].items():
                print(f"    {prop}: {ptype}")

    finally:
        driver.close()


if __name__ == "__main__":
    main()
