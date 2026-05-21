import os
import pandas as pd
from dotenv import load_dotenv
from neo4j import GraphDatabase

load_dotenv()

NEO4J_URI = os.getenv("NEO4J_URI", "bolt://localhost:7687")
NEO4J_USER = os.getenv("NEO4J_USER", "neo4j")
NEO4J_PASSWORD = os.getenv("NEO4J_PASSWORD", "password123")
CSV_PATH = os.path.join(os.path.dirname(__file__), "..", "..", "data", "titanic_cleaned.csv")

BATCH_SIZE = 200

PCLASS_NAMES = {1: "First", 2: "Second", 3: "Third"}
PORT_NAMES = {"S": "Southampton", "C": "Cherbourg", "Q": "Queenstown"}


def create_constraints(driver):
    with driver.session() as session:
        session.run(
            "CREATE CONSTRAINT passenger_id IF NOT EXISTS "
            "FOR (p:Passenger) REQUIRE p.passengerId IS UNIQUE"
        )
        session.run(
            "CREATE CONSTRAINT pclass_id IF NOT EXISTS "
            "FOR (c:Pclass) REQUIRE c.classId IS UNIQUE"
        )
        session.run(
            "CREATE CONSTRAINT port_code IF NOT EXISTS "
            "FOR (e:Embarked) REQUIRE e.portCode IS UNIQUE"
        )
    print("Constraints created.")


def create_dimension_nodes(driver):
    with driver.session() as session:
        # Pclass nodes
        session.run(
            "UNWIND $rows AS row "
            "MERGE (c:Pclass {classId: row.classId}) "
            "SET c.className = row.className",
            rows=[
                {"classId": k, "className": v} for k, v in PCLASS_NAMES.items()
            ],
        )
        # Embarked nodes
        session.run(
            "UNWIND $rows AS row "
            "MERGE (e:Embarked {portCode: row.portCode}) "
            "SET e.portName = row.portName",
            rows=[
                {"portCode": k, "portName": v} for k, v in PORT_NAMES.items()
            ],
        )
    print("Dimension nodes created (3 Pclass, 3 Embarked).")


def create_passenger_nodes(driver, df):
    total = 0
    with driver.session() as session:
        for start in range(0, len(df), BATCH_SIZE):
            batch = df.iloc[start : start + BATCH_SIZE]
            rows = []
            for _, r in batch.iterrows():
                embarked = r["Embarked"] if pd.notna(r["Embarked"]) else None
                rows.append(
                    {
                        "passengerId": int(r["PassengerId"]),
                        "name": r["Name"],
                        "age": float(r["Age"]) if pd.notna(r["Age"]) else None,
                        "sex": r["Sex"],
                        "survived": int(r["Survived"]),
                        "sibSp": int(r["SibSp"]),
                        "parch": int(r["Parch"]),
                        "ticket": r["Ticket"],
                        "fare": float(r["Fare"]) if pd.notna(r["Fare"]) else None,
                        "cabin": r["Cabin"] if pd.notna(r["Cabin"]) else None,
                        "pclass": int(r["Pclass"]),
                        "embarked": embarked,
                    }
                )
            session.run(
                "UNWIND $rows AS row "
                "MERGE (p:Passenger {passengerId: row.passengerId}) "
                "SET p.name = row.name, p.age = row.age, p.sex = row.sex, "
                "    p.survived = row.survived, p.sibSp = row.sibSp, "
                "    p.parch = row.parch, p.ticket = row.ticket, "
                "    p.fare = row.fare, p.cabin = row.cabin "
                "WITH p, row "
                "MATCH (c:Pclass {classId: row.pclass}) "
                "MERGE (p)-[:BELONGS_TO_CLASS]->(c) "
                "WITH p, row "
                "WHERE row.embarked IS NOT NULL "
                "MATCH (e:Embarked {portCode: row.embarked}) "
                "MERGE (p)-[:EMBARKED_AT]->(e)",
                rows=rows,
            )
            total += len(rows)
            print(f"  Created {total}/{len(df)} passengers...")
    print(f"All {total} passenger nodes created.")


def print_stats(driver):
    with driver.session() as session:
        result = session.run(
            "MATCH (n) RETURN labels(n)[0] AS label, count(n) AS cnt "
            "ORDER BY label"
        )
        print("\n--- Graph Stats ---")
        for record in result:
            print(f"  {record['label']}: {record['cnt']} nodes")

        result = session.run(
            "MATCH ()-[r]->() RETURN type(r) AS rel, count(r) AS cnt "
            "ORDER BY rel"
        )
        for record in result:
            print(f"  {record['rel']}: {record['cnt']} relationships")


def main():
    print(f"Reading CSV: {CSV_PATH}")
    df = pd.read_csv(CSV_PATH)
    print(f"Loaded {len(df)} rows, {len(df.columns)} columns.")

    driver = GraphDatabase.driver(NEO4J_URI, auth=(NEO4J_USER, NEO4J_PASSWORD))
    try:
        driver.verify_connectivity()
        print("Connected to Neo4j.")

        # Clear existing data
        with driver.session() as session:
            session.run("MATCH (n) DETACH DELETE n")
        print("Cleared existing graph data.")

        create_constraints(driver)
        create_dimension_nodes(driver)
        create_passenger_nodes(driver, df)
        print_stats(driver)
    finally:
        driver.close()
    print("\nDone!")


if __name__ == "__main__":
    main()
