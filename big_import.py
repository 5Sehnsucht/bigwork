import json
from pymongo import MongoClient
from neo4j import GraphDatabase
from tqdm import tqdm

# ========================
# 1. 初始化数据库
# ========================

# MongoDB
mongo_client = MongoClient("mongodb://localhost:27017/")
mongo_db = mongo_client["wikimultihop_db"]
mongo_col = mongo_db["qa"]

# Neo4j
neo_driver = GraphDatabase.driver(
    "bolt://localhost:7687",
    auth=("neo4j", "12345678")  # 改成你的密码
)

# ========================
# 2. 读取数据
# ========================

with open("./data/test.json", "r") as f:
    data = json.load(f)


# ========================
# 3. MongoDB结构转换
# ========================

def transform_sample(sample):
    context = []
    for title, sentences in sample["context"]:
        context.append({
            "title": title,
            "sentences": sentences
        })

    supporting = []
    for title, sent_id in sample["supporting_facts"]:
        supporting.append({
            "title": title,
            "sent_id": sent_id
        })

    return {
        "_id": sample["_id"],
        "question": sample["question"],
        "answer": sample.get("answer", ""),
        "type": sample["type"],
        "context": context,
        "supporting_facts": supporting,
        "evidences": sample.get("evidences", []),
        "entity_ids": sample.get("entity_ids", "")
    }


# ========================
# 4. Neo4j写入
# ========================

def write_to_neo4j(tx, sample):

    qid = sample["_id"]
    question = sample["question"]

    # -------------------
    # Question节点
    # -------------------
    tx.run("""
    MERGE (q:Question {id:$qid})
    SET q.text = $question
    """, qid=qid, question=question)

    # -------------------
    # Document节点
    # -------------------
    for title, _ in sample["context"]:
        tx.run("""
        MERGE (d:Document {title:$title})
        """, title=title)

        tx.run("""
        MERGE (q:Question {id:$qid})
        MERGE (d:Document {title:$title})
        MERGE (q)-[:CONTEXT]->(d)
        """, qid=qid, title=title)

    # -------------------
    # Supporting Facts（推理路径）
    # -------------------
    for title, _ in sample["supporting_facts"]:
        tx.run("""
        MERGE (q:Question {id:$qid})
        MERGE (d:Document {title:$title})
        MERGE (q)-[:SUPPORTS]->(d)
        """, qid=qid, title=title)

    # -------------------
    # Evidence（三元组）
    # -------------------
    for subj, rel, obj in sample.get("evidences", []):

        tx.run("""
        MERGE (e1:Entity {name:$subj})
        MERGE (e2:Entity {name:$obj})
        MERGE (e1)-[:RELATION {type:$rel}]->(e2)
        """, subj=subj, obj=obj, rel=rel)

        # Question关联实体
        tx.run("""
        MERGE (q:Question {id:$qid})
        MERGE (e:Entity {name:$ent})
        MERGE (q)-[:MENTIONS]->(e)
        """, qid=qid, ent=subj)

        tx.run("""
        MERGE (q:Question {id:$qid})
        MERGE (e:Entity {name:$ent})
        MERGE (q)-[:MENTIONS]->(e)
        """, qid=qid, ent=obj)


# ========================
# 5. 主流程
# ========================

def main():

    for sample in tqdm(data):

        # -------------------
        # 1️⃣ MongoDB
        # -------------------
        doc = transform_sample(sample)

        try:
            mongo_col.insert_one(doc)
        except:
            pass

        # -------------------
        # 2️⃣ Neo4j
        # -------------------
        with neo_driver.session() as session:
            session.execute_write(write_to_neo4j, sample)


if __name__ == "__main__":
    main()