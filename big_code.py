import json
from pymongo import MongoClient
from neo4j import GraphDatabase
from tqdm import tqdm
from collections import Counter

# ============================================================
# 数据库连接配置
# ============================================================

# MongoDB 连接
MONGO_URI = "mongodb://localhost:27017/"
MONGO_DB = "wikimultihop_db"
MONGO_COLLECTION = "qa"

# Neo4j 连接
NEO4J_URI = "bolt://localhost:7687"
NEO4J_USER = "neo4j"
NEO4J_PASSWORD = "12345678"


# ============================================================
# 初始化数据库连接
# ============================================================

def init_mongodb():
    """初始化 MongoDB 连接"""
    client = MongoClient(MONGO_URI)
    db = client[MONGO_DB]
    collection = db[MONGO_COLLECTION]
    return client, collection


def init_neo4j():
    """初始化 Neo4j 连接"""
    driver = GraphDatabase.driver(NEO4J_URI, auth=(NEO4J_USER, NEO4J_PASSWORD))
    return driver


# ============================================================
# MongoDB CRUD 操作
# ============================================================

def mongo_insert_one(collection, doc):
    """MongoDB: 插入单个文档"""
    try:
        result = collection.insert_one(doc)
        print(f"  ✓ MongoDB: 插入文档成功，ID: {result.inserted_id}")
        return result.inserted_id
    except Exception as e:
        print(f"  ✗ MongoDB: 插入失败 - {e}")
        return None


def mongo_insert_many(collection, docs):
    """MongoDB: 批量插入文档"""
    try:
        result = collection.insert_many(docs)
        print(f"  ✓ MongoDB: 批量插入 {len(result.inserted_ids)} 条文档")
        return result.inserted_ids
    except Exception as e:
        print(f"  ✗ MongoDB: 批量插入失败 - {e}")
        return []


def mongo_find_one(collection, query):
    """MongoDB: 查询单个文档"""
    result = collection.find_one(query)
    if result:
        print(f"  ✓ MongoDB: 找到文档 {result.get('_id')}")
    else:
        print(f"  ✗ MongoDB: 未找到匹配文档")
    return result


def mongo_find_many(collection, query, limit=10):
    """MongoDB: 查询多个文档"""
    results = list(collection.find(query).limit(limit))
    print(f"  ✓ MongoDB: 找到 {len(results)} 条文档")
    return results


def mongo_update_one(collection, query, update_data):
    """MongoDB: 更新单个文档"""
    try:
        result = collection.update_one(query, {"$set": update_data})
        if result.modified_count > 0:
            print(f"  ✓ MongoDB: 更新成功，修改 {result.modified_count} 条文档")
        else:
            print(f"  ✗ MongoDB: 未找到匹配文档或无需更新")
        return result.modified_count
    except Exception as e:
        print(f"  ✗ MongoDB: 更新失败 - {e}")
        return 0


def mongo_update_many(collection, query, update_data):
    """MongoDB: 批量更新文档"""
    try:
        result = collection.update_many(query, {"$set": update_data})
        print(f"  ✓ MongoDB: 批量更新成功，修改 {result.modified_count} 条文档")
        return result.modified_count
    except Exception as e:
        print(f"  ✗ MongoDB: 批量更新失败 - {e}")
        return 0


def mongo_delete_one(collection, query):
    """MongoDB: 删除单个文档"""
    try:
        result = collection.delete_one(query)
        if result.deleted_count > 0:
            print(f"  ✓ MongoDB: 删除成功，删除 {result.deleted_count} 条文档")
        else:
            print(f"  ✗ MongoDB: 未找到匹配文档")
        return result.deleted_count
    except Exception as e:
        print(f"  ✗ MongoDB: 删除失败 - {e}")
        return 0


def mongo_delete_many(collection, query):
    """MongoDB: 批量删除文档"""
    try:
        result = collection.delete_many(query)
        print(f"  ✓ MongoDB: 批量删除成功，删除 {result.deleted_count} 条文档")
        return result.deleted_count
    except Exception as e:
        print(f"  ✗ MongoDB: 批量删除失败 - {e}")
        return 0


def mongo_count(collection, query=None):
    """MongoDB: 统计文档数量"""
    if query:
        count = collection.count_documents(query)
    else:
        count = collection.count_documents({})
    print(f"  ✓ MongoDB: 当前文档总数 {count}")
    return count


def mongo_aggregate(collection, pipeline):
    """MongoDB: 聚合查询"""
    try:
        results = list(collection.aggregate(pipeline))
        print(f"  ✓ MongoDB: 聚合查询返回 {len(results)} 条结果")
        return results
    except Exception as e:
        print(f"  ✗ MongoDB: 聚合查询失败 - {e}")
        return []


# ============================================================
# Neo4j CRUD 操作
# ============================================================

def neo4j_create_question(tx, qid, text):
    """Neo4j: 创建 Question 节点"""
    query = """
    MERGE (q:Question {id: $qid})
    SET q.text = $text, q.created_at = datetime()
    RETURN q.id AS id
    """
    result = tx.run(query, qid=qid, text=text)
    if result.single():
        print(f"  ✓ Neo4j: 创建 Question 节点成功: {qid}")
        return True
    return False


def neo4j_create_document(tx, title):
    """Neo4j: 创建 Document 节点"""
    query = """
    MERGE (d:Document {title: $title})
    SET d.created_at = datetime()
    RETURN d.title AS title
    """
    result = tx.run(query, title=title)
    if result.single():
        print(f"  ✓ Neo4j: 创建 Document 节点成功: {title}")
        return True
    return False


def neo4j_create_entity(tx, name):
    """Neo4j: 创建 Entity 节点"""
    query = """
    MERGE (e:Entity {name: $name})
    SET e.created_at = datetime()
    RETURN e.name AS name
    """
    result = tx.run(query, name=name)
    if result.single():
        print(f"  ✓ Neo4j: 创建 Entity 节点成功: {name}")
        return True
    return False


def neo4j_create_relationship(tx, from_label, from_id, from_prop, to_label, to_id, to_prop, rel_type, rel_props=None):
    """Neo4j: 创建关系"""
    query = f"""
    MATCH (a:{from_label} {{{from_prop}: $from_value}})
    MATCH (b:{to_label} {{{to_prop}: $to_value}})
    MERGE (a)-[r:{rel_type}]->(b)
    SET r.created_at = datetime()
    """
    if rel_props:
        for key, value in rel_props.items():
            query += f", r.{key} = ${key}"

    params = {"from_value": from_id, "to_value": to_id}
    if rel_props:
        params.update(rel_props)

    result = tx.run(query, **params)
    print(f"  ✓ Neo4j: 创建关系成功: {from_id} -[{rel_type}]-> {to_id}")
    return True


def neo4j_find_question(tx, qid):
    """Neo4j: 查询 Question 节点及其关系"""
    query = """
    MATCH (q:Question {id: $qid})
    OPTIONAL MATCH (q)-[r1:CONTEXT]->(d:Document)
    OPTIONAL MATCH (q)-[r2:SUPPORTS]->(s:Document)
    OPTIONAL MATCH (q)-[r3:MENTIONS]->(e:Entity)
    RETURN q.id AS id, 
           q.text AS text,
           collect(DISTINCT d.title) AS context_docs,
           collect(DISTINCT s.title) AS supporting_docs,
           collect(DISTINCT e.name) AS mentioned_entities
    """
    result = tx.run(query, qid=qid)
    record = result.single()
    if record:
        print(f"  ✓ Neo4j: 找到 Question: {qid}")
        return dict(record)
    print(f"  ✗ Neo4j: 未找到 Question: {qid}")
    return None


def neo4j_find_entity_relations(tx, entity_name):
    """Neo4j: 查询 Entity 的所有关系"""
    query = """
    MATCH (e:Entity {name: $entity_name})
    OPTIONAL MATCH (e)-[r:RELATION]->(other)
    OPTIONAL MATCH (other2)-[r2:RELATION]->(e)
    RETURN e.name AS name,
           collect(DISTINCT {type: r.type, target: other.name}) AS outgoing,
           collect(DISTINCT {type: r2.type, source: other2.name}) AS incoming
    """
    result = tx.run(query, entity_name=entity_name)
    record = result.single()
    if record:
        print(f"  ✓ Neo4j: 找到 Entity: {entity_name}")
        return dict(record)
    print(f"  ✗ Neo4j: 未找到 Entity: {entity_name}")
    return None


def neo4j_update_question_text(tx, qid, new_text):
    """Neo4j: 更新 Question 文本"""
    query = """
    MATCH (q:Question {id: $qid})
    SET q.text = $new_text, q.updated_at = datetime()
    RETURN q.id AS id
    """
    result = tx.run(query, qid=qid, new_text=new_text)
    if result.single():
        print(f"  ✓ Neo4j: 更新 Question 文本成功: {qid}")
        return True
    print(f"  ✗ Neo4j: 未找到 Question: {qid}")
    return False


def neo4j_update_entity_name(tx, old_name, new_name):
    """Neo4j: 更新 Entity 名称"""
    query = """
    MATCH (e:Entity {name: $old_name})
    SET e.name = $new_name, e.updated_at = datetime()
    RETURN e.name AS name
    """
    result = tx.run(query, old_name=old_name, new_name=new_name)
    if result.single():
        print(f"  ✓ Neo4j: 更新 Entity 名称成功: {old_name} -> {new_name}")
        return True
    print(f"  ✗ Neo4j: 未找到 Entity: {old_name}")
    return False


def neo4j_delete_question(tx, qid):
    """Neo4j: 删除 Question 节点及所有关系"""
    query = """
    MATCH (q:Question {id: $qid})
    DETACH DELETE q
    RETURN count(q) AS deleted
    """
    result = tx.run(query, qid=qid)
    deleted = result.single()['deleted']
    if deleted > 0:
        print(f"  ✓ Neo4j: 删除 Question 成功: {qid}")
        return True
    print(f"  ✗ Neo4j: 未找到 Question: {qid}")
    return False


def neo4j_delete_entity(tx, name):
    """Neo4j: 删除 Entity 节点及所有关系"""
    query = """
    MATCH (e:Entity {name: $name})
    DETACH DELETE e
    RETURN count(e) AS deleted
    """
    result = tx.run(query, name=name)
    deleted = result.single()['deleted']
    if deleted > 0:
        print(f"  ✓ Neo4j: 删除 Entity 成功: {name}")
        return True
    print(f"  ✗ Neo4j: 未找到 Entity: {name}")
    return False


def neo4j_delete_relationship(tx, from_id, to_id, rel_type):
    """Neo4j: 删除关系"""
    query = """
    MATCH (a:Entity {name: $from_id})-[r:RELATION]->(b:Entity {name: $to_id})
    WHERE type(r) = $rel_type
    DELETE r
    RETURN count(r) AS deleted
    """
    result = tx.run(query, from_id=from_id, to_id=to_id, rel_type=rel_type)
    deleted = result.single()['deleted']
    if deleted > 0:
        print(f"  ✓ Neo4j: 删除关系成功: {from_id} -[{rel_type}]-> {to_id}")
        return True
    print(f"  ✗ Neo4j: 未找到关系")
    return False


def neo4j_count(tx, label=None):
    """Neo4j: 统计节点数量"""
    if label:
        query = f"MATCH (n:{label}) RETURN count(n) AS count"
    else:
        query = "MATCH (n) RETURN count(n) AS count"
    result = tx.run(query)
    count = result.single()['count']
    print(f"  ✓ Neo4j: {label if label else '总'}节点数: {count}")
    return count


def neo4j_find_multi_hop_path(tx, start_name, start_label, max_hops=3):
    """Neo4j: 多跳路径查询"""
    prop_name = "id" if start_label == "Question" else ("title" if start_label == "Document" else "name")
    query = f"""
    MATCH path = (start:{start_label} {{{prop_name}: $start_name}})-[*1..{max_hops}]-(end)
    WHERE ALL(n IN nodes(path) WHERE single(m IN nodes(path) WHERE m = n))
    RETURN [node IN nodes(path) | labels(node)[0] + ':' + coalesce(node.name, node.title, node.id)] AS path_nodes,
           [rel IN relationships(path) | type(rel)] AS path_relationships,
           length(path) AS path_length
    LIMIT 10
    """
    result = tx.run(query, start_name=start_name)
    paths = []
    for record in result:
        paths.append({
            'nodes': record['path_nodes'],
            'relationships': record['path_relationships'],
            'length': record['path_length']
        })
    print(f"  ✓ Neo4j: 找到 {len(paths)} 条多跳路径")
    return paths


# ============================================================
# 联合操作（同时操作 MongoDB 和 Neo4j）
# ============================================================

def joint_insert(collection, neo4j_driver, sample_data):
    """联合插入：同时插入 MongoDB 和 Neo4j"""
    print(f"\n[联合插入] 插入文档: {sample_data.get('_id')}")

    # 1. 插入 MongoDB
    mongo_result = mongo_insert_one(collection, sample_data)
    if not mongo_result:
        print("  ✗ 联合插入失败: MongoDB 插入失败")
        return False

    # 2. 插入 Neo4j
    with neo4j_driver.session() as session:
        # 创建 Question 节点
        session.execute_write(neo4j_create_question,
                              sample_data['_id'],
                              sample_data['question'])

        # 创建 Document 节点和 CONTEXT 关系
        for ctx in sample_data.get('context', []):
            session.execute_write(neo4j_create_document, ctx['title'])
            session.execute_write(
                neo4j_create_relationship,
                "Question", sample_data['_id'], "id",
                "Document", ctx['title'], "title",
                "CONTEXT"
            )

        # 创建 SUPPORTS 关系
        for fact in sample_data.get('supporting_facts', []):
            session.execute_write(neo4j_create_document, fact['title'])
            session.execute_write(
                neo4j_create_relationship,
                "Question", sample_data['_id'], "id",
                "Document", fact['title'], "title",
                "SUPPORTS"
            )

        # 创建 Entity 节点和关系
        for evidence in sample_data.get('evidences', []):
            if len(evidence) == 3:
                subj, rel, obj = evidence
                session.execute_write(neo4j_create_entity, subj)
                session.execute_write(neo4j_create_entity, obj)
                session.execute_write(
                    neo4j_create_relationship,
                    "Entity", subj, "name",
                    "Entity", obj, "name",
                    "RELATION",
                    {"type": rel}
                )
                # Question 关联实体
                session.execute_write(
                    neo4j_create_relationship,
                    "Question", sample_data['_id'], "id",
                    "Entity", subj, "name",
                    "MENTIONS"
                )
                session.execute_write(
                    neo4j_create_relationship,
                    "Question", sample_data['_id'], "id",
                    "Entity", obj, "name",
                    "MENTIONS"
                )

    print(f"  ✓ 联合插入成功: {sample_data.get('_id')}")
    return True


def joint_update_question(collection, neo4j_driver, qid, new_question_text, new_answer=None):
    """联合更新：同时更新 MongoDB 和 Neo4j 中的 Question"""
    print(f"\n[联合更新] 更新 Question: {qid}")

    # 1. 更新 MongoDB
    update_data = {"question": new_question_text}
    if new_answer:
        update_data["answer"] = new_answer

    mongo_result = mongo_update_one(collection, {"_id": qid}, update_data)

    # 2. 更新 Neo4j
    with neo4j_driver.session() as session:
        neo4j_result = session.execute_write(neo4j_update_question_text, qid, new_question_text)

    if mongo_result > 0 and neo4j_result:
        print(f"  ✓ 联合更新成功: {qid}")
        return True
    print(f"  ✗ 联合更新失败: {qid}")
    return False


def joint_delete_question(collection, neo4j_driver, qid):
    """联合删除：同时从 MongoDB 和 Neo4j 中删除 Question"""
    print(f"\n[联合删除] 删除 Question: {qid}")

    # 保存备份以便验证
    backup = collection.find_one({"_id": qid})

    # 1. 删除 MongoDB
    mongo_result = mongo_delete_one(collection, {"_id": qid})

    # 2. 删除 Neo4j
    with neo4j_driver.session() as session:
        neo4j_result = session.execute_write(neo4j_delete_question, qid)

    if mongo_result > 0 and neo4j_result:
        print(f"  ✓ 联合删除成功: {qid}")
        return backup
    print(f"  ✗ 联合删除失败: {qid}")
    return None


def joint_query_full(collection, neo4j_driver, qid):
    """联合查询：从 MongoDB 和 Neo4j 同时获取数据"""
    print(f"\n[联合查询] 查询 Question: {qid}")

    # 1. 查询 MongoDB
    mongo_data = mongo_find_one(collection, {"_id": qid})

    # 2. 查询 Neo4j
    with neo4j_driver.session() as session:
        neo4j_data = session.execute_read(neo4j_find_question, qid)

    if mongo_data and neo4j_data:
        print(f"  ✓ 联合查询成功: {qid}")
        return {
            "mongodb": mongo_data,
            "neo4j": neo4j_data
        }
    print(f"  ✗ 联合查询失败: 部分数据缺失")
    return None


# ============================================================
# 主程序：完整的增删改查实验
# ============================================================

def main():
    print("=" * 80)
    print("MongoDB + Neo4j 联合增删改查实验")
    print("=" * 80)

    # 初始化连接
    mongo_client, mongo_collection = init_mongodb()
    neo4j_driver = init_neo4j()

    try:
        # ============================================================
        # 1. 保存原始数据（用于最终校验）
        # ============================================================
        print("\n[步骤1] 保存原始数据快照")
        original_mongo_docs = list(mongo_collection.find({}, {"_id": 1}))
        original_mongo_ids = sorted([doc["_id"] for doc in original_mongo_docs])
        print(f"  MongoDB 原始文档总数: {len(original_mongo_ids)}")

        with neo4j_driver.session() as session:
            original_question_count = session.execute_read(neo4j_count, "Question")

        # ============================================================
        # 2. 查询操作演示
        # ============================================================
        print("\n[步骤2] 查询操作演示")

        # 2.1 MongoDB 查询示例
        print("\n  [2.1] MongoDB 查询示例")
        sample_question = mongo_find_one(mongo_collection, {"type": "comparison"})
        if sample_question:
            print(f"    示例文档: {sample_question.get('_id')}")
            print(f"      问题: {sample_question.get('question')[:50]}...")
            print(f"      答案: {sample_question.get('answer')}")
            print(f"      类型: {sample_question.get('type')}")

        # 2.2 Neo4j 查询示例
        print("\n  [2.2] Neo4j 查询示例")
        with neo4j_driver.session() as session:
            # 获取第一个 Question
            result = session.run("MATCH (q:Question) RETURN q.id AS id LIMIT 1")
            record = result.single()
            if record:
                neo4j_question = session.execute_read(neo4j_find_question, record['id'])
                if neo4j_question:
                    print(f"    示例 Question: {neo4j_question.get('id')}")
                    print(f"      文本: {neo4j_question.get('text', '')[:50]}...")
                    print(f"      Context文档: {neo4j_question.get('context_docs', [])[:3]}")

        # 2.3 多跳路径查询
        print("\n  [2.3] 多跳路径查询示例")
        with neo4j_driver.session() as session:
            # 查找一个 Entity 进行多跳查询
            result = session.run("MATCH (e:Entity) RETURN e.name AS name LIMIT 1")
            record = result.single()
            if record:
                paths = session.execute_read(neo4j_find_multi_hop_path, record['name'], "Entity", 3)
                if paths:
                    print(f"    从实体 '{record['name']}' 出发的路径:")
                    for i, path in enumerate(paths[:3], 1):
                        print(f"      路径{i}: {' -> '.join(path['nodes'])}")

        # 2.4 统计查询
        print("\n  [2.4] 统计查询")
        # MongoDB 聚合：按类型统计
        pipeline = [
            {"$group": {"_id": "$type", "count": {"$sum": 1}}},
            {"$sort": {"count": -1}}
        ]
        type_stats = mongo_aggregate(mongo_collection, pipeline)
        print("    问题类型分布 (MongoDB):")
        for stat in type_stats[:5]:
            print(f"      {stat.get('_id', 'unknown')}: {stat.get('count')} 条")

        # Neo4j 统计
        with neo4j_driver.session() as session:
            doc_count = session.execute_read(neo4j_count, "Document")
            entity_count = session.execute_read(neo4j_count, "Entity")

        # ============================================================
        # 3. 插入操作（临时数据，后续会删除）
        # ============================================================
        print("\n[步骤3] 插入临时测试数据")

        test_question_id = "TEST_QA_001"
        test_data = {
            "_id": test_question_id,
            "question": "这是一个测试问题，用于验证CRUD操作？",
            "answer": "测试答案",
            "type": "test",
            "context": [
                {
                    "title": "测试文档",
                    "sentences": ["这是测试文档的第一句话。", "这是第二句话。"]
                }
            ],
            "supporting_facts": [
                {"title": "测试文档", "sent_id": 0}
            ],
            "evidences": [
                ["测试实体1", "测试关系", "测试实体2"]
            ],
            "entity_ids": ""
        }

        # 联合插入
        joint_insert(mongo_collection, neo4j_driver, test_data)

        # ============================================================
        # 4. 更新操作
        # ============================================================
        print("\n[步骤4] 更新测试数据")

        # 联合更新
        joint_update_question(
            mongo_collection, neo4j_driver,
            test_question_id,
            "这是一个【已更新】的测试问题，验证更新操作是否成功？",
            "已更新的答案"
        )

        # 验证更新
        joint_query_full(mongo_collection, neo4j_driver, test_question_id)

        # ============================================================
        # 5. 删除操作
        # ============================================================
        print("\n[步骤5] 删除测试数据")

        # 联合删除
        deleted_backup = joint_delete_question(mongo_collection, neo4j_driver, test_question_id)

        # 验证删除
        verify = joint_query_full(mongo_collection, neo4j_driver, test_question_id)
        if not verify:
            print("  ✓ 验证成功: 测试数据已完全删除")

        # ============================================================
        # 6. 批量操作演示
        # ============================================================
        print("\n[步骤6] 批量操作演示")

        # 批量插入临时数据
        batch_data = []
        for i in range(3):
            batch_data.append({
                "_id": f"BATCH_TEST_{i:03d}",
                "question": f"批量测试问题 {i}",
                "answer": f"批量答案 {i}",
                "type": "batch_test",
                "context": [],
                "supporting_facts": [],
                "evidences": [],
                "entity_ids": ""
            })

        print("  [6.1] MongoDB 批量插入")
        mongo_insert_many(mongo_collection, batch_data)

        print("  [6.2] MongoDB 批量更新")
        mongo_update_many(
            mongo_collection,
            {"type": "batch_test"},
            {"status": "processed"}
        )

        print("  [6.3] MongoDB 批量删除")
        mongo_delete_many(mongo_collection, {"type": "batch_test"})

        # ============================================================
        # 7. 最终一致性校验
        # ============================================================
        print("\n[步骤7] 最终一致性校验")

        final_mongo_docs = list(mongo_collection.find({}, {"_id": 1}))
        final_mongo_ids = sorted([doc["_id"] for doc in final_mongo_docs])

        with neo4j_driver.session() as session:
            final_question_count = session.execute_read(neo4j_count, "Question")

        print(f"  MongoDB 原始文档数: {len(original_mongo_ids)}")
        print(f"  MongoDB 最终文档数: {len(final_mongo_ids)}")
        print(f"  Neo4j 原始 Question 数: {original_question_count}")
        print(f"  Neo4j 最终 Question 数: {final_question_count}")

        if len(original_mongo_ids) == len(final_mongo_ids) and original_question_count == final_question_count:
            print("  ✓ 数据完整性校验通过！原始数据未被破坏")
        else:
            print("  ✗ 警告: 数据总数发生变化，请检查操作")

        # ============================================================
        # 8. 导出部分数据示例
        # ============================================================
        print("\n[步骤8] 导出示例数据")

        export_data = list(mongo_collection.find({}, {"_id": 1, "question": 1, "type": 1}).limit(5))
        for doc in export_data:
            print(f"    {doc.get('_id')}: {doc.get('question', '')[:40]}... [{doc.get('type')}]")

        print("\n" + "=" * 80)
        print("实验完成")
        print("=" * 80)

    finally:
        # 关闭连接
        mongo_client.close()
        neo4j_driver.close()
        print("\n数据库连接已关闭")


if __name__ == "__main__":
    main()