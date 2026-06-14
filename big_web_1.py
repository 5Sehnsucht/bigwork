#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import streamlit as st
import pandas as pd
from datetime import datetime
import time
import json
from collections import Counter
from neo4j import GraphDatabase
from pymongo import MongoClient
import plotly.graph_objects as go
import plotly.express as px
import networkx as nx
from typing import List, Dict, Any, Optional, Tuple
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA


class Neo4jManager:
    """Neo4j database management class for multi-hop QA system"""

    def __init__(self, uri='bolt://localhost:7687', username=None, password=None):
        self.uri = uri
        self.username = username
        self.password = password
        self.driver = None
        self.current_db = "neo4j"

    def connect(self):
        try:
            self.driver = GraphDatabase.driver(self.uri, auth=(self.username, self.password))
            with self.driver.session() as session:
                result = session.run("RETURN 1 as test")
                result.single()
            return True, "Connected to Neo4j successfully"
        except Exception as e:
            return False, f"Connection failed: {str(e)}"

    def disconnect(self):
        if self.driver:
            self.driver.close()
            self.driver = None

    def get_databases(self):
        try:
            with self.driver.session() as session:
                result = session.run("SHOW DATABASES")
                dbs = [record['name'] for record in result]
                system_dbs = ['system']
                return [db for db in dbs if db not in system_dbs]
        except Exception as e:
            return ["neo4j"]

    def create_database(self, db_name):
        try:
            with self.driver.session() as session:
                session.run(f"CREATE DATABASE `{db_name}`")
            return True, f"Database '{db_name}' created successfully"
        except Exception as e:
            if "already exists" in str(e):
                return False, f"Database '{db_name}' already exists"
            return False, f"Failed to create database: {str(e)}"

    def delete_database(self, db_name):
        try:
            with self.driver.session() as session:
                session.run(f"DROP DATABASE `{db_name}`")
            return True, f"Database '{db_name}' deleted"
        except Exception as e:
            return False, f"Failed to delete database: {str(e)}"

    def set_database(self, db_name):
        self.current_db = db_name
        return True, f"Using database '{db_name}'"

    # ==================== Multi-hop QA Specific Methods ====================

    def get_all_questions(self, limit=100, skip=0):
        """Get all Question nodes"""
        try:
            with self.driver.session(database=self.current_db) as session:
                query = """
                    MATCH (q:Question)
                    OPTIONAL MATCH (q)-[:CONTEXT]->(d:Document)
                    OPTIONAL MATCH (q)-[:SUPPORTS]->(s:Document)
                    OPTIONAL MATCH (q)-[:MENTIONS]->(e:Entity)
                    RETURN q.id AS id,
                           q.text AS text,
                           collect(DISTINCT d.title) AS context_docs,
                           collect(DISTINCT s.title) AS supporting_docs,
                           collect(DISTINCT e.name) AS mentioned_entities
                    ORDER BY q.id
                    SKIP $skip LIMIT $limit
                """
                result = session.run(query, skip=skip, limit=limit)
                questions = []
                for record in result:
                    questions.append({
                        'id': record['id'],
                        'text': record['text'],
                        'context_docs': record['context_docs'],
                        'supporting_docs': record['supporting_docs'],
                        'mentioned_entities': record['mentioned_entities']
                    })

                count_result = session.run("MATCH (q:Question) RETURN count(q) AS total")
                total = count_result.single()['total']
                return questions, total
        except Exception as e:
            st.error(f"Failed to get questions: {e}")
            return [], 0

    def get_all_documents(self, limit=100):
        """Get all Document nodes"""
        try:
            with self.driver.session(database=self.current_db) as session:
                query = """
                    MATCH (d:Document)
                    OPTIONAL MATCH (q:Question)-[:CONTEXT|SUPPORTS]->(d)
                    RETURN d.title AS title,
                           collect(DISTINCT q.id) AS related_questions
                    LIMIT $limit
                """
                result = session.run(query, limit=limit)
                return [dict(record) for record in result]
        except Exception as e:
            st.error(f"Failed to get documents: {e}")
            return []

    def get_all_entities(self, limit=100):
        """Get all Entity nodes"""
        try:
            with self.driver.session(database=self.current_db) as session:
                query = """
                    MATCH (e:Entity)
                    OPTIONAL MATCH (q:Question)-[:MENTIONS]->(e)
                    OPTIONAL MATCH (e1:Entity)-[r:RELATION]->(e2:Entity)
                    WHERE e1.name = e.name OR e2.name = e.name
                    RETURN e.name AS name,
                           collect(DISTINCT q.id) AS mentioned_in_questions,
                           count(DISTINCT r) AS relation_count
                    LIMIT $limit
                """
                result = session.run(query, limit=limit)
                return [dict(record) for record in result]
        except Exception as e:
            st.error(f"Failed to get entities: {e}")
            return []

    def get_question_by_id(self, qid):
        """Get single question by ID"""
        try:
            with self.driver.session(database=self.current_db) as session:
                query = """
                    MATCH (q:Question {id: $qid})
                    OPTIONAL MATCH (q)-[:CONTEXT]->(d:Document)
                    OPTIONAL MATCH (q)-[:SUPPORTS]->(s:Document)
                    OPTIONAL MATCH (q)-[:MENTIONS]->(e:Entity)
                    RETURN q.id AS id,
                           q.text AS text,
                           collect(DISTINCT d.title) AS context_docs,
                           collect(DISTINCT s.title) AS supporting_docs,
                           collect(DISTINCT e.name) AS mentioned_entities
                """
                result = session.run(query, qid=qid)
                record = result.single()
                if record:
                    return {
                        'id': record['id'],
                        'text': record['text'],
                        'context_docs': record['context_docs'],
                        'supporting_docs': record['supporting_docs'],
                        'mentioned_entities': record['mentioned_entities']
                    }
                return None
        except Exception as e:
            return None

    def find_questions_by_entity(self, entity_name, limit=100):
        """Find questions that mention a specific entity"""
        try:
            with self.driver.session(database=self.current_db) as session:
                query = """
                    MATCH (e:Entity {name: $entity_name})<-[:MENTIONS]-(q:Question)
                    OPTIONAL MATCH (q)-[:CONTEXT]->(d:Document)
                    RETURN q.id AS id,
                           q.text AS text,
                           collect(DISTINCT d.title) AS context_docs
                    LIMIT $limit
                """
                result = session.run(query, entity_name=entity_name, limit=limit)
                return [dict(record) for record in result]
        except Exception as e:
            st.error(f"Query failed: {e}")
            return []

    def find_questions_by_document(self, doc_title, limit=100):
        """Find questions related to a specific document"""
        try:
            with self.driver.session(database=self.current_db) as session:
                query = """
                    MATCH (d:Document {title: $doc_title})
                    MATCH (q:Question)-[:CONTEXT|SUPPORTS]->(d)
                    RETURN DISTINCT q.id AS id,
                           q.text AS text
                    LIMIT $limit
                """
                result = session.run(query, doc_title=doc_title, limit=limit)
                return [dict(record) for record in result]
        except Exception as e:
            st.error(f"Query failed: {e}")
            return []

    def multi_hop_path_query(self, start_node_name, start_label, max_hops=3):
        """Find multi-hop paths from a starting node"""
        try:
            with self.driver.session(database=self.current_db) as session:
                query = f"""
                    MATCH path = (start:{start_label} {{name: $start_name}})-[*1..{max_hops}]-(end)
                    WHERE ALL(n IN nodes(path) WHERE single(m IN nodes(path) WHERE m = n))
                    RETURN [node IN nodes(path) | labels(node)[0] + ':' + coalesce(node.name, node.title, node.id)] AS path_nodes,
                           [rel IN relationships(path) | type(rel)] AS path_relationships,
                           length(path) AS path_length
                    LIMIT 50
                """
                result = session.run(query, start_name=start_node_name)
                paths = []
                for record in result:
                    paths.append({
                        'nodes': record['path_nodes'],
                        'relationships': record['path_relationships'],
                        'length': record['path_length']
                    })
                return paths
        except Exception as e:
            st.error(f"Multi-hop query failed: {e}")
            return []

    def shortest_path_between_entities(self, entity1, entity2):
        """Find shortest path between two entities"""
        try:
            with self.driver.session(database=self.current_db) as session:
                query = """
                    MATCH path = shortestPath((e1:Entity {name: $entity1})-[*]-(e2:Entity {name: $entity2}))
                    WHERE e1.name <> e2.name
                    RETURN [node IN nodes(path) | labels(node)[0] + ':' + coalesce(node.name, node.title, node.id)] AS path_nodes,
                           [rel IN relationships(path) | type(rel)] AS path_relationships,
                           length(path) AS path_length
                """
                result = session.run(query, entity1=entity1, entity2=entity2)
                record = result.single()
                if record:
                    return {
                        'nodes': record['path_nodes'],
                        'relationships': record['path_relationships'],
                        'length': record['path_length']
                    }
                return None
        except Exception as e:
            st.error(f"Shortest path query failed: {e}")
            return None

    def get_node_neighbors(self, node_label, node_name, hop=1):
        """Get neighbors of a specific node within given hops"""
        try:
            with self.driver.session(database=self.current_db) as session:
                if node_label == "Question":
                    prop_name = "id"
                elif node_label == "Document":
                    prop_name = "title"
                else:
                    prop_name = "name"

                query = f"""
                    MATCH (start:{node_label} {{{prop_name}: $node_name}})-[r*1..{hop}]-(neighbor)
                    RETURN DISTINCT labels(neighbor)[0] AS neighbor_label,
                           coalesce(neighbor.name, neighbor.title, neighbor.id) AS neighbor_name,
                           type(r[0]) AS relationship_type
                    LIMIT 100
                """
                result = session.run(query, node_name=node_name)
                neighbors = []
                for record in result:
                    neighbors.append({
                        'label': record['neighbor_label'],
                        'name': record['neighbor_name'],
                        'relationship': record['relationship_type']
                    })
                return neighbors
        except Exception as e:
            st.error(f"Neighbor query failed: {e}")
            return []

    # ==================== CRUD Operations ====================

    def create_question_node(self, qid, text):
        """Create a new Question node"""
        try:
            with self.driver.session(database=self.current_db) as session:
                query = """
                    MERGE (q:Question {id: $qid})
                    SET q.text = $text,
                        q.created_at = datetime()
                    RETURN q.id AS id
                """
                result = session.run(query, qid=qid, text=text)
                if result.single():
                    return True, f"Question '{qid}' created"
                return False, "Creation failed"
        except Exception as e:
            return False, str(e)

    def create_document_node(self, title):
        """Create a new Document node"""
        try:
            with self.driver.session(database=self.current_db) as session:
                query = """
                    MERGE (d:Document {title: $title})
                    SET d.created_at = datetime()
                    RETURN d.title AS title
                """
                result = session.run(query, title=title)
                if result.single():
                    return True, f"Document '{title}' created"
                return False, "Creation failed"
        except Exception as e:
            return False, str(e)

    def create_entity_node(self, name):
        """Create a new Entity node"""
        try:
            with self.driver.session(database=self.current_db) as session:
                query = """
                    MERGE (e:Entity {name: $name})
                    SET e.created_at = datetime()
                    RETURN e.name AS name
                """
                result = session.run(query, name=name)
                if result.single():
                    return True, f"Entity '{name}' created"
                return False, "Creation failed"
        except Exception as e:
            return False, str(e)

    def create_relationship(self, from_label, from_id, from_prop, to_label, to_id, to_prop, rel_type):
        """Create relationship between any two nodes"""
        try:
            with self.driver.session(database=self.current_db) as session:
                query = f"""
                    MATCH (a:{from_label} {{{from_prop}: $from_value}})
                    MATCH (b:{to_label} {{{to_prop}: $to_value}})
                    MERGE (a)-[r:{rel_type}]->(b)
                    SET r.created_at = datetime()
                    RETURN type(r) AS rel_type
                """
                result = session.run(query, from_value=from_id, to_value=to_id)
                if result.single():
                    return True, f"Relationship '{rel_type}' created"
                return False, "Creation failed"
        except Exception as e:
            return False, str(e)

    def delete_relationship(self, from_label, from_id, from_prop, to_label, to_id, to_prop, rel_type):
        """Delete relationship between two nodes"""
        try:
            with self.driver.session(database=self.current_db) as session:
                query = f"""
                    MATCH (a:{from_label} {{{from_prop}: $from_value}})
                    MATCH (b:{to_label} {{{to_prop}: $to_value}})
                    MATCH (a)-[r:{rel_type}]->(b)
                    DELETE r
                    RETURN count(r) AS deleted
                """
                result = session.run(query, from_value=from_id, to_value=to_id)
                if result.single()['deleted'] > 0:
                    return True, f"Relationship '{rel_type}' deleted"
                return False, "Relationship not found"
        except Exception as e:
            return False, str(e)

    def update_question_text(self, qid, new_text):
        """Update question text"""
        try:
            with self.driver.session(database=self.current_db) as session:
                query = """
                    MATCH (q:Question {id: $qid})
                    SET q.text = $new_text,
                        q.updated_at = datetime()
                    RETURN q.id AS id
                """
                result = session.run(query, qid=qid, new_text=new_text)
                if result.single():
                    return True, f"Question '{qid}' updated"
                return False, f"Question '{qid}' not found"
        except Exception as e:
            return False, str(e)

    def delete_question(self, qid):
        """Delete a Question node and all its relationships"""
        try:
            with self.driver.session(database=self.current_db) as session:
                query = """
                    MATCH (q:Question {id: $qid})
                    DETACH DELETE q
                    RETURN count(q) AS deleted
                """
                result = session.run(query, qid=qid)
                if result.single()['deleted'] > 0:
                    return True, f"Question '{qid}' deleted"
                return False, f"Question '{qid}' not found"
        except Exception as e:
            return False, str(e)

    def delete_document(self, title):
        """Delete a Document node and all its relationships"""
        try:
            with self.driver.session(database=self.current_db) as session:
                query = """
                    MATCH (d:Document {title: $title})
                    DETACH DELETE d
                    RETURN count(d) AS deleted
                """
                result = session.run(query, title=title)
                if result.single()['deleted'] > 0:
                    return True, f"Document '{title}' deleted"
                return False, f"Document '{title}' not found"
        except Exception as e:
            return False, str(e)

    def delete_entity(self, name):
        """Delete an Entity node and all its relationships"""
        try:
            with self.driver.session(database=self.current_db) as session:
                query = """
                    MATCH (e:Entity {name: $name})
                    DETACH DELETE e
                    RETURN count(e) AS deleted
                """
                result = session.run(query, name=name)
                if result.single()['deleted'] > 0:
                    return True, f"Entity '{name}' deleted"
                return False, f"Entity '{name}' not found"
        except Exception as e:
            return False, str(e)

    def get_node_by_label_and_property(self, label, prop_name, prop_value):
        """Get node by label and property"""
        try:
            with self.driver.session(database=self.current_db) as session:
                query = f"""
                    MATCH (n:{label} {{{prop_name}: $value}})
                    RETURN n
                """
                result = session.run(query, value=prop_value)
                record = result.single()
                if record:
                    return dict(record['n'])
                return None
        except Exception as e:
            return None

    def get_nodes_by_label(self, label, limit=200):
        """Get all nodes with a specific label"""
        try:
            with self.driver.session(database=self.current_db) as session:
                query = f"""
                    MATCH (n:{label})
                    RETURN n
                    LIMIT {limit}
                """
                result = session.run(query)
                nodes = []
                for record in result:
                    node = dict(record['n'])
                    nodes.append(node)
                return nodes
        except Exception as e:
            st.error(f"Failed to get nodes: {e}")
            return []

    def get_all_node_labels(self):
        try:
            with self.driver.session(database=self.current_db) as session:
                query = "CALL db.labels() YIELD label RETURN label ORDER BY label"
                result = session.run(query)
                return [record['label'] for record in result]
        except Exception as e:
            return []

    def get_relationships_by_node_generic(self, label, prop_name, prop_value):
        """Get all relationships for a specific node"""
        try:
            with self.driver.session(database=self.current_db) as session:
                query = f"""
                    MATCH (n:{label} {{{prop_name}: $value}})
                    OPTIONAL MATCH (n)-[r]->(m)
                    OPTIONAL MATCH (k)-[r2]->(n)
                    RETURN n,
                           collect(DISTINCT {{
                               direction: 'OUTGOING',
                               type: type(r),
                               target_label: labels(m)[0],
                               target_name: coalesce(m.name, m.title, m.id)
                           }}) as outgoing,
                           collect(DISTINCT {{
                               direction: 'INCOMING',
                               type: type(r2),
                               source_label: labels(k)[0],
                               source_name: coalesce(k.name, k.title, k.id)
                           }}) as incoming
                """
                result = session.run(query, value=prop_value)
                record = result.single()
                if record:
                    relationships = []
                    for rel in record['outgoing']:
                        if rel['type']:
                            relationships.append({
                                'direction': rel['direction'],
                                'type': rel['type'],
                                'connected_label': rel['target_label'],
                                'connected_name': rel['target_name']
                            })
                    for rel in record['incoming']:
                        if rel['type']:
                            relationships.append({
                                'direction': rel['direction'],
                                'type': rel['type'],
                                'connected_label': rel['source_label'],
                                'connected_name': rel['source_name']
                            })
                    return relationships
                return []
        except Exception as e:
            st.error(f"Failed to get relationships: {e}")
            return []

    # ==================== Statistics ====================

    def get_db_stats(self):
        try:
            with self.driver.session(database=self.current_db) as session:
                question_result = session.run("MATCH (q:Question) RETURN count(q) AS count")
                question_count = question_result.single()['count']

                doc_result = session.run("MATCH (d:Document) RETURN count(d) AS count")
                doc_count = doc_result.single()['count']

                entity_result = session.run("MATCH (e:Entity) RETURN count(e) AS count")
                entity_count = entity_result.single()['count']

                rel_result = session.run("MATCH ()-[r]->() RETURN count(r) AS count")
                rel_count = rel_result.single()['count']

                return {
                    'question_count': question_count,
                    'document_count': doc_count,
                    'entity_count': entity_count,
                    'relationship_count': rel_count
                }
        except Exception as e:
            return None

    def get_most_connected_entities(self, limit=10):
        """Get entities with most relationships"""
        try:
            with self.driver.session(database=self.current_db) as session:
                query = """
                    MATCH (e:Entity)-[r:RELATION]-()
                    RETURN e.name AS name, count(r) AS degree
                    ORDER BY degree DESC
                    LIMIT $limit
                """
                result = session.run(query, limit=limit)
                return [dict(record) for record in result]
        except Exception as e:
            return []

    def get_all_relationships_for_graph(self, limit=200):
        """Get all relationships for graph visualization"""
        try:
            with self.driver.session(database=self.current_db) as session:
                query = """
                    MATCH (a)-[r]->(b)
                    WHERE labels(a)[0] IN ['Question', 'Document', 'Entity'] 
                      AND labels(b)[0] IN ['Question', 'Document', 'Entity']
                    RETURN labels(a)[0] AS source_label,
                           coalesce(a.name, a.title, a.id) AS source_name,
                           type(r) AS relationship_type,
                           labels(b)[0] AS target_label,
                           coalesce(b.name, b.title, b.id) AS target_name
                    LIMIT $limit
                """
                result = session.run(query, limit=limit)
                return [dict(record) for record in result]
        except Exception as e:
            st.error(f"Failed to get graph data: {e}")
            return []

    def get_server_info(self):
        try:
            with self.driver.session() as session:
                result = session.run("CALL dbms.components() YIELD name, versions, edition")
                record = result.single()
                return {
                    'name': record['name'] if record else 'Neo4j',
                    'version': record['versions'][0] if record else 'Unknown',
                    'edition': record['edition'] if record else 'Community'
                }
        except Exception as e:
            return None


class MongoDBManager:
    """MongoDB management class for storing detailed QA content"""

    def __init__(self, uri='mongodb://localhost:27017/', db_name='wikimultihop_db'):
        self.uri = uri
        self.db_name = db_name
        self.client = None
        self.db = None

    def connect(self):
        try:
            # 使用正确的 pymongo TLS 参数
            self.client = MongoClient(
                self.uri,
                tls=True,
                tlsAllowInvalidCertificates=True,
                tlsAllowInvalidHostnames=True,
                serverSelectionTimeoutMS=30000
            )
            self.db = self.client[self.db_name]
            self.client.admin.command('ping')
            return True, "Connected to MongoDB successfully"
        except Exception as e:
            return False, f"Connection failed: {str(e)}"

    def disconnect(self):
        if self.client:
            self.client.close()
            self.client = None
            self.db = None

    def get_qa_by_id(self, qid):
        """Get full QA document by ID"""
        try:
            collection = self.db['qa']
            doc = collection.find_one({'_id': qid})
            if doc:
                doc['_id'] = str(doc['_id']) if doc.get('_id') else doc.get('_id')
            return doc
        except Exception as e:
            st.error(f"Failed to get QA: {e}")
            return None

    def search_questions(self, search_term, limit=50):
        """Search questions by text content"""
        try:
            collection = self.db['qa']
            docs = collection.find(
                {'question': {'$regex': search_term, '$options': 'i'}},
                {'_id': 1, 'question': 1, 'answer': 1, 'type': 1}
            ).limit(limit)
            return list(docs)
        except Exception as e:
            st.error(f"Search failed: {e}")
            return []

    def get_questions_by_type(self, qtype, limit=100):
        """Get questions by type (bridge/comparison)"""
        try:
            collection = self.db['qa']
            docs = collection.find({'type': qtype}, {'_id': 1, 'question': 1, 'answer': 1}).limit(limit)
            return list(docs)
        except Exception as e:
            st.error(f"Query failed: {e}")
            return []

    def get_qa_count(self):
        """Get total number of QA documents"""
        try:
            collection = self.db['qa']
            return collection.count_documents({})
        except Exception as e:
            return 0

    def get_type_distribution(self):
        """Get distribution of question types"""
        try:
            collection = self.db['qa']
            pipeline = [
                {'$group': {'_id': '$type', 'count': {'$sum': 1}}},
                {'$sort': {'count': -1}}
            ]
            results = list(collection.aggregate(pipeline))
            return [{'type': r['_id'], 'count': r['count']} for r in results if r['_id']]
        except Exception as e:
            return []

    def get_full_context_by_question(self, qid):
        """Get full context (all sentences) for a question"""
        try:
            collection = self.db['qa']
            doc = collection.find_one({'_id': qid}, {'context': 1, 'supporting_facts': 1})
            return doc
        except Exception as e:
            return None

    def insert_qa_document(self, qa_document):
        """Insert a new QA document"""
        try:
            collection = self.db['qa']
            result = collection.insert_one(qa_document)
            return True, f"Document inserted with ID: {result.inserted_id}"
        except Exception as e:
            return False, str(e)

    def update_qa_document(self, qid, update_fields):
        """Update a QA document"""
        try:
            collection = self.db['qa']
            result = collection.update_one({'_id': qid}, {'$set': update_fields})
            if result.modified_count > 0:
                return True, f"Document '{qid}' updated"
            return False, f"Document '{qid}' not found or no changes made"
        except Exception as e:
            return False, str(e)

    def delete_qa_document(self, qid):
        """Delete a QA document"""
        try:
            collection = self.db['qa']
            result = collection.delete_one({'_id': qid})
            if result.deleted_count > 0:
                return True, f"Document '{qid}' deleted"
            return False, f"Document '{qid}' not found"
        except Exception as e:
            return False, str(e)

    def get_all_qa_documents(self, limit=100, skip=0):
        """Get all QA documents with pagination"""
        try:
            collection = self.db['qa']
            docs = list(collection.find({}, {'_id': 1, 'question': 1, 'answer': 1, 'type': 1})
                        .skip(skip).limit(limit))
            for doc in docs:
                doc['_id'] = str(doc['_id']) if doc.get('_id') else doc.get('_id')
            total = collection.count_documents({})
            return docs, total
        except Exception as e:
            st.error(f"Failed to get documents: {e}")
            return [], 0

    def get_most_frequently_supported_docs(self, limit=10):
        """Get documents that appear most frequently as supporting facts"""
        try:
            collection = self.db['qa']
            pipeline = [
                {'$unwind': '$supporting_facts'},
                {'$group': {'_id': '$supporting_facts.title', 'count': {'$sum': 1}}},
                {'$sort': {'count': -1}},
                {'$limit': limit}
            ]
            results = list(collection.aggregate(pipeline))
            return [{'title': r['_id'], 'count': r['count']} for r in results if r['_id']]
        except Exception as e:
            return []

    def get_avg_supporting_facts_count(self):
        """Get average number of supporting facts per question"""
        try:
            collection = self.db['qa']
            pipeline = [
                {'$project': {'facts_count': {'$size': '$supporting_facts'}}},
                {'$group': {'_id': None, 'avg_count': {'$avg': '$facts_count'}}}
            ]
            result = list(collection.aggregate(pipeline))
            if result:
                return result[0]['avg_count']
            return 0
        except Exception as e:
            return 0

    # ==================== Clustering Methods ====================

    def get_questions_for_clustering(self, limit=500):
        """Get question texts for clustering analysis"""
        try:
            collection = self.db['qa']
            docs = list(collection.find({}, {'_id': 1, 'question': 1, 'type': 1, 'answer': 1}).limit(limit))
            for doc in docs:
                doc['_id'] = str(doc['_id']) if doc.get('_id') else doc.get('_id')
            return docs
        except Exception as e:
            st.error(f"Failed to get questions for clustering: {e}")
            return []

    def perform_question_clustering(self, n_clusters=5, max_samples=500):
        """Perform K-Means clustering on question texts"""
        questions_data = self.get_questions_for_clustering(limit=max_samples)

        if len(questions_data) < n_clusters:
            return None, None, None, None, f"Not enough data for clustering. Need at least {n_clusters} samples, got {len(questions_data)}"

        questions = [q.get('question', '') for q in questions_data]
        question_ids = [q.get('_id') for q in questions_data]
        question_types = [q.get('type', 'unknown') for q in questions_data]
        question_answers = [q.get('answer', '') for q in questions_data]

        vectorizer = TfidfVectorizer(
            max_features=1000,
            stop_words='english',
            ngram_range=(1, 2),
            min_df=2,
            max_df=0.9
        )

        try:
            tfidf_matrix = vectorizer.fit_transform(questions)

            actual_k = min(n_clusters, len(questions_data))
            kmeans = KMeans(n_clusters=actual_k, random_state=42, n_init=10)
            cluster_labels = kmeans.fit_predict(tfidf_matrix)

            pca = PCA(n_components=2, random_state=42)
            coords = pca.fit_transform(tfidf_matrix.toarray())

            clusters = []
            for i in range(actual_k):
                cluster_indices = [idx for idx, label in enumerate(cluster_labels) if label == i]
                cluster_data = {
                    'cluster_id': i,
                    'size': len(cluster_indices),
                    'questions': [questions[idx] for idx in cluster_indices],
                    'question_ids': [question_ids[idx] for idx in cluster_indices],
                    'types': [question_types[idx] for idx in cluster_indices],
                    'answers': [question_answers[idx] for idx in cluster_indices],
                }
                if len(cluster_indices) > 0:
                    center = kmeans.cluster_centers_[i]
                    distances = np.linalg.norm(tfidf_matrix[cluster_indices].toarray() - center, axis=1)
                    closest_idx = cluster_indices[np.argmin(distances)]
                    cluster_data['representative'] = questions[closest_idx]
                clusters.append(cluster_data)

            feature_names = vectorizer.get_feature_names_out()
            cluster_keywords = []
            for i in range(actual_k):
                center = kmeans.cluster_centers_[i]
                top_indices = center.argsort()[-10:][::-1]
                keywords = [feature_names[idx] for idx in top_indices if center[idx] > 0]
                cluster_keywords.append(keywords[:5])

            viz_data = []
            for idx, (x, y) in enumerate(coords):
                viz_data.append({
                    'x': float(x),
                    'y': float(y),
                    'question': questions[idx][:100],
                    'question_id': question_ids[idx],
                    'type': question_types[idx],
                    'cluster': int(cluster_labels[idx])
                })

            return clusters, cluster_keywords, viz_data, actual_k, "success"

        except Exception as e:
            return None, None, None, None, f"Clustering failed: {str(e)}"


def create_graph_visualization(edges):
    """Create interactive graph visualization using Plotly"""
    if not edges:
        return None

    G = nx.DiGraph()

    for edge in edges:
        G.add_edge(edge['source_name'], edge['target_name'],
                   label=edge['relationship_type'],
                   source_label=edge['source_label'],
                   target_label=edge['target_label'])

    pos = nx.spring_layout(G, k=2, iterations=50, seed=42)

    edge_traces = []
    for edge in G.edges(data=True):
        x0, y0 = pos[edge[0]]
        x1, y1 = pos[edge[1]]

        edge_trace = go.Scatter(
            x=[x0, x1, None],
            y=[y0, y1, None],
            line=dict(width=1.5, color='#888'),
            hoverinfo='none',
            mode='lines',
            name=f"{edge[2]['label']}"
        )
        edge_traces.append(edge_trace)

    node_x = []
    node_y = []
    node_text = []
    node_colors_list = []

    color_map = {
        'Question': '#FF6B6B',
        'Document': '#4ECDC4',
        'Entity': '#45B7D1'
    }

    node_type_map = {}
    for edge in edges:
        node_type_map[edge['source_name']] = edge['source_label']
        node_type_map[edge['target_name']] = edge['target_label']

    for node in G.nodes():
        x, y = pos[node]
        node_x.append(x)
        node_y.append(y)
        node_text.append(node)
        node_type = node_type_map.get(node, 'Entity')
        node_colors_list.append(color_map.get(node_type, '#95A5A6'))

    node_trace = go.Scatter(
        x=node_x, y=node_y,
        mode='markers+text',
        hoverinfo='text',
        text=node_text,
        textposition="top center",
        textfont=dict(size=10),
        marker=dict(
            size=25,
            color=node_colors_list,
            line=dict(width=2, color='DarkSlateGray')
        )
    )

    fig = go.Figure(data=edge_traces + [node_trace],
                    layout=go.Layout(
                        title=dict(text="Knowledge Graph Visualization", font=dict(size=16)),
                        showlegend=True,
                        hovermode='closest',
                        xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
                        yaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
                        plot_bgcolor='white',
                        height=600
                    ))

    return fig


def main():
    st.set_page_config(
        page_title="Multi-Hop QA System - Neo4j + MongoDB Visualization",
        page_icon="🕸️",
        layout="wide"
    )

    st.title("🕸️ Multi-Hop Question Answering System")
    st.markdown("### Neo4j Graph Database + MongoDB Document Store")
    st.markdown("Visualizing knowledge graphs for multi-hop reasoning")

    if 'neo4j_manager' not in st.session_state:
        st.session_state.neo4j_manager = None
    if 'mongodb_manager' not in st.session_state:
        st.session_state.mongodb_manager = None
    if 'current_db' not in st.session_state:
        st.session_state.current_db = None
    if 'run_clustering' not in st.session_state:
        st.session_state.run_clustering = False
    if 'cluster_results' not in st.session_state:
        st.session_state.cluster_results = None

    with st.sidebar:
        st.markdown("## 🔌 Database Connections")

        st.markdown("### 🕸️ Neo4j Connection")
        # 云端 Neo4j AuraDB 连接配置
        neo4j_uri = st.text_input(
            "Neo4j URI",
            value="neo4j+s://da656e87.databases.neo4j.io",
            key="neo4j_uri"
        )
        neo4j_user = st.text_input(
            "Neo4j Username",
            value="da656e87",
            key="neo4j_user"
        )
        neo4j_password = st.text_input(
            "Neo4j Password",
            type="password",
            value="yqv0IavwPN7MJBREZWHBwEpHPGaqHM6rEpXrPElB8eY",
            key="neo4j_password"
        )

        if st.button("🔗 Connect Neo4j", type="primary", use_container_width=True, key="connect_neo4j"):
            manager = Neo4jManager(neo4j_uri, neo4j_user, neo4j_password)
            success, msg = manager.connect()
            if success:
                st.session_state.neo4j_manager = manager
                st.success(f"✅ {msg}")
                st.rerun()
            else:
                st.error(f"❌ {msg}")

        st.markdown("### 🍃 MongoDB Connection")
        # 云端 MongoDB Atlas 连接配置（密码中的 ! 已编码为 %21）
        mongodb_uri = st.text_input(
            "MongoDB URI",
            value="mongodb+srv://sifanxiang0627_db_user:1234567890@cluster0.s3uzp7i.mongodb.net/",
            key="mongodb_uri"
        )
        mongodb_db = st.text_input(
            "Database Name",
            value="wikimultihop_db",
            key="mongodb_db"
        )

        if st.button("🔗 Connect MongoDB", type="primary", use_container_width=True, key="connect_mongodb"):
            manager = MongoDBManager(mongodb_uri, mongodb_db)
            success, msg = manager.connect()
            if success:
                st.session_state.mongodb_manager = manager
                st.success(f"✅ {msg}")
                st.rerun()
            else:
                st.error(f"❌ {msg}")

        col1, col2 = st.columns(2)
        with col1:
            if st.session_state.neo4j_manager and st.button("🔌 Disconnect Neo4j", use_container_width=True):
                st.session_state.neo4j_manager.disconnect()
                st.session_state.neo4j_manager = None
                st.rerun()
        with col2:
            if st.session_state.mongodb_manager and st.button("🍃 Disconnect MongoDB", use_container_width=True):
                st.session_state.mongodb_manager.disconnect()
                st.session_state.mongodb_manager = None
                st.rerun()

        if st.session_state.neo4j_manager:
            st.markdown("---")
            st.markdown("### 📊 Neo4j Server Info")
            server_info = st.session_state.neo4j_manager.get_server_info()
            if server_info:
                st.info(f"""
                **Version:** {server_info.get('version', 'N/A')}
                **Edition:** {server_info.get('edition', 'N/A')}
                """)

        if st.session_state.mongodb_manager:
            st.markdown("---")
            st.markdown("### 📊 MongoDB Info")
            qa_count = st.session_state.mongodb_manager.get_qa_count()
            st.info(f"**QA Documents:** {qa_count}")

    if not st.session_state.neo4j_manager or not st.session_state.mongodb_manager:
        st.warning("⚠️ Please connect to both Neo4j and MongoDB to use the full system features")
        if not st.session_state.neo4j_manager:
            st.info("👈 Connect to Neo4j first")
        if not st.session_state.mongodb_manager:
            st.info("👈 Connect to MongoDB first")
        return

    neo4j = st.session_state.neo4j_manager
    mongodb = st.session_state.mongodb_manager

    st.markdown("---")
    databases = neo4j.get_databases()
    if databases:
        selected_db = st.selectbox("📁 Neo4j Graph Database", databases, key="db_selector")
        if selected_db != st.session_state.current_db:
            st.session_state.current_db = selected_db
            neo4j.set_database(selected_db)
            st.rerun()

    st.markdown("---")
    st.header("📊 Database Overview")

    neo4j_stats = neo4j.get_db_stats()
    if neo4j_stats:
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("❓ Question Nodes", neo4j_stats.get('question_count', 0))
        with col2:
            st.metric("📄 Document Nodes", neo4j_stats.get('document_count', 0))
        with col3:
            st.metric("🏷️ Entity Nodes", neo4j_stats.get('entity_count', 0))
        with col4:
            st.metric("🔗 Relationships", neo4j_stats.get('relationship_count', 0))

    mongodb_stats = mongodb.get_qa_count()
    if mongodb_stats:
        st.metric("📚 MongoDB QA Documents", mongodb_stats)

    tab1, tab2, tab3, tab4, tab5, tab6, tab7, tab8 = st.tabs([
        "🔍 Content Query", "🕸️ Graph Visualization", "🔗 Multi-Hop Path Query",
        "📝 CRUD Operations", "📈 Analytics", "🌐 Node Management", "📋 Cross-DB Query",
        "🔬 Clustering Analysis"
    ])

    # ==================== Tab 1: Content Query (MongoDB) ====================
    with tab1:
        st.header("🔍 Content Query & Exploration")
        st.caption("Query detailed question-answer content from MongoDB")

        query_type = st.radio(
            "Query Type",
            ["Search by Question Text", "Get Question by ID", "Browse All Questions", "Questions by Type"],
            horizontal=True
        )

        if query_type == "Search by Question Text":
            search_term = st.text_input("Enter search term", placeholder="e.g., Who founded Microsoft?")
            if st.button("🔍 Search", type="primary"):
                if search_term:
                    results = mongodb.search_questions(search_term)
                    if results:
                        st.success(f"Found {len(results)} questions")
                        for result in results:
                            with st.expander(f"📝 {result.get('question', 'N/A')[:100]}"):
                                st.json({
                                    'id': result.get('_id'),
                                    'question': result.get('question'),
                                    'answer': result.get('answer'),
                                    'type': result.get('type')
                                })
                    else:
                        st.info("No matching questions found")

        elif query_type == "Get Question by ID":
            qid = st.text_input("Enter Question ID", placeholder="e.g., 5a7b8c9d...")
            if st.button("🔍 Get Question", type="primary"):
                if qid:
                    result = mongodb.get_qa_by_id(qid)
                    if result:
                        st.success("Question found:")
                        st.json(result)

                        if 'context' in result:
                            st.subheader("📚 Full Context")
                            for ctx in result['context']:
                                st.markdown(f"**Document:** {ctx.get('title', 'Unknown')}")
                                st.markdown(f"**Sentences:** {ctx.get('sentences', [])}")
                                st.markdown("---")
                    else:
                        st.warning(f"No question found with ID: {qid}")

        elif query_type == "Browse All Questions":
            col1, col2 = st.columns([3, 1])
            with col1:
                docs_per_page = st.selectbox("Items per page", [10, 20, 50, 100], index=1, key="browse_per_page")
            with col2:
                page_num = st.number_input("Page", min_value=1, value=1, key="browse_page")

            skip = (page_num - 1) * docs_per_page
            docs, total = mongodb.get_all_qa_documents(limit=docs_per_page, skip=skip)

            if docs:
                st.info(f"Showing {len(docs)} of {total} documents (Page {page_num})")
                for doc in docs:
                    with st.expander(f"❓ {doc.get('question', 'N/A')[:100]}"):
                        st.json(doc)

        elif query_type == "Questions by Type":
            types = mongodb.get_type_distribution()
            if types:
                type_options = [t['type'] for t in types]
                selected_type = st.selectbox("Select Question Type", type_options)
                if selected_type:
                    results = mongodb.get_questions_by_type(selected_type, limit=20)
                    if results:
                        st.success(f"Found {len(results)} questions of type '{selected_type}'")
                        for result in results:
                            with st.expander(f"❓ {result.get('question', 'N/A')[:100]}"):
                                st.json({
                                    'id': result.get('_id'),
                                    'question': result.get('question'),
                                    'answer': result.get('answer')
                                })

    # ==================== Tab 2: Graph Visualization ====================
    with tab2:
        st.header("🕸️ Knowledge Graph Visualization")
        st.caption("Visual representation of Questions, Documents, and Entities with their relationships")

        col1, col2 = st.columns([3, 1])
        with col1:
            max_relations = st.slider("Maximum relationships to display", 50, 500, 150, key="graph_limit")
        with col2:
            if st.button("🔄 Refresh Graph", use_container_width=True):
                st.rerun()

        with st.spinner("Loading graph data..."):
            edges = neo4j.get_all_relationships_for_graph(limit=max_relations)

        if edges:
            st.info(f"📊 Displaying {len(edges)} relationships")

            fig = create_graph_visualization(edges)
            if fig:
                # 添加唯一 key 修复 removeChild 错误
                st.plotly_chart(fig, use_container_width=True, key=f"graph_viz_{len(edges)}_{int(time.time())}")

            st.markdown("---")
            st.markdown("#### 📊 Graph Statistics")

            unique_sources = set([e['source_name'] for e in edges])
            unique_targets = set([e['target_name'] for e in edges])

            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("Unique Source Nodes", len(unique_sources))
            with col2:
                st.metric("Unique Target Nodes", len(unique_targets))
            with col3:
                st.metric("Total Relationships", len(edges))

            st.markdown("#### 🎨 Legend")
            col1, col2, col3 = st.columns(3)
            with col1:
                st.markdown("🔴 **Red**: Question nodes")
            with col2:
                st.markdown("🟢 **Teal**: Document nodes")
            with col3:
                st.markdown("🔵 **Blue**: Entity nodes")
        else:
            st.info("No relationships found. Import data first using the import script.")

    # ==================== Tab 3: Multi-Hop Path Query ====================
    with tab3:
        st.header("🔗 Multi-Hop Path Query")
        st.caption("Discover connections through multiple hops in the knowledge graph")

        path_type = st.radio(
            "Path Query Type",
            ["Multi-Hop from Node", "Shortest Path between Entities", "Node Neighbors"],
            horizontal=True
        )

        if path_type == "Multi-Hop from Node":
            col1, col2, col3 = st.columns([2, 1, 1])
            with col1:
                node_type = st.selectbox("Node Type", ["Entity", "Document", "Question"])
                node_name = st.text_input("Node Name/ID", placeholder="e.g., Apple Inc. or document title")
            with col2:
                max_hops = st.slider("Max Hops", 1, 5, 3)
            with col3:
                st.write("")
                st.write("")
                if st.button("🔍 Find Paths", type="primary"):
                    if node_name:
                        with st.spinner("Searching for paths..."):
                            label_map = {"Entity": "Entity", "Document": "Document", "Question": "Question"}
                            node_label = label_map[node_type]

                            paths = neo4j.multi_hop_path_query(node_name, node_label, max_hops)
                            if paths:
                                st.subheader(f"📊 Found {len(paths)} paths")
                                for i, path in enumerate(paths[:20], 1):
                                    with st.expander(f"Path {i}: {path['length']} hops"):
                                        st.markdown("**Path:** " + " → ".join(path['nodes']))
                                        st.markdown("**Relationships:** " + " → ".join(path['relationships']))
                            else:
                                st.info("No paths found from this node")

        elif path_type == "Shortest Path between Entities":
            col1, col2 = st.columns(2)
            with col1:
                entity1 = st.text_input("Entity 1", placeholder="e.g., Apple Inc.")
            with col2:
                entity2 = st.text_input("Entity 2", placeholder="e.g., Steve Jobs")

            if st.button("🔍 Find Shortest Path", type="primary"):
                if entity1 and entity2:
                    with st.spinner("Finding shortest path..."):
                        path = neo4j.shortest_path_between_entities(entity1, entity2)
                        if path:
                            st.success(f"Shortest path length: {path['length']} hops")
                            st.markdown("**Path:** " + " → ".join(path['nodes']))
                            st.markdown("**Relationships:** " + " → ".join(path['relationships']))

                            path_edges = []
                            for i in range(len(path['nodes']) - 1):
                                path_edges.append({
                                    'source_name': path['nodes'][i].split(':')[1] if ':' in path['nodes'][i] else
                                    path['nodes'][i],
                                    'target_name': path['nodes'][i + 1].split(':')[1] if ':' in path['nodes'][
                                        i + 1] else path['nodes'][i + 1],
                                    'relationship_type': path['relationships'][i] if i < len(
                                        path['relationships']) else 'UNKNOWN',
                                    'source_label': 'Unknown',
                                    'target_label': 'Unknown'
                                })
                            if path_edges:
                                fig = create_graph_visualization(path_edges)
                                if fig:
                                    # 添加唯一 key 修复 removeChild 错误
                                    st.plotly_chart(fig, use_container_width=True, key=f"shortest_path_{entity1}_{entity2}_{int(time.time())}")
                        else:
                            st.warning(f"No path found between '{entity1}' and '{entity2}'")

        elif path_type == "Node Neighbors":
            node_type = st.selectbox("Node Type", ["Entity", "Document", "Question"], key="neighbor_type")
            node_name = st.text_input("Node Name/ID", placeholder="e.g., Apple Inc.")
            hop_count = st.slider("Number of Hops", 1, 3, 1)

            if st.button("🔍 Find Neighbors", type="primary"):
                if node_name:
                    with st.spinner("Finding neighbors..."):
                        label_map = {"Entity": "Entity", "Document": "Document", "Question": "Question"}
                        neighbors = neo4j.get_node_neighbors(label_map[node_type], node_name, hop_count)

                        if neighbors:
                            st.success(f"Found {len(neighbors)} neighbors within {hop_count} hop(s)")
                            neighbors_df = pd.DataFrame(neighbors)
                            st.dataframe(neighbors_df, use_container_width=True)
                        else:
                            st.info(f"No neighbors found for '{node_name}'")

    # ==================== Tab 4: CRUD Operations ====================
    with tab4:
        st.header("📝 CRUD Operations")
        st.caption("Create, Read, Update, Delete operations on both Neo4j and MongoDB")

        operation_type = st.radio(
            "Operation Type",
            ["Create Node", "Update Node", "Delete Node", "Create Relationship", "Delete Relationship"],
            horizontal=True
        )

        if operation_type == "Create Node":
            st.subheader("Create New Node in Neo4j")
            node_type = st.selectbox("Node Type", ["Question", "Document", "Entity"])

            if node_type == "Question":
                qid = st.text_input("Question ID")
                question_text = st.text_area("Question Text")
                if st.button("✅ Create Question"):
                    if qid and question_text:
                        success, msg = neo4j.create_question_node(qid, question_text)
                        if success:
                            st.success(f"✅ {msg}")
                        else:
                            st.error(f"❌ {msg}")

            elif node_type == "Document":
                doc_title = st.text_input("Document Title")
                if st.button("✅ Create Document"):
                    if doc_title:
                        success, msg = neo4j.create_document_node(doc_title)
                        if success:
                            st.success(f"✅ {msg}")
                        else:
                            st.error(f"❌ {msg}")

            elif node_type == "Entity":
                entity_name = st.text_input("Entity Name")
                if st.button("✅ Create Entity"):
                    if entity_name:
                        success, msg = neo4j.create_entity_node(entity_name)
                        if success:
                            st.success(f"✅ {msg}")
                        else:
                            st.error(f"❌ {msg}")

        elif operation_type == "Update Node":
            st.subheader("Update Node Properties")
            node_type = st.selectbox("Node Type", ["Question"], key="update_type")

            if node_type == "Question":
                qid = st.text_input("Question ID to Update")
                new_text = st.text_area("New Question Text")
                if st.button("✏️ Update Question"):
                    if qid and new_text:
                        success, msg = neo4j.update_question_text(qid, new_text)
                        if success:
                            st.success(f"✅ {msg}")
                        else:
                            st.error(f"❌ {msg}")

        elif operation_type == "Delete Node":
            st.subheader("Delete Node from Neo4j")
            st.warning("⚠️ Deleting a node will also delete all its relationships!")

            node_type = st.selectbox("Node Type", ["Question", "Document", "Entity"], key="delete_type")

            if node_type == "Question":
                qid = st.text_input("Question ID to Delete")
                confirm = st.text_input("Type 'DELETE' to confirm")
                if st.button("🗑️ Delete Question"):
                    if confirm == "DELETE" and qid:
                        success, msg = neo4j.delete_question(qid)
                        if success:
                            st.success(f"✅ {msg}")
                            st.rerun()
                        else:
                            st.error(f"❌ {msg}")

            elif node_type == "Document":
                doc_title = st.text_input("Document Title to Delete")
                confirm = st.text_input("Type 'DELETE' to confirm", key="confirm_doc")
                if st.button("🗑️ Delete Document"):
                    if confirm == "DELETE" and doc_title:
                        success, msg = neo4j.delete_document(doc_title)
                        if success:
                            st.success(f"✅ {msg}")
                            st.rerun()
                        else:
                            st.error(f"❌ {msg}")

            elif node_type == "Entity":
                entity_name = st.text_input("Entity Name to Delete")
                confirm = st.text_input("Type 'DELETE' to confirm", key="confirm_entity")
                if st.button("🗑️ Delete Entity"):
                    if confirm == "DELETE" and entity_name:
                        success, msg = neo4j.delete_entity(entity_name)
                        if success:
                            st.success(f"✅ {msg}")
                            st.rerun()
                        else:
                            st.error(f"❌ {msg}")

        elif operation_type == "Create Relationship":
            st.subheader("Create Relationship Between Nodes")

            col1, col2 = st.columns(2)
            with col1:
                from_label = st.selectbox("From Node Type", ["Question", "Document", "Entity"])
                if from_label == "Question":
                    from_prop = "id"
                    from_id = st.text_input("Question ID")
                elif from_label == "Document":
                    from_prop = "title"
                    from_id = st.text_input("Document Title")
                else:
                    from_prop = "name"
                    from_id = st.text_input("Entity Name")

            with col2:
                to_label = st.selectbox("To Node Type", ["Question", "Document", "Entity"])
                if to_label == "Question":
                    to_prop = "id"
                    to_id = st.text_input("Target Question ID")
                elif to_label == "Document":
                    to_prop = "title"
                    to_id = st.text_input("Target Document Title")
                else:
                    to_prop = "name"
                    to_id = st.text_input("Target Entity Name")

            rel_type = st.text_input("Relationship Type", placeholder="e.g., MENTIONS, CONTEXT, SUPPORTS, RELATION")

            if st.button("🔗 Create Relationship"):
                if from_id and to_id and rel_type:
                    success, msg = neo4j.create_relationship(
                        from_label, from_id, from_prop,
                        to_label, to_id, to_prop,
                        rel_type.upper()
                    )
                    if success:
                        st.success(f"✅ {msg}")
                    else:
                        st.error(f"❌ {msg}")

        elif operation_type == "Delete Relationship":
            st.subheader("Delete Relationship Between Nodes")

            col1, col2 = st.columns(2)
            with col1:
                from_label = st.selectbox("From Node Type", ["Question", "Document", "Entity"], key="del_from_label")
                if from_label == "Question":
                    from_prop = "id"
                    from_id = st.text_input("Question ID", key="del_from_id")
                elif from_label == "Document":
                    from_prop = "title"
                    from_id = st.text_input("Document Title", key="del_from_title")
                else:
                    from_prop = "name"
                    from_id = st.text_input("Entity Name", key="del_from_entity")

            with col2:
                to_label = st.selectbox("To Node Type", ["Question", "Document", "Entity"], key="del_to_label")
                if to_label == "Question":
                    to_prop = "id"
                    to_id = st.text_input("Target Question ID", key="del_to_id")
                elif to_label == "Document":
                    to_prop = "title"
                    to_id = st.text_input("Target Document Title", key="del_to_title")
                else:
                    to_prop = "name"
                    to_id = st.text_input("Target Entity Name", key="del_to_entity")

            rel_type = st.text_input("Relationship Type to Delete", placeholder="e.g., MENTIONS, CONTEXT",
                                     key="del_rel_type")
            confirm = st.text_input("Type 'DELETE' to confirm", key="del_rel_confirm")

            if st.button("❌ Delete Relationship", key="delete_rel_btn"):
                if confirm == "DELETE" and from_id and to_id and rel_type:
                    success, msg = neo4j.delete_relationship(
                        from_label, from_id, from_prop,
                        to_label, to_id, to_prop,
                        rel_type.upper()
                    )
                    if success:
                        st.success(f"✅ {msg}")
                        st.rerun()
                    else:
                        st.error(f"❌ {msg}")

    # ==================== Tab 5: Analytics ====================
    with tab5:
        st.header("📈 Data Analytics & Statistics")

        col1, col2 = st.columns(2)

        with col1:
            st.subheader("Question Type Distribution (MongoDB)")
            type_dist = mongodb.get_type_distribution()
            if type_dist:
                type_df = pd.DataFrame(type_dist)
                st.dataframe(type_df, use_container_width=True)
                fig = px.pie(type_df, values='count', names='type', title='Question Types')
                # 添加唯一 key
                st.plotly_chart(fig, use_container_width=True, key="type_dist_pie")
            else:
                st.info("No type distribution data available")

        with col2:
            st.subheader("Most Connected Entities (Neo4j)")
            top_entities = neo4j.get_most_connected_entities(limit=10)
            if top_entities:
                entity_df = pd.DataFrame(top_entities)
                st.dataframe(entity_df, use_container_width=True)
                fig = px.bar(entity_df, x='name', y='degree', title='Top Entities by Connection Degree')
                # 添加唯一 key
                st.plotly_chart(fig, use_container_width=True, key="top_entities_bar")
            else:
                st.info("No entity data available")

        st.markdown("---")

        col1, col2 = st.columns(2)

        with col1:
            st.subheader("Most Frequently Supported Documents")
            top_docs = mongodb.get_most_frequently_supported_docs(limit=10)
            if top_docs:
                doc_df = pd.DataFrame(top_docs)
                st.dataframe(doc_df, use_container_width=True)
                fig = px.bar(doc_df, x='title', y='count', title='Documents by Supporting Frequency')
                # 添加唯一 key
                st.plotly_chart(fig, use_container_width=True, key="docs_support_bar")
            else:
                st.info("No document data available")

        with col2:
            st.subheader("Average Supporting Facts per Question")
            avg_facts = mongodb.get_avg_supporting_facts_count()
            if avg_facts > 0:
                st.metric("Average Supporting Facts", f"{avg_facts:.2f}")
                st.info("Multi-hop complexity indicator: Higher average = more complex reasoning paths")
            else:
                st.info("No data available")

        st.markdown("---")

        st.subheader("Neo4j Database Statistics")
        neo4j_stats = neo4j.get_db_stats()
        if neo4j_stats:
            stats_df = pd.DataFrame([
                {"Node Type": "Question", "Count": neo4j_stats.get('question_count', 0)},
                {"Node Type": "Document", "Count": neo4j_stats.get('document_count', 0)},
                {"Node Type": "Entity", "Count": neo4j_stats.get('entity_count', 0)}
            ])
            st.dataframe(stats_df, use_container_width=True)
            fig = px.bar(stats_df, x='Node Type', y='Count', title='Node Distribution in Neo4j')
            # 添加唯一 key
            st.plotly_chart(fig, use_container_width=True, key="node_dist_bar")

    # ==================== Tab 6: Node Management ====================
    with tab6:
        st.header("🌐 Node Management")
        st.caption("Browse and manage all nodes in the knowledge graph")

        node_labels = neo4j.get_all_node_labels()

        if node_labels:
            selected_label = st.selectbox("Select Node Label", node_labels)

            if selected_label:
                nodes = neo4j.get_nodes_by_label(selected_label, limit=200)

                if nodes:
                    st.info(f"Found {len(nodes)} nodes with label '{selected_label}'")

                    nodes_data = []
                    for node in nodes:
                        node_info = {"id": node.get('id') or node.get('name') or node.get('title') or 'N/A'}
                        for key, value in node.items():
                            if key not in ['id', 'name', 'title', '_id'] and not key.startswith('_'):
                                node_info[key] = str(value)[:100]
                        nodes_data.append(node_info)

                    df = pd.DataFrame(nodes_data)
                    st.dataframe(df, use_container_width=True, height=400)

                    st.subheader("Node Detail View")
                    node_identifiers = []
                    for node in nodes:
                        identifier = node.get('id') or node.get('name') or node.get('title')
                        if identifier:
                            node_identifiers.append(identifier)

                    if node_identifiers:
                        selected_node_id = st.selectbox("Select node to view details", node_identifiers)

                        if selected_node_id:
                            for node in nodes:
                                node_identifier = node.get('id') or node.get('name') or node.get('title')
                                if node_identifier == selected_node_id:
                                    st.json(node)

                                    st.subheader("Connected Relationships")

                                    if 'id' in node:
                                        rels = neo4j.get_relationships_by_node_generic(selected_label, 'id',
                                                                                       selected_node_id)
                                    elif 'name' in node:
                                        rels = neo4j.get_relationships_by_node_generic(selected_label, 'name',
                                                                                       selected_node_id)
                                    elif 'title' in node:
                                        rels = neo4j.get_relationships_by_node_generic(selected_label, 'title',
                                                                                       selected_node_id)
                                    else:
                                        rels = []

                                    if rels:
                                        rels_df = pd.DataFrame(rels)
                                        st.dataframe(rels_df, use_container_width=True)
                                    else:
                                        st.info("No relationships found for this node")
                                    break
                else:
                    st.info(f"No nodes found with label '{selected_label}'")
        else:
            st.info("No node labels found in database")

    # ==================== Tab 7: Cross-Database Query ====================
    with tab7:
        st.header("📋 Cross-Database Integrated Query")
        st.caption("Query both Neo4j and MongoDB simultaneously to get complete information")

        st.subheader("Question Lookup with Full Context")

        qid_lookup = st.text_input("Enter Question ID", placeholder="e.g., 5a7b8c9d...", key="cross_qid")

        if st.button("🔍 Lookup Question", type="primary"):
            if qid_lookup:
                col1, col2 = st.columns(2)

                with col1:
                    st.markdown("### 🕸️ Neo4j Graph Data")
                    neo4j_question = neo4j.get_question_by_id(qid_lookup)
                    if neo4j_question:
                        st.success("Neo4j Data Found")
                        st.json(neo4j_question)
                    else:
                        st.warning("Question not found in Neo4j")

                with col2:
                    st.markdown("### 🍃 MongoDB Document Data")
                    mongodb_question = mongodb.get_qa_by_id(qid_lookup)
                    if mongodb_question:
                        st.success("MongoDB Data Found")
                        st.json(mongodb_question)
                    else:
                        st.warning("Question not found in MongoDB")

                st.markdown("---")

                if neo4j_question and mongodb_question:
                    st.subheader("📊 Combined Knowledge")

                    st.markdown(f"**Question:** {neo4j_question.get('text', 'N/A')}")
                    st.markdown(f"**Answer:** {mongodb_question.get('answer', 'N/A')}")
                    st.markdown(f"**Type:** {mongodb_question.get('type', 'N/A')}")

                    st.markdown("### 🔗 Connected Nodes in Knowledge Graph")
                    neighbors = neo4j.get_node_neighbors("Question", qid_lookup, hop=1)
                    if neighbors:
                        neighbors_df = pd.DataFrame(neighbors)
                        st.dataframe(neighbors_df, use_container_width=True)
                    else:
                        st.info("No connected nodes found")

                    if 'context' in mongodb_question:
                        st.markdown("### 📚 Full Document Context")
                        for ctx in mongodb_question['context']:
                            with st.expander(f"📄 {ctx.get('title', 'Unknown Document')}"):
                                sentences = ctx.get('sentences', [])
                                for i, sentence in enumerate(sentences):
                                    st.markdown(f"{i + 1}. {sentence}")

                    if 'supporting_facts' in mongodb_question:
                        st.markdown("### 🔍 Supporting Facts (Multi-hop Evidence)")
                        for fact in mongodb_question['supporting_facts']:
                            st.markdown(
                                f"- **Document:** {fact.get('title', 'Unknown')}, **Sentence ID:** {fact.get('sent_id', 'N/A')}")

    # ==================== Tab 8: Clustering Analysis ====================
    with tab8:
        st.header("🔬 Question Clustering Analysis")
        st.caption("Unsupervised clustering of questions using TF-IDF and K-Means")

        st.markdown("""
        ### 📊 What is Clustering?
        Clustering groups similar questions together based on their text content. 
        This helps discover common topics and patterns in the question set without manual labeling.
        """)

        col1, col2, col3 = st.columns([1, 1, 1])
        with col1:
            n_clusters = st.slider("Number of Clusters (K)", min_value=2, max_value=15, value=5,
                                   help="K in K-Means algorithm")
        with col2:
            max_samples = st.slider("Max Questions to Analyze", min_value=100, max_value=2000, value=500, step=100,
                                    help="More samples = slower but more accurate")
        with col3:
            if st.button("🎯 Run Clustering", type="primary", use_container_width=True):
                st.session_state.run_clustering = True

        if st.session_state.run_clustering:
            with st.spinner("Performing clustering analysis... This may take a moment."):
                clusters, keywords, viz_data, actual_clusters, status = mongodb.perform_question_clustering(
                    n_clusters=n_clusters, max_samples=max_samples
                )

                if status == "success" and clusters:
                    st.session_state.cluster_results = {
                        'clusters': clusters,
                        'keywords': keywords,
                        'viz_data': viz_data,
                        'actual_clusters': actual_clusters
                    }
                    st.success(
                        f"✅ Clustering completed! {len(viz_data)} questions grouped into {actual_clusters} clusters.")
                    st.balloons()
                else:
                    st.error(f"❌ Clustering failed: {status}")
                    st.session_state.run_clustering = False

        if st.session_state.cluster_results:
            results = st.session_state.cluster_results
            clusters = results['clusters']
            keywords = results['keywords']
            viz_data = results['viz_data']

            st.subheader("📊 Cluster Overview")

            cluster_summary = []
            for cluster in clusters:
                cluster_summary.append({
                    "Cluster ID": cluster['cluster_id'],
                    "Size": cluster['size'],
                    "Representative Question": cluster.get('representative', 'N/A')[:80] + "...",
                    "Top Keywords": ", ".join(keywords[cluster['cluster_id']]) if keywords and cluster[
                        'cluster_id'] < len(keywords) else "N/A"
                })

            summary_df = pd.DataFrame(cluster_summary)
            st.dataframe(summary_df, use_container_width=True)

            st.subheader("📈 Cluster Visualization (PCA 2D Projection)")
            st.caption("Each point represents a question. Colors indicate cluster membership.")

            if viz_data:
                viz_df = pd.DataFrame(viz_data)
                fig = px.scatter(
                    viz_df, x='x', y='y', color='cluster',
                    hover_data=['question', 'type', 'question_id'],
                    title="Question Clusters (PCA Projection)",
                    color_continuous_scale='Viridis',
                    labels={'cluster': 'Cluster ID', 'x': 'PCA Component 1', 'y': 'PCA Component 2'}
                )
                fig.update_traces(marker=dict(size=8, opacity=0.7))
                fig.update_layout(height=500)
                # 添加唯一 key
                st.plotly_chart(fig, use_container_width=True, key=f"pca_scatter_{n_clusters}_{max_samples}_{int(time.time())}")

            col1, col2 = st.columns(2)
            with col1:
                st.subheader("📊 Cluster Size Distribution")
                size_data = [{"Cluster": c['cluster_id'], "Size": c['size']} for c in clusters]
                size_df = pd.DataFrame(size_data)
                fig = px.bar(size_df, x='Cluster', y='Size', title="Questions per Cluster",
                             color='Size', color_continuous_scale='Blues')
                # 添加唯一 key
                st.plotly_chart(fig, use_container_width=True, key="cluster_size_bar")

            with col2:
                st.subheader("🏷️ Question Type Distribution by Cluster")
                type_cluster_data = []
                for point in viz_data:
                    type_cluster_data.append({"cluster": point['cluster'], "type": point['type']})
                type_df = pd.DataFrame(type_cluster_data)
                type_pivot = pd.crosstab(type_df['cluster'], type_df['type'])
                fig = px.bar(type_pivot, barmode='stack', title="Question Types by Cluster")
                # 添加唯一 key
                st.plotly_chart(fig, use_container_width=True, key="type_distribution_bar")

            st.subheader("🔍 Inspect Individual Clusters")
            selected_cluster = st.selectbox("Select Cluster to Explore", [c['cluster_id'] for c in clusters])

            if selected_cluster is not None:
                cluster_data = next((c for c in clusters if c['cluster_id'] == selected_cluster), None)
                if cluster_data:
                    st.markdown(f"### Cluster {selected_cluster} Details")
                    col1, col2 = st.columns(2)
                    with col1:
                        st.metric("Cluster Size", cluster_data['size'])
                    with col2:
                        st.metric("Percentage", f"{(cluster_data['size'] / len(viz_data) * 100):.1f}%")

                    st.markdown(
                        f"**Top Keywords:** {', '.join(keywords[selected_cluster]) if keywords and selected_cluster < len(keywords) else 'N/A'}")
                    st.markdown(f"**Representative Question:** {cluster_data.get('representative', 'N/A')}")

                    st.markdown("### 📝 Sample Questions in this Cluster")
                    samples = cluster_data['questions'][:10]
                    for i, q in enumerate(samples, 1):
                        with st.expander(f"Question {i}"):
                            st.markdown(f"**Question:** {q}")
                            if cluster_data['answers'] and i <= len(cluster_data['answers']):
                                st.markdown(f"**Answer:** {cluster_data['answers'][i - 1]}")
                            if cluster_data['types'] and i <= len(cluster_data['types']):
                                st.markdown(f"**Type:** {cluster_data['types'][i - 1]}")

                    if cluster_data['size'] > 10:
                        st.info(
                            f"Showing 10 of {cluster_data['size']} questions. Adjust the clustering parameters to analyze more.")

            st.markdown("---")
            st.subheader("📥 Export Clustering Results")

            col1, col2 = st.columns(2)
            with col1:
                csv_data = summary_df.to_csv(index=False)
                st.download_button(
                    label="📊 Export Cluster Summary as CSV",
                    data=csv_data,
                    file_name=f"cluster_summary_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                    mime="text/csv"
                )

            with col2:
                if st.button("🗑️ Clear Results"):
                    st.session_state.cluster_results = None
                    st.session_state.run_clustering = False
                    st.rerun()

            st.markdown("---")
            st.subheader("💡 Understanding the Clustering Results")
            st.info("""
            - **Cluster Size**: Number of questions in each cluster
            - **Top Keywords**: Most frequent words that define the cluster's theme
            - **PCA Visualization**: 2D projection showing how questions are grouped
            - **Question Type Distribution**: Shows which question types (bridge/comparison) appear in each cluster
            - **Sample Questions**: Example questions from the selected cluster
            """)


if __name__ == "__main__":
    main()