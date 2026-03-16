"""
Standalone script to export knowledge graph from existing knowledge.db.
Run when you have knowledge.db but don't want to re-run the full pipeline.

  python export_graph.py
"""
from pathlib import Path

import sqlite3

from patent_agent_pipeline import DB_PATH, GRAPH_JSON_PATH, export_graph_json, _copy_to_frontend

if __name__ == "__main__":
    if not DB_PATH.exists():
        print(f"[fail] {DB_PATH} not found. Run patent_agent_pipeline first.")
        exit(1)
    conn = sqlite3.connect(DB_PATH)
    export_graph_json(conn, GRAPH_JSON_PATH)
    conn.close()
    print(f"[ok] {GRAPH_JSON_PATH}")
    _copy_to_frontend(GRAPH_JSON_PATH)
