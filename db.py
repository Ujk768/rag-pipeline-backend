# DATABASE
import os
import psycopg2
from pgvector.psycopg2 import register_vector
from psycopg2.extras import execute_values
from dotenv import load_dotenv

load_dotenv()

DATA_BASE_URL = os.getenv("DATA_BASE_URL")
DB_WRITE_BATCH = 100

def get_db_connection():
    conn = psycopg2.connect(DATA_BASE_URL, sslmode="require", connect_timeout=5)
    register_vector(conn)
    return conn


def init_db():
    conn = psycopg2.connect(DATA_BASE_URL, sslmode="require", connect_timeout=5)
    cur = conn.cursor()
    cur.execute("CREATE EXTENSION IF NOT EXISTS vector;")
    conn.commit()
    register_vector(conn)
    cur.execute("""
        CREATE TABLE IF NOT EXISTS app_status (
            id SERIAL PRIMARY KEY,
            key TEXT UNIQUE,
            value TEXT,
            updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        );
    """)
    cur.execute("""
        INSERT INTO app_status (key, value)
        VALUES ('processing_status', '{"status": "idle", "chunks": 0, "error": null, "mode": null}')
        ON CONFLICT (key) DO NOTHING;
    """)
    cur.execute("""
        CREATE TABLE IF NOT EXISTS document_chunks (
            id           SERIAL PRIMARY KEY,
            page_number  INTEGER,
            content      TEXT,
            embedding    vector(384)
        );
    """)
    conn.commit()
    cur.close()
    conn.close()
    print("[INFO] Database initialized.")


# PDF PROCESSING
def _insert_rows_batched(cur, rows: list[tuple], batch_size: int = DB_WRITE_BATCH):
    for i in range(0, len(rows), batch_size):
        execute_values(
            cur,
            "INSERT INTO document_chunks (page_number, content, embedding) VALUES %s",
            rows[i:i + batch_size],
        )
        cur.connection.commit()


def clear_existing_data():
    # Clear existing data
    print("[INFO] Clearing existing document data...")
    conn = get_db_connection()
    cur = conn.cursor()
    if has_stored_data():
        cur.execute("DELETE FROM document_chunks;")
        conn.commit()
    cur.close()
    conn.close()
    print("[INFO] Database cleared.")


def has_stored_data() -> bool:
    try:
        conn = get_db_connection()
        cur = conn.cursor()
        cur.execute("SELECT EXISTS (SELECT 1 FROM document_chunks LIMIT 1);")
        exists = cur.fetchone()[0]
        cur.close()
        conn.close()
        return exists
    except Exception as e:
        print(f"[ERROR] Database check failed: {e}")
        return False
