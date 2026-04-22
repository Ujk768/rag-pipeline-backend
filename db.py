# DATABASE
import os
import json
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


# def init_db():
#     conn = psycopg2.connect(DATA_BASE_URL, sslmode="require", connect_timeout=5)
#     cur = conn.cursor()

#     cur.execute("CREATE EXTENSION IF NOT EXISTS vector;")
#     conn.commit()
#     register_vector(conn)

#     # Stores one row per uploaded document
#     cur.execute("""
#         CREATE TABLE IF NOT EXISTS document_info (
#             id              SERIAL PRIMARY KEY,
#             document_name   TEXT        NOT NULL,
#             total_pages     INTEGER     NOT NULL,
#             uploaded_at     TIMESTAMPTZ NOT NULL DEFAULT NOW()
#         );
#     """)

#     # # Stores persistent key/value app state (e.g. processing_status)
#     # cur.execute("""
#     #     CREATE TABLE IF NOT EXISTS app_status (
#     #         key   TEXT PRIMARY KEY,
#     #         value TEXT NOT NULL
#     #     );
#     # """)

#     # cur.execute("""
#     #     INSERT INTO app_status (key, value)
#     #     VALUES ('processing_status', %s)
#     #     ON CONFLICT (key) DO NOTHING;
#     # """, (json.dumps({"status": "idle", "chunks": 0, "error": None, "mode": None}),))

#     cur.execute("""
#         CREATE TABLE IF NOT EXISTS document_chunks (
#             id           SERIAL  PRIMARY KEY,
#             page_number  INTEGER NOT NULL,
#             content      TEXT    NOT NULL,
#             embedding    vector(3072),
#             pruned       BOOLEAN NOT NULL DEFAULT FALSE
#         );
#     """)

#     conn.commit()
#     cur.close()
#     conn.close()
#     print("[INFO] Database initialized.")


# DOCUMENT INFO
def upsert_document_info(document_name: str, total_pages: int):
    """
    Replaces whatever document info is stored — only one document
    is active at a time, matching the single-document chunk model.
    """
    conn = get_db_connection()
    cur = conn.cursor()
    cur.execute("DELETE FROM document_info;")
    cur.execute(
        "INSERT INTO document_info (document_name, total_pages) VALUES (%s, %s);",
        (document_name, total_pages),
    )
    conn.commit()
    cur.close()
    conn.close()


def get_document_info() -> dict | None:
    """Returns the stored document metadata, or None if no document is loaded."""
    try:
        conn = get_db_connection()
        cur = conn.cursor()
        cur.execute("""
            SELECT document_name, total_pages, uploaded_at
            FROM document_info
            LIMIT 1;
        """)
        row = cur.fetchone()
        cur.close()
        conn.close()
        if not row:
            return None
        return {
            "document_name": row[0],
            "total_pages":   row[1],
            "uploaded_at":   row[2].isoformat(),
        }
    except Exception as e:
        print(f"[ERROR] get_document_info failed: {e}")
        return None


# CHUNK HELPERS
def _insert_rows_batched(conn, rows: list[tuple], batch_size: int = DB_WRITE_BATCH):
    """Rows must be (page_number, content, embedding) tuples. Commits once at the end."""
    cur = conn.cursor()
    for i in range(0, len(rows), batch_size):
        execute_values(
            cur,
            "INSERT INTO document_chunks (page_number, content, embedding) VALUES %s",
            rows[i:i + batch_size],
        )
    conn.commit()
    cur.close()

def clear_existing_data():
    """Hard-deletes all chunk rows and document_info for a fresh upload."""
    print("[INFO] Clearing existing document data...")
    conn = get_db_connection()
    cur = conn.cursor()
    cur.execute("DELETE FROM document_chunks;")
    cur.execute("DELETE FROM document_info;")
    conn.commit()
    cur.close()
    conn.close()
    print("[INFO] Database cleared.")


def has_stored_data() -> bool:
    """True if there is at least one active (non-pruned) chunk in the DB."""
    try:
        conn = get_db_connection()
        cur = conn.cursor()
        cur.execute("SELECT EXISTS (SELECT 1 FROM document_chunks WHERE pruned = FALSE LIMIT 1);")
        exists = cur.fetchone()[0]
        cur.close()
        conn.close()
        return exists
    except Exception as e:
        print(f"[ERROR] Database check failed: {e}")
        return False

# def count_active_chunks() -> int:
#     conn = get_db_connection()
#     cur = conn.cursor()
#     cur.execute("SELECT COUNT(*) FROM document_chunks WHERE pruned = FALSE;")
#     n = cur.fetchone()[0]
#     cur.close()
#     conn.close()
#     return n


def init_db():
    conn = psycopg2.connect(DATA_BASE_URL, sslmode="require", connect_timeout=5)
    cur = conn.cursor()

    cur.execute("CREATE EXTENSION IF NOT EXISTS vector;")
    conn.commit()
    register_vector(conn)

    cur.execute("""
        CREATE TABLE IF NOT EXISTS document_info (
            id              SERIAL PRIMARY KEY,
            document_name   TEXT        NOT NULL,
            total_pages     INTEGER     NOT NULL,
            uploaded_at     TIMESTAMPTZ NOT NULL DEFAULT NOW()
        );
    """)
    # Source of truth — never touched after upload
    cur.execute("""
        CREATE TABLE IF NOT EXISTS document_chunks (
            id           SERIAL  PRIMARY KEY,
            page_number  INTEGER NOT NULL,
            content      TEXT    NOT NULL,
            embedding    vector(1024)
        );
    """)

    # Materialized pruning result — replaced on every prune call
    cur.execute("""
        CREATE TABLE IF NOT EXISTS document_chunks_pruned (
            id              SERIAL  PRIMARY KEY,
            source_chunk_id INTEGER NOT NULL REFERENCES document_chunks(id) ON DELETE CASCADE,
            page_number     INTEGER NOT NULL,
            content         TEXT    NOT NULL,
            embedding       vector(1024),
            strategy        TEXT    NOT NULL,
            pruned_at       TIMESTAMPTZ NOT NULL DEFAULT NOW()
        );
    """)

    conn.commit()
    cur.close()
    conn.close()
    print("[INFO] Database initialized.")


def clear_pruned_chunks():
    """Wipe the pruning result table — called before every new prune run."""
    conn = get_db_connection()
    cur = conn.cursor()
    cur.execute("DELETE FROM document_chunks_pruned;")
    conn.commit()
    cur.close()
    conn.close()


def has_pruned_data() -> bool:
    try:
        conn = get_db_connection()
        cur = conn.cursor()
        cur.execute("SELECT EXISTS (SELECT 1 FROM document_chunks_pruned LIMIT 1);")
        exists = cur.fetchone()[0]
        cur.close()
        conn.close()
        return exists
    except Exception as e:
        print(f"[ERROR] has_pruned_data check failed: {e}")
        return False


def get_active_pruning_strategy() -> str | None:
    """Returns the strategy name stored in document_chunks_pruned, or None."""
    try:
        conn = get_db_connection()
        cur = conn.cursor()
        cur.execute("SELECT DISTINCT strategy FROM document_chunks_pruned LIMIT 1;")
        row = cur.fetchone()
        cur.close()
        conn.close()
        return row[0] if row else None
    except Exception as e:
        print(f"[ERROR] get_active_pruning_strategy failed: {e}")
        return None


def insert_pruned_rows_batched(conn, rows: list[tuple], batch_size: int = DB_WRITE_BATCH):
    """Rows must be (source_chunk_id, page_number, content, embedding, strategy) tuples."""
    cur = conn.cursor()
    for i in range(0, len(rows), batch_size):
        execute_values(
            cur,
            """
            INSERT INTO document_chunks_pruned
                (source_chunk_id, page_number, content, embedding, strategy)
            VALUES %s
            """,
            rows[i:i + batch_size],
        )
    conn.commit()
    cur.close()


def count_active_chunks_from_pruned() -> int:
    """Count from pruned table if a pruning run exists, else from main table."""
    conn = get_db_connection()
    cur = conn.cursor()
    table = "document_chunks_pruned" if has_pruned_data() else "document_chunks"
    cur.execute(f"SELECT COUNT(*) FROM {table};")
    n = cur.fetchone()[0]
    cur.close()
    conn.close()
    return n


def clear_existing_data():
    """Full reset on new upload — clears both tables and document_info."""
    print("[INFO] Clearing existing document data...")
    conn = get_db_connection()
    cur = conn.cursor()
    cur.execute("DELETE FROM document_chunks_pruned;")
    cur.execute("DELETE FROM document_chunks;")
    cur.execute("DELETE FROM document_info;")
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