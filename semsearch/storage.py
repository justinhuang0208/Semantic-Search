from __future__ import annotations

import json
import sqlite3
from collections import defaultdict
from pathlib import Path

from .models import ChunkDraft, DocumentRecord


class Storage:
    def __init__(self, db_path: Path) -> None:
        self.db_path = db_path
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self.conn = sqlite3.connect(self.db_path)
        self.conn.row_factory = sqlite3.Row
        self.conn.execute("PRAGMA foreign_keys = ON")

    def close(self) -> None:
        self.conn.close()

    def create_schema(self) -> None:
        self.conn.executescript(
            """
            CREATE TABLE IF NOT EXISTS documents (
                doc_id TEXT PRIMARY KEY,
                title TEXT NOT NULL,
                source_path TEXT NOT NULL,
                tags_json TEXT NOT NULL,
                out_links_json TEXT NOT NULL,
                updated_at TEXT NOT NULL,
                content_hash TEXT NOT NULL,
                char_count INTEGER NOT NULL
            );

            CREATE TABLE IF NOT EXISTS chunks (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                chunk_id TEXT UNIQUE NOT NULL,
                doc_id TEXT NOT NULL,
                title TEXT NOT NULL,
                source_path TEXT NOT NULL,
                section_path TEXT NOT NULL,
                chunk_type TEXT NOT NULL,
                context_prefix TEXT NOT NULL,
                text TEXT NOT NULL,
                token_count INTEGER NOT NULL,
                content_hash TEXT NOT NULL,
                tags_json TEXT NOT NULL,
                out_links_json TEXT NOT NULL,
                updated_at TEXT NOT NULL,
                FOREIGN KEY (doc_id) REFERENCES documents(doc_id) ON DELETE CASCADE
            );

            CREATE INDEX IF NOT EXISTS idx_chunks_doc_id ON chunks(doc_id);
            CREATE INDEX IF NOT EXISTS idx_chunks_type ON chunks(chunk_type);
            CREATE INDEX IF NOT EXISTS idx_chunks_content_hash ON chunks(content_hash);

            CREATE TABLE IF NOT EXISTS bm25_terms (
                chunk_rowid INTEGER NOT NULL,
                term TEXT NOT NULL,
                tf INTEGER NOT NULL,
                PRIMARY KEY (chunk_rowid, term),
                FOREIGN KEY (chunk_rowid) REFERENCES chunks(id) ON DELETE CASCADE
            );

            CREATE INDEX IF NOT EXISTS idx_bm25_term ON bm25_terms(term);

            CREATE TABLE IF NOT EXISTS bm25_stats (
                term TEXT PRIMARY KEY,
                df INTEGER NOT NULL
            );

            CREATE TABLE IF NOT EXISTS bm25_meta (
                key TEXT PRIMARY KEY,
                value REAL NOT NULL
            );
            """
        )
        self.conn.commit()

    def clear_for_rebuild(self) -> None:
        self.conn.executescript(
            """
            DELETE FROM bm25_terms;
            DELETE FROM bm25_stats;
            DELETE FROM bm25_meta;
            DELETE FROM chunks;
            DELETE FROM documents;
            """
        )
        self.conn.commit()

    def insert_document(self, doc: DocumentRecord) -> None:
        self.conn.execute(
            """
            INSERT OR REPLACE INTO documents
            (doc_id, title, source_path, tags_json, out_links_json, updated_at, content_hash, char_count)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                doc.doc_id,
                doc.title,
                doc.source_path,
                json.dumps(doc.tags, ensure_ascii=False),
                json.dumps(doc.out_links, ensure_ascii=False),
                doc.updated_at,
                doc.content_hash,
                doc.char_count,
            ),
        )

    def insert_chunk(self, chunk: ChunkDraft) -> int:
        cursor = self.conn.execute(
            """
            INSERT INTO chunks
            (chunk_id, doc_id, title, source_path, section_path, chunk_type, context_prefix, text,
             token_count, content_hash, tags_json, out_links_json, updated_at)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                chunk.chunk_id,
                chunk.doc_id,
                chunk.title,
                chunk.source_path,
                chunk.section_path,
                chunk.chunk_type,
                chunk.context_prefix,
                chunk.text,
                chunk.token_count,
                chunk.content_hash,
                json.dumps(chunk.tags, ensure_ascii=False),
                json.dumps(chunk.out_links, ensure_ascii=False),
                chunk.updated_at,
            ),
        )
        return int(cursor.lastrowid)

    def insert_bm25_terms(self, chunk_rowid: int, term_freq: dict[str, int]) -> None:
        rows = [(chunk_rowid, term, tf) for term, tf in term_freq.items()]
        self.conn.executemany(
            "INSERT INTO bm25_terms (chunk_rowid, term, tf) VALUES (?, ?, ?)",
            rows,
        )

    def finalize_bm25(
        self,
        doc_count: int,
        avgdl: float,
        term_df: dict[str, int],
    ) -> None:
        self.conn.executemany(
            "INSERT OR REPLACE INTO bm25_stats (term, df) VALUES (?, ?)",
            [(term, df) for term, df in term_df.items()],
        )
        self.conn.executemany(
            "INSERT OR REPLACE INTO bm25_meta (key, value) VALUES (?, ?)",
            [("doc_count", float(doc_count)), ("avgdl", float(avgdl))],
        )

    def commit(self) -> None:
        self.conn.commit()

    def counts(self) -> tuple[int, int]:
        doc_count = self.conn.execute("SELECT COUNT(*) FROM documents").fetchone()[0]
        chunk_count = self.conn.execute("SELECT COUNT(*) FROM chunks").fetchone()[0]
        return int(doc_count), int(chunk_count)

    def bm25_globals(self) -> tuple[int, float]:
        rows = self.conn.execute("SELECT key, value FROM bm25_meta").fetchall()
        mapping = {row["key"]: row["value"] for row in rows}
        return int(mapping.get("doc_count", 0)), float(mapping.get("avgdl", 0.0))

    def bm25_postings(self, terms: list[str]) -> list[sqlite3.Row]:
        if not terms:
            return []
        placeholders = ",".join("?" for _ in terms)
        sql = f"""
            SELECT
                t.chunk_rowid,
                t.term,
                t.tf,
                s.df,
                c.token_count AS dl
            FROM bm25_terms t
            JOIN bm25_stats s ON s.term = t.term
            JOIN chunks c ON c.id = t.chunk_rowid
            WHERE t.term IN ({placeholders})
        """
        return self.conn.execute(sql, terms).fetchall()

    def chunks_by_ids(self, chunk_ids: list[int]) -> dict[int, sqlite3.Row]:
        if not chunk_ids:
            return {}
        placeholders = ",".join("?" for _ in chunk_ids)
        rows = self.conn.execute(
            f"SELECT * FROM chunks WHERE id IN ({placeholders})",
            chunk_ids,
        ).fetchall()
        return {int(row["id"]): row for row in rows}

    def compute_term_df(self) -> dict[str, int]:
        rows = self.conn.execute(
            "SELECT term, COUNT(*) AS df FROM bm25_terms GROUP BY term"
        ).fetchall()
        return {str(row["term"]): int(row["df"]) for row in rows}

    def token_lengths(self) -> list[int]:
        rows = self.conn.execute("SELECT token_count FROM chunks").fetchall()
        return [int(row["token_count"]) for row in rows]

    def all_docs_grouped_by_source(self) -> dict[str, list[str]]:
        rows = self.conn.execute(
            "SELECT doc_id, source_path FROM documents ORDER BY source_path"
        ).fetchall()
        grouped: dict[str, list[str]] = defaultdict(list)
        for row in rows:
            grouped[str(row["source_path"])].append(str(row["doc_id"]))
        return dict(grouped)
