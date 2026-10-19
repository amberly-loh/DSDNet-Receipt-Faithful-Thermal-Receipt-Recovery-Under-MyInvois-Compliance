from __future__ import annotations

import json
import sqlite3
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


SCHEMA = """
PRAGMA foreign_keys = ON;
CREATE TABLE IF NOT EXISTS users (
    id INTEGER PRIMARY KEY,
    public_id TEXT NOT NULL UNIQUE,
    name TEXT NOT NULL,
    email TEXT NOT NULL UNIQUE COLLATE NOCASE,
    password_hash TEXT NOT NULL,
    created_at TEXT NOT NULL,
    role TEXT NOT NULL DEFAULT 'user',
    status TEXT NOT NULL DEFAULT 'pending',
    last_login_at TEXT,
    failed_login_count INTEGER NOT NULL DEFAULT 0,
    locked_until TEXT
);
CREATE TABLE IF NOT EXISTS receipts (
    id INTEGER PRIMARY KEY,
    public_id TEXT NOT NULL UNIQUE,
    user_id INTEGER NOT NULL REFERENCES users(id) ON DELETE CASCADE,
    source_name TEXT NOT NULL,
    created_at TEXT NOT NULL,
    updated_at TEXT NOT NULL,
    status TEXT NOT NULL DEFAULT 'needs_review',
    merchant TEXT,
    transaction_date TEXT,
    receipt_number TEXT,
    total_amount TEXT,
    tags TEXT NOT NULL DEFAULT '[]',
    notes TEXT NOT NULL DEFAULT '',
    ocr_text TEXT NOT NULL DEFAULT '',
    audit_json TEXT NOT NULL,
    input_path TEXT NOT NULL,
    restored_path TEXT NOT NULL,
    runtime_seconds REAL NOT NULL,
    restoration_status TEXT NOT NULL,
    support_status TEXT NOT NULL,
    deleted_at TEXT,
    retention_until TEXT,
    deletion_verified_at TEXT,
    content_sha256 TEXT
);
CREATE INDEX IF NOT EXISTS idx_receipts_user_created ON receipts(user_id, created_at DESC);
CREATE INDEX IF NOT EXISTS idx_receipts_user_status ON receipts(user_id, status);
CREATE TABLE IF NOT EXISTS audit_events (
    id INTEGER PRIMARY KEY,
    event_id TEXT NOT NULL UNIQUE,
    occurred_at TEXT NOT NULL,
    actor_user_id INTEGER REFERENCES users(id) ON DELETE SET NULL,
    action TEXT NOT NULL,
    object_type TEXT NOT NULL,
    object_id TEXT,
    outcome TEXT NOT NULL,
    details_json TEXT NOT NULL DEFAULT '{}'
);
CREATE INDEX IF NOT EXISTS idx_audit_events_time ON audit_events(occurred_at DESC);
CREATE INDEX IF NOT EXISTS idx_audit_events_actor ON audit_events(actor_user_id, occurred_at DESC);
"""


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


class ReceiptStore:
    def __init__(self, database: str | Path) -> None:
        self.database = Path(database)
        self.database.parent.mkdir(parents=True, exist_ok=True)
        with self.connect() as db:
            db.executescript(SCHEMA)
            self._migrate(db)

    @staticmethod
    def _migrate(db: sqlite3.Connection) -> None:
        user_columns = {row["name"] for row in db.execute("PRAGMA table_info(users)")}
        receipt_columns = {row["name"] for row in db.execute("PRAGMA table_info(receipts)")}
        for name, definition in {
            "role": "TEXT NOT NULL DEFAULT 'user'", "status": "TEXT NOT NULL DEFAULT 'pending'",
            "last_login_at": "TEXT", "failed_login_count": "INTEGER NOT NULL DEFAULT 0", "locked_until": "TEXT",
        }.items():
            if name not in user_columns:
                db.execute(f"ALTER TABLE users ADD COLUMN {name} {definition}")
        for name, definition in {
            "retention_until": "TEXT", "deletion_verified_at": "TEXT", "content_sha256": "TEXT",
        }.items():
            if name not in receipt_columns:
                db.execute(f"ALTER TABLE receipts ADD COLUMN {name} {definition}")
        first = db.execute("SELECT id FROM users ORDER BY id LIMIT 1").fetchone()
        if first and not db.execute("SELECT 1 FROM users WHERE role='admin'").fetchone():
            db.execute("UPDATE users SET role='admin', status='active' WHERE id=?", (first["id"],))

    def connect(self) -> sqlite3.Connection:
        db = sqlite3.connect(self.database, timeout=20)
        db.row_factory = sqlite3.Row
        db.execute("PRAGMA foreign_keys = ON")
        return db

    def create_user(self, public_id: str, name: str, email: str, password_hash: str) -> dict[str, Any]:
        with self.connect() as db:
            first_user = db.execute("SELECT COUNT(*) FROM users").fetchone()[0] == 0
            cursor = db.execute(
                "INSERT INTO users(public_id,name,email,password_hash,created_at,role,status) VALUES(?,?,?,?,?,?,?)",
                (public_id, name.strip(), email.strip().lower(), password_hash, utc_now(), "admin" if first_user else "user", "active" if first_user else "pending"),
            )
            return dict(db.execute("SELECT id,public_id,name,email,created_at,role,status FROM users WHERE id=?", (cursor.lastrowid,)).fetchone())

    def user_by_email(self, email: str):
        with self.connect() as db:
            row = db.execute("SELECT * FROM users WHERE email=? COLLATE NOCASE", (email.strip(),)).fetchone()
            return dict(row) if row else None

    def user_by_id(self, user_id: int):
        with self.connect() as db:
            row = db.execute("SELECT id,public_id,name,email,created_at,role,status,last_login_at FROM users WHERE id=?", (user_id,)).fetchone()
            return dict(row) if row else None

    def list_users(self) -> list[dict[str, Any]]:
        with self.connect() as db:
            rows = db.execute("SELECT id,public_id,name,email,created_at,role,status,last_login_at FROM users ORDER BY created_at DESC").fetchall()
        return [dict(row) for row in rows]

    def update_user(self, public_id: str, status: str | None = None, role: str | None = None):
        values = {}
        if status in {"pending", "active", "suspended"}: values["status"] = status
        if role in {"user", "admin"}: values["role"] = role
        if not values: return None
        with self.connect() as db:
            db.execute(f"UPDATE users SET {','.join(f'{key}=?' for key in values)} WHERE public_id=?", (*values.values(), public_id))
            row = db.execute("SELECT id,public_id,name,email,created_at,role,status,last_login_at FROM users WHERE public_id=?", (public_id,)).fetchone()
        return dict(row) if row else None

    def record_login(self, user_id: int, success: bool) -> None:
        with self.connect() as db:
            if success:
                db.execute("UPDATE users SET last_login_at=?,failed_login_count=0,locked_until=NULL WHERE id=?", (utc_now(), user_id))
            else:
                db.execute("UPDATE users SET failed_login_count=failed_login_count+1 WHERE id=?", (user_id,))

    def log_event(self, actor_user_id: int | None, action: str, object_type: str, object_id: str | None = None, outcome: str = "success", details: dict[str, Any] | None = None) -> None:
        import uuid
        with self.connect() as db:
            db.execute("INSERT INTO audit_events(event_id,occurred_at,actor_user_id,action,object_type,object_id,outcome,details_json) VALUES(?,?,?,?,?,?,?,?)", (uuid.uuid4().hex, utc_now(), actor_user_id, action, object_type, object_id, outcome, json.dumps(details or {}, ensure_ascii=False)))

    def list_events(self, limit: int = 250) -> list[dict[str, Any]]:
        with self.connect() as db:
            rows = db.execute("SELECT e.event_id,e.occurred_at,e.action,e.object_type,e.object_id,e.outcome,e.details_json,u.name actor_name FROM audit_events e LEFT JOIN users u ON u.id=e.actor_user_id ORDER BY e.id DESC LIMIT ?", (limit,)).fetchall()
        result = []
        for row in rows:
            item = dict(row); item["details"] = json.loads(item.pop("details_json") or "{}"); result.append(item)
        return result

    def admin_stats(self) -> dict[str, int]:
        with self.connect() as db:
            users = db.execute("SELECT COUNT(*) FROM users").fetchone()[0]
            pending = db.execute("SELECT COUNT(*) FROM users WHERE status='pending'").fetchone()[0]
            receipts = db.execute("SELECT COUNT(*) FROM receipts WHERE deleted_at IS NULL").fetchone()[0]
            deleted = db.execute("SELECT COUNT(*) FROM receipts WHERE deleted_at IS NOT NULL").fetchone()[0]
        return {"users": users, "pending_users": pending, "receipts": receipts, "deleted_receipts": deleted}

    def create_receipt(self, values: dict[str, Any]) -> dict[str, Any]:
        keys = list(values)
        with self.connect() as db:
            cursor = db.execute(
                f"INSERT INTO receipts({','.join(keys)}) VALUES({','.join('?' for _ in keys)})",
                tuple(values[k] for k in keys),
            )
            row = db.execute("SELECT * FROM receipts WHERE id=?", (cursor.lastrowid,)).fetchone()
        return self.present(dict(row))

    def list_receipts(self, user_id: int, query: str = "", status: str = "all") -> list[dict[str, Any]]:
        clauses = ["user_id=?", "deleted_at IS NULL"]
        args: list[Any] = [user_id]
        if status != "all":
            clauses.append("status=?"); args.append(status)
        if query:
            clauses.append("(source_name LIKE ? OR COALESCE(merchant,'') LIKE ? OR COALESCE(receipt_number,'') LIKE ? OR COALESCE(total_amount,'') LIKE ? OR tags LIKE ?)")
            term = f"%{query.strip()}%"; args.extend([term] * 5)
        with self.connect() as db:
            rows = db.execute(f"SELECT * FROM receipts WHERE {' AND '.join(clauses)} ORDER BY created_at DESC", args).fetchall()
        return [self.present(dict(row), compact=True) for row in rows]

    def get_receipt(self, user_id: int, public_id: str, include_deleted: bool = False):
        suffix = "" if include_deleted else " AND deleted_at IS NULL"
        with self.connect() as db:
            row = db.execute(f"SELECT * FROM receipts WHERE user_id=? AND public_id=?{suffix}", (user_id, public_id)).fetchone()
        return self.present(dict(row)) if row else None

    def update_receipt(self, user_id: int, public_id: str, values: dict[str, Any]):
        allowed = {"merchant", "transaction_date", "receipt_number", "total_amount", "status", "notes", "tags"}
        clean = {k: v for k, v in values.items() if k in allowed}
        if "tags" in clean:
            clean["tags"] = json.dumps(clean["tags"] if isinstance(clean["tags"], list) else [])
        clean["updated_at"] = utc_now()
        with self.connect() as db:
            db.execute(f"UPDATE receipts SET {','.join(f'{k}=?' for k in clean)} WHERE user_id=? AND public_id=? AND deleted_at IS NULL", (*clean.values(), user_id, public_id))
        return self.get_receipt(user_id, public_id)

    def delete_receipt(self, user_id: int, public_id: str) -> bool:
        with self.connect() as db:
            now = utc_now()
            cursor = db.execute("UPDATE receipts SET deleted_at=?, deletion_verified_at=?, updated_at=? WHERE user_id=? AND public_id=? AND deleted_at IS NULL", (now, now, now, user_id, public_id))
            return cursor.rowcount == 1

    def all_receipts(self) -> list[dict[str, Any]]:
        with self.connect() as db:
            rows = db.execute("SELECT r.*,u.name owner_name,u.email owner_email FROM receipts r JOIN users u ON u.id=r.user_id WHERE r.deleted_at IS NULL ORDER BY r.created_at DESC").fetchall()
        return [self.present(dict(row), compact=True) for row in rows]

    def stats(self, user_id: int) -> dict[str, int]:
        with self.connect() as db:
            rows = db.execute("SELECT status,COUNT(*) count FROM receipts WHERE user_id=? AND deleted_at IS NULL GROUP BY status", (user_id,)).fetchall()
        values = {"total": 0, "needs_review": 0, "reviewed": 0, "archived": 0}
        for row in rows:
            values[row["status"]] = row["count"]; values["total"] += row["count"]
        return values

    @staticmethod
    def present(row: dict[str, Any], compact: bool = False) -> dict[str, Any]:
        row["tags"] = json.loads(row.get("tags") or "[]")
        row["fields"] = {
            "merchant": row.get("merchant"), "transaction_date": row.get("transaction_date"),
            "receipt_number": row.get("receipt_number"), "total_amount": row.get("total_amount"),
        }
        row["original_url"] = f"/api/receipts/{row['public_id']}/image/original"
        row["restored_url"] = f"/api/receipts/{row['public_id']}/image/restored"
        if compact:
            for key in ("audit_json", "ocr_text", "input_path", "restored_path", "deleted_at", "user_id", "id"):
                row.pop(key, None)
        else:
            row["audit"] = json.loads(row.pop("audit_json"))
            row.pop("user_id", None); row.pop("id", None); row.pop("deleted_at", None)
        return row
