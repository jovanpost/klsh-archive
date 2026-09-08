#!/usr/bin/env python3
"""
Weekly archiver for the shared Supabase free-tier database.

Exports old rows to Parquet on Google Drive, verifies the file,
then reclaims space in Postgres.

Two bots share this database:
  depth_* / paper_*  -> jovanpost/klsh-15m   (15-minute depth collector)
  tm_*               -> jovanpost/dt-klsh-bot (Kalshi mentions no-fade bot)

Safety: only tables listed in PLAN below are ever touched. Anything not
in PLAN is invisible to this script. Deletes are gated on a successful
readback of the Parquet file.
"""

import io
import json
import os
import sys
from datetime import datetime, timedelta, timezone

import pandas as pd
import psycopg
import pyarrow as pa
import pyarrow.parquet as pq
from google.oauth2 import service_account
from googleapiclient.discovery import build
from googleapiclient.http import MediaIoBaseUpload

# --------------------------------------------------------------------------
# CONFIG — the allowlist. Nothing outside this dict is ever read or deleted.
# --------------------------------------------------------------------------
#
# mode:
#   "backup"     -> copy to Drive, delete nothing. For data you need live.
#   "prune"      -> copy to Drive, then DELETE rows older than keep_days.
#   "null_json"  -> copy to Drive, then set the heavy JSON columns to NULL
#                   on old rows. Keeps the numeric time series live forever.
#
# time_col: the timestamp column used to decide what counts as "old".
# json_cols: only used by mode="null_json".

PLAN = {
    # ---- klsh-15m (depth collector) ----
    "depth_book_sample": {
        "mode": "prune",
        "time_col": "created_at",
        "keep_days": 14,
        "owner": "klsh-15m",
    },
    "depth_minute": {
        "mode": "null_json",
        "time_col": "minute_ts",
        "keep_days": 21,
        "json_cols": ["yes_book", "no_book"],
        "owner": "klsh-15m",
    },
    "depth_sample": {
        "mode": "backup",
        "time_col": "ts",
        "owner": "klsh-15m",
    },
    "depth_settlement": {
        "mode": "backup",
        "time_col": None,  # tiny, dump whole table
        "owner": "klsh-15m",
    },
    "paper_upcont": {
        "mode": "backup",
        "time_col": None,
        "owner": "klsh-15m",
    },
    # ---- dt-klsh-bot (mentions) ----
    "tm_ticks": {
        "mode": "prune",
        "time_col": "ts",
        "keep_days": 14,
        "owner": "dt-klsh-bot",
    },
    "tm_log": {
        "mode": "prune",
        "time_col": "ts",
        "keep_days": 14,
        "owner": "dt-klsh-bot",
    },
    "tm_orders": {
        "mode": "backup",
        "time_col": None,
        "owner": "dt-klsh-bot",
    },
    "tm_events": {
        "mode": "backup",
        "time_col": None,
        "owner": "dt-klsh-bot",
    },
    "tm_series": {
        "mode": "backup",
        "time_col": None,
        "owner": "dt-klsh-bot",
    },
    "tm_state": {
        "mode": "backup",
        "time_col": None,
        "owner": "dt-klsh-bot",
    },
}

# Rows pulled per chunk. Keeps any single query short so the pooler
# does not kill it, and makes a failed run resumable.
CHUNK_ROWS = 50_000

DRY_RUN = os.environ.get("DRY_RUN", "true").lower() == "true"
DATABASE_URL = os.environ["DATABASE_URL"]
DRIVE_FOLDER_ID = os.environ["GDRIVE_FOLDER_ID"]

RUN_TAG = datetime.now(timezone.utc).strftime("%Y%m%d")


def log(msg):
    print(f"[{datetime.now(timezone.utc).strftime('%H:%M:%S')}] {msg}", flush=True)


# --------------------------------------------------------------------------
# Google Drive
# --------------------------------------------------------------------------

def drive_client():
    info = json.loads(os.environ["GDRIVE_SA_JSON"])
    creds = service_account.Credentials.from_service_account_info(
        info, scopes=["https://www.googleapis.com/auth/drive"]
    )
    return build("drive", "v3", credentials=creds, cache_discovery=False)


def ensure_folder(svc, name, parent):
    """Find or create a subfolder. Returns its id."""
    q = (
        f"name = '{name}' and '{parent}' in parents "
        "and mimeType = 'application/vnd.google-apps.folder' and trashed = false"
    )
    hits = svc.files().list(
        q=q, fields="files(id)", supportsAllDrives=True,
        includeItemsFromAllDrives=True,
    ).execute().get("files", [])
    if hits:
        return hits[0]["id"]
    meta = {
        "name": name,
        "mimeType": "application/vnd.google-apps.folder",
        "parents": [parent],
    }
    return svc.files().create(
        body=meta, fields="id", supportsAllDrives=True
    ).execute()["id"]


def upload_parquet(svc, folder_id, filename, buf):
    buf.seek(0)
    media = MediaIoBaseUpload(
        buf, mimetype="application/octet-stream", resumable=True
    )
    meta = {"name": filename, "parents": [folder_id]}
    f = svc.files().create(
        body=meta, media_body=media, fields="id,size", supportsAllDrives=True
    ).execute()
    return f["id"], int(f.get("size", 0))


def download_parquet(svc, file_id):
    data = svc.files().get_media(fileId=file_id, supportsAllDrives=True).execute()
    return pq.read_table(io.BytesIO(data))


# --------------------------------------------------------------------------
# Postgres
# --------------------------------------------------------------------------

def preflight(conn):
    """Verify every table and column in PLAN actually exists. Abort if not."""
    with conn.cursor() as cur:
        cur.execute(
            "select table_name, column_name from information_schema.columns "
            "where table_schema = 'public'"
        )
        cols = {}
        for t, c in cur.fetchall():
            cols.setdefault(t, set()).add(c)

    problems = []
    for table, cfg in PLAN.items():
        if table not in cols:
            problems.append(f"table '{table}' does not exist")
            continue
        tc = cfg.get("time_col")
        if tc and tc not in cols[table]:
            problems.append(
                f"{table}: time_col '{tc}' not found. "
                f"available: {sorted(cols[table])}"
            )
        for jc in cfg.get("json_cols", []):
            if jc not in cols[table]:
                problems.append(f"{table}: json_col '{jc}' not found")

    if problems:
        log("PREFLIGHT FAILED — fix PLAN in archive.py:")
        for p in problems:
            log(f"  - {p}")
        sys.exit(1)
    log("preflight ok — all tables and columns in PLAN exist")


def cutoff_for(cfg):
    if cfg.get("time_col") is None:
        return None
    return datetime.now(timezone.utc) - timedelta(days=cfg["keep_days"]) \
        if "keep_days" in cfg else None


def fetch_chunks(conn, table, cfg):
    """Yield DataFrames. For backup mode with no cutoff, dumps whole table."""
    tc = cfg.get("time_col")
    cut = cutoff_for(cfg)

    if cut is not None:
        where = f"where {tc} < %s"
        params = (cut,)
        order = f"order by {tc}"
    elif tc is not None:
        where, params, order = "", (), f"order by {tc}"
    else:
        where, params, order = "", (), ""

    offset = 0
    while True:
        sql = f"select * from {table} {where} {order} limit {CHUNK_ROWS} offset {offset}"
        df = pd.read_sql(sql, conn, params=params if params else None)
        if df.empty:
            break
        yield df
        if len(df) < CHUNK_ROWS:
            break
        offset += CHUNK_ROWS


def count_target(conn, table, cfg):
    cut = cutoff_for(cfg)
    with conn.cursor() as cur:
        if cut is None:
            cur.execute(f"select count(*) from {table}")
        else:
            cur.execute(
                f"select count(*) from {table} where {cfg['time_col']} < %s", (cut,)
            )
        return cur.fetchone()[0]


# --------------------------------------------------------------------------
# Main
# --------------------------------------------------------------------------

def process_table(conn, svc, root, table, cfg):
    n = count_target(conn, table, cfg)
    log(f"{table} [{cfg['mode']}, {cfg['owner']}] — {n:,} rows in scope")
    if n == 0:
        return

    frames = list(fetch_chunks(conn, table, cfg))
    if not frames:
        return
    df = pd.concat(frames, ignore_index=True)

    # jsonb comes back as dict/list; Parquet wants a stable type, so store
    # those columns as JSON text. duckdb can parse them back on read.
    for col in df.columns:
        if df[col].map(lambda v: isinstance(v, (dict, list))).any():
            df[col] = df[col].map(
                lambda v: json.dumps(v) if isinstance(v, (dict, list)) else v
            )

    buf = io.BytesIO()
    pq.write_table(pa.Table.from_pandas(df), buf, compression="zstd")
    size_mb = buf.tell() / 1e6
    log(f"  parquet built: {len(df):,} rows, {size_mb:.1f} MB")

    if DRY_RUN:
        log("  DRY_RUN — not uploading, not deleting")
        return

    folder = ensure_folder(svc, table, root)
    fname = f"{table}_{RUN_TAG}.parquet"
    fid, _ = upload_parquet(svc, folder, fname, buf)
    log(f"  uploaded {fname}")

    # Verify: read it back off Drive and confirm the row count.
    back = download_parquet(svc, fid)
    if back.num_rows != len(df):
        log(f"  VERIFY FAILED ({back.num_rows} != {len(df)}) — skipping delete")
        return
    log("  verified")

    if cfg["mode"] == "backup":
        return

    cut = cutoff_for(cfg)
    with conn.cursor() as cur:
        if cfg["mode"] == "prune":
            cur.execute(
                f"delete from {table} where {cfg['time_col']} < %s", (cut,)
            )
            log(f"  deleted {cur.rowcount:,} rows")
        elif cfg["mode"] == "null_json":
            sets = ", ".join(f"{c} = null" for c in cfg["json_cols"])
            guard = " or ".join(f"{c} is not null" for c in cfg["json_cols"])
            cur.execute(
                f"update {table} set {sets} "
                f"where {cfg['time_col']} < %s and ({guard})", (cut,)
            )
            log(f"  nulled JSON on {cur.rowcount:,} rows")
    conn.commit()

    # Reclaim space. Plain VACUUM, not FULL — FULL takes an exclusive lock
    # and would stall both bots.
    conn.autocommit = True
    with conn.cursor() as cur:
        cur.execute(f"vacuum analyze {table}")
    conn.autocommit = False
    log("  vacuumed")


def main():
    log(f"start — DRY_RUN={DRY_RUN}")
    svc = drive_client()
    root = ensure_folder(svc, f"week_{RUN_TAG}", DRIVE_FOLDER_ID) \
        if not DRY_RUN else None

    with psycopg.connect(DATABASE_URL, connect_timeout=30) as conn:
        preflight(conn)

        with conn.cursor() as cur:
            cur.execute("select pg_size_pretty(pg_database_size(current_database()))")
            log(f"database size before: {cur.fetchone()[0]}")

        for table, cfg in PLAN.items():
            try:
                process_table(conn, svc, root, table, cfg)
            except Exception as e:
                log(f"  ERROR on {table}: {e}")
                conn.rollback()

        with conn.cursor() as cur:
            cur.execute("select pg_size_pretty(pg_database_size(current_database()))")
            log(f"database size after: {cur.fetchone()[0]}")

    log("done")


if __name__ == "__main__":
    main()
