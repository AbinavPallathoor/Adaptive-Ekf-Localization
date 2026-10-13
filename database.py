"""SQLite recording storage; all writes use transactions."""

import csv
from datetime import datetime, timezone
import json
from pathlib import Path
import sqlite3

from simulation import FIELDS


def now():
    return datetime.now(timezone.utc).isoformat()


class RunStore:
    def __init__(self, path):
        self.path = Path(path).resolve()
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.conn = sqlite3.connect(self.path)
        self.conn.row_factory = sqlite3.Row
        self.conn.execute('PRAGMA foreign_keys=ON')
        self.conn.execute('PRAGMA journal_mode=WAL')
        columns = ', '.join(f'{f} {"TEXT" if f == "wall_time" else "REAL"} NOT NULL' for f in FIELDS)
        self.conn.executescript(f'''
            CREATE TABLE IF NOT EXISTS runs (
                id INTEGER PRIMARY KEY, name TEXT NOT NULL,
                started_at TEXT NOT NULL, ended_at TEXT,
                metadata_json TEXT NOT NULL, sample_count INTEGER NOT NULL DEFAULT 0
            );
            CREATE TABLE IF NOT EXISTS samples (
                run_id INTEGER NOT NULL REFERENCES runs(id),
                sequence INTEGER NOT NULL, {columns},
                PRIMARY KEY (run_id, sequence)
            );
            PRAGMA user_version=1;
        ''')

    def start_run(self, name, metadata):
        with self.conn:
            cursor = self.conn.execute(
                'INSERT INTO runs(name, started_at, metadata_json) VALUES(?,?,?)',
                (name.strip() or 'Untitled recording', now(), json.dumps(metadata)))
        return cursor.lastrowid

    def append(self, run_id, samples):
        if not samples:
            return
        with self.conn:
            run = self.conn.execute('SELECT * FROM runs WHERE id=?', (run_id,)).fetchone()
            if run is None or run['ended_at'] is not None:
                raise ValueError('Run does not exist or is already finished')
            start = run['sample_count']
            rows = [(run_id, start+i, *(s[f] for f in FIELDS)) for i, s in enumerate(samples)]
            placeholders = ','.join('?' for _ in range(len(FIELDS)+2))
            self.conn.executemany(f'INSERT INTO samples VALUES({placeholders})', rows)
            self.conn.execute('UPDATE runs SET sample_count=sample_count+? WHERE id=?',
                              (len(samples), run_id))

    def finish_run(self, run_id):
        with self.conn:
            self.conn.execute('UPDATE runs SET ended_at=? WHERE id=? AND ended_at IS NULL',
                              (now(), run_id))

    def list_runs(self):
        return [dict(row) for row in self.conn.execute('SELECT * FROM runs ORDER BY id DESC')]

    def read_samples(self, run_id, limit=100):
        rows = self.conn.execute(
            'SELECT * FROM samples WHERE run_id=? ORDER BY sequence DESC LIMIT ?',
            (run_id, limit)).fetchall()
        return [dict(row) for row in reversed(rows)]

    def export_csv(self, run_id, path):
        if self.conn.execute('SELECT id FROM runs WHERE id=?', (run_id,)).fetchone() is None:
            raise ValueError('Run not found')
        count = 0
        with Path(path).open('w', newline='') as out:
            writer = csv.writer(out)
            writer.writerow(['run_id', 'sequence', *FIELDS])
            for row in self.conn.execute('SELECT * FROM samples WHERE run_id=? ORDER BY sequence', (run_id,)):
                writer.writerow(tuple(row))
                count += 1
        return count

    def close(self):
        self.conn.close()
