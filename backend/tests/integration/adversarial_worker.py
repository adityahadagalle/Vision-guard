"""Adversarial worker module for independent OS process concurrency testing."""

import json
import sys
from pathlib import Path

from sqlalchemy import create_engine, event
from sqlalchemy.orm import sessionmaker

from backend.app.db.repositories.job_repository import SqliteJobQueue
from backend.app.db.session import _set_sqlite_pragmas


def main() -> None:
    if len(sys.argv) < 4:
        sys.exit(1)

    db_path = sys.argv[1]
    worker_id = sys.argv[2]
    out_file = sys.argv[3]

    eng = create_engine(f"sqlite:///{db_path}", echo=False)
    event.listen(eng, "connect", _set_sqlite_pragmas)
    session_factory = sessionmaker(bind=eng)
    session = session_factory()
    queue = SqliteJobQueue(session)

    claimed: list[str] = []
    while True:
        job = queue.claim_next_job(worker_id=worker_id, lease_duration_seconds=300)
        if not job:
            break
        claimed.append(job["job_id"])

    session.close()
    eng.dispose()

    Path(out_file).write_text(json.dumps(claimed), encoding="utf-8")


if __name__ == "__main__":
    main()
