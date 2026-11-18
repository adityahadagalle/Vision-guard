"""Independent OS process worker script for model ingestion concurrency tests.

Executed via:
    python -m backend.tests.integration.model_ingestion_worker <db_url> <worker_id> <out_json_path> <mode>
"""

import json
import sys
from pathlib import Path

from sqlalchemy import create_engine, event
from sqlalchemy.orm import sessionmaker

from backend.app.db.repositories.job_repository import SqliteJobQueue
from backend.app.db.session import _set_sqlite_pragmas


def main() -> None:
    if len(sys.argv) < 5:
        sys.exit(1)

    db_url = sys.argv[1]
    worker_id = sys.argv[2]
    out_file = Path(sys.argv[3])
    mode = sys.argv[4]

    engine = create_engine(db_url, echo=False)
    event.listen(engine, "connect", _set_sqlite_pragmas)
    session_factory = sessionmaker(bind=engine)
    session = session_factory()

    try:
        if mode == "claim_all":
            queue = SqliteJobQueue(session)
            claims = []
            while True:
                job = queue.claim_next_job(worker_id=worker_id, lease_duration_seconds=300)
                if not job:
                    break
                claims.append(
                    {
                        "job_id": job["job_id"],
                        "task_type": job["task_type"],
                        "lease_token": job["lease_token"],
                        "lease_generation": job["lease_generation"],
                    }
                )
            out_file.write_text(
                json.dumps({"worker_id": worker_id, "claims": claims}), encoding="utf-8"
            )

    finally:
        session.close()
        engine.dispose()


if __name__ == "__main__":
    main()
