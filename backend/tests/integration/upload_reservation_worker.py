"""Independent OS process worker for testing upload reservation concurrency and quota limits."""

import json
import sys
from pathlib import Path
from typing import Any

from sqlalchemy import create_engine, event
from sqlalchemy.orm import sessionmaker

from backend.app.db.repositories.upload_reservation_repository import (
    QuarantineQuotaExceededError,
    SqliteUploadReservationRepository,
    UploadConcurrencyLimitExceededError,
)
from backend.app.db.session import _set_sqlite_pragmas


def main() -> None:
    if len(sys.argv) < 8:
        sys.exit(1)

    db_url = sys.argv[1]
    quarantine_dir_str = sys.argv[2]
    req_bytes = int(sys.argv[3])
    max_concurrent = int(sys.argv[4])
    quota_bytes = int(sys.argv[5])
    out_file_str = sys.argv[6]
    worker_id = int(sys.argv[7])

    engine = create_engine(db_url, echo=False)
    event.listen(engine, "connect", _set_sqlite_pragmas)
    session_factory = sessionmaker(bind=engine)
    session = session_factory()
    quarantine_dir = Path(quarantine_dir_str)
    out_file = Path(out_file_str)

    res_data: dict[str, Any] = {"worker_id": worker_id}
    try:
        repo = SqliteUploadReservationRepository(session)
        res = repo.acquire_reservation(
            asset_type="model",
            requested_bytes=req_bytes,
            quarantine_dir=quarantine_dir,
            max_concurrent_uploads=max_concurrent,
            global_quarantine_quota_bytes=quota_bytes,
            timeout_seconds=300,
        )
        res_data["success"] = True
        res_data["reservation_id"] = res.reservation_id
        res_data["reserved_bytes"] = res.reserved_bytes
    except UploadConcurrencyLimitExceededError as err:
        res_data["success"] = False
        res_data["error"] = "CONCURRENCY_LIMIT_EXCEEDED"
        res_data["message"] = str(err)
    except QuarantineQuotaExceededError as err:
        res_data["success"] = False
        res_data["error"] = "QUOTA_EXCEEDED"
        res_data["message"] = str(err)
    except Exception as err:
        res_data["success"] = False
        res_data["error"] = type(err).__name__
        res_data["message"] = str(err)
    finally:
        session.close()
        engine.dispose()
        out_file.write_text(json.dumps(res_data), encoding="utf-8")


if __name__ == "__main__":
    main()
