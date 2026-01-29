from sqlalchemy import create_engine
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker
import os
import shutil

_env_url = os.getenv("DATABASE_URL")
if _env_url:
    SQLALCHEMY_DATABASE_URL = _env_url
else:
    def _candidate_db_paths() -> list[str]:
        backend_dir = os.path.dirname(os.path.abspath(__file__))
        project_dir = os.path.abspath(os.path.join(backend_dir, os.pardir))
        data_dir = os.path.join(project_dir, "data")
        canonical = os.path.join(data_dir, "lottery.db")
        legacy_root = os.path.join(project_dir, "lottery.db")
        legacy_backend = os.path.join(backend_dir, "lottery.db")

        return [canonical, legacy_root, legacy_backend]

    db_path = os.getenv("DB_PATH")
    if not db_path:
        candidates = _candidate_db_paths()
        canonical = candidates[0]
        existing: list[str] = []
        for p in candidates:
            try:
                if os.path.exists(p) and os.path.getsize(p) > 0:
                    existing.append(p)
            except Exception:
                continue

        def _db_score(path: str) -> int:
            try:
                import sqlite3

                con = sqlite3.connect(path)
                cur = con.cursor()
                cur.execute("SELECT name FROM sqlite_master WHERE type='table'")
                tables = {r[0] for r in cur.fetchall()}

                def _count(t: str) -> int:
                    if t not in tables:
                        return 0
                    cur.execute(f"SELECT COUNT(*) FROM {t}")
                    n = cur.fetchone()
                    return int(n[0] or 0) if n else 0

                score = (
                    _count("prediction_records") * 1_000_000
                    + _count("ai_generation_logs") * 1_000
                    + _count("lottery_records") * 10
                    + _count("app_settings")
                )
                con.close()
                return int(score)
            except Exception:
                return 0

        if existing:
            best = max(existing, key=_db_score)
            best_score = _db_score(best)
            canonical_score = _db_score(canonical) if canonical in existing else 0
            try:
                os.makedirs(os.path.dirname(canonical), exist_ok=True)
            except Exception:
                pass
            if best != canonical and best_score > canonical_score:
                try:
                    shutil.copy2(best, canonical)
                    db_path = canonical
                except Exception:
                    db_path = best
            else:
                db_path = canonical if canonical in existing else best
        else:
            try:
                os.makedirs(os.path.dirname(canonical), exist_ok=True)
            except Exception:
                pass
            db_path = canonical

    if db_path.startswith("sqlite:"):
        SQLALCHEMY_DATABASE_URL = db_path
    elif os.path.isabs(db_path):
        SQLALCHEMY_DATABASE_URL = f"sqlite:///{db_path}"
    else:
        SQLALCHEMY_DATABASE_URL = f"sqlite:///./{db_path.lstrip('./')}"

engine = create_engine(
    SQLALCHEMY_DATABASE_URL, connect_args={"check_same_thread": False}
)
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)

Base = declarative_base()
