from fastapi import FastAPI, Depends, HTTPException, Request, status, Form
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse, RedirectResponse, StreamingResponse
from starlette.middleware.sessions import SessionMiddleware
from sqlalchemy.orm import Session
from sqlalchemy import text
from pydantic import BaseModel
from typing import Optional
from datetime import datetime, timedelta, time as dtime, timezone
from zoneinfo import ZoneInfo
from urllib.parse import urlparse, urlunparse
from email.utils import parsedate_to_datetime
import asyncio
import base64
import hashlib
import json
import logging
import os
import signal
import sys
import subprocess
import shutil
import threading




# Configure logging
logging.basicConfig(
    stream=sys.stdout,
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)

import secrets
import time
import requests

# Add current directory to sys.path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

import models
import database
import scraper
import predictor

logger = logging.getLogger(__name__)

_llm_status: dict[str, dict] = {"ssq": {}, "dlt": {}}
_time_offset_seconds: int = 0
_time_offset_source: str = ""
_time_offset_updated_at: Optional[datetime] = None

def _load_time_offset_from_db(db: Session):
    global _time_offset_seconds, _time_offset_source, _time_offset_updated_at
    raw = get_setting(db, "time_offset_seconds")
    try:
        _time_offset_seconds = int(str(raw).strip()) if raw is not None else 0
    except Exception:
        _time_offset_seconds = 0
    _time_offset_source = (get_setting(db, "time_offset_source") or "").strip()
    ts = (get_setting(db, "time_offset_updated_at") or "").strip()
    try:
        _time_offset_updated_at = datetime.fromisoformat(ts) if ts else None
    except Exception:
        _time_offset_updated_at = None

def get_now_cn() -> datetime:
    base = datetime.now(LOTTERY_TZ)
    if _time_offset_seconds:
        return base + timedelta(seconds=_time_offset_seconds)
    return base

def _dt_to_utc_iso(dt: Optional[datetime]) -> Optional[str]:
    if not dt:
        return None
    try:
        if dt.tzinfo is None:
            return dt.replace(tzinfo=timezone.utc).isoformat()
        return dt.astimezone(timezone.utc).isoformat()
    except Exception:
        try:
            return dt.isoformat()
        except Exception:
            return None

def ensure_schema():
    with database.engine.begin() as conn:
        old_exists = conn.execute(
            text("SELECT name FROM sqlite_master WHERE type='table' AND name='lottery_records_old'")
        ).fetchone()
        exists = conn.execute(
            text("SELECT name FROM sqlite_master WHERE type='table' AND name='lottery_records'")
        ).fetchone()
        if not exists:
            models.Base.metadata.create_all(bind=conn)
            return

        if old_exists:
            cols = conn.execute(text("PRAGMA table_info(lottery_records)")).fetchall()
            col_names = {c[1] for c in cols}
            required = {
                "id",
                "lottery_type",
                "issue",
                "date",
                "red_balls",
                "blue_balls",
                "sales",
                "pool",
            }
            if required.issubset(col_names):
                conn.execute(
                    text(
                        "INSERT OR IGNORE INTO lottery_records (id, lottery_type, issue, date, red_balls, blue_balls, sales, pool) "
                        "SELECT id, lottery_type, issue, date, red_balls, blue_balls, sales, pool FROM lottery_records_old"
                    )
                )
                conn.execute(text("DROP TABLE lottery_records_old"))

        cols = conn.execute(text("PRAGMA table_info(lottery_records)")).fetchall()
        col_names = {c[1] for c in cols}
        if "lottery_type" not in col_names:
            models.Base.metadata.create_all(bind=conn)
            return

        uniq_issue_only = False
        for idx in conn.execute(text("PRAGMA index_list(lottery_records)")).fetchall():
            idx_name = idx[1]
            is_unique = bool(idx[2])
            if not is_unique:
                continue
            idx_cols = conn.execute(text(f"PRAGMA index_info({idx_name})")).fetchall()
            idx_col_names = [c[2] for c in idx_cols]
            if idx_col_names == ["issue"]:
                uniq_issue_only = True
                break

        if uniq_issue_only:
            conn.execute(text("ALTER TABLE lottery_records RENAME TO lottery_records_old"))
            models.Base.metadata.create_all(bind=conn)
            old_cols = conn.execute(text("PRAGMA table_info(lottery_records_old)")).fetchall()
            old_col_names = {c[1] for c in old_cols}
            copy_cols = [
                "id",
                "lottery_type",
                "issue",
                "date",
                "red_balls",
                "blue_balls",
                "sales",
                "pool",
            ]
            if not set(copy_cols).issubset(old_col_names):
                conn.execute(text("DROP TABLE lottery_records"))
                conn.execute(text("ALTER TABLE lottery_records_old RENAME TO lottery_records"))
                return
            conn.execute(
                text(
                    "INSERT INTO lottery_records (id, lottery_type, issue, date, red_balls, blue_balls, sales, pool) "
                    "SELECT id, lottery_type, issue, date, red_balls, blue_balls, sales, pool FROM lottery_records_old"
                )
            )
            conn.execute(text("DROP TABLE lottery_records_old"))
            return

        models.Base.metadata.create_all(bind=conn)

        pred_exists = conn.execute(
            text("SELECT name FROM sqlite_master WHERE type='table' AND name='prediction_records'")
        ).fetchone()
        if pred_exists:
            pred_cols = conn.execute(text("PRAGMA table_info(prediction_records)")).fetchall()
            pred_col_names = {c[1] for c in pred_cols}
            if "used_llm" not in pred_col_names:
                conn.execute(text("ALTER TABLE prediction_records ADD COLUMN used_llm BOOLEAN DEFAULT 0"))
            if "llm_model" not in pred_col_names:
                conn.execute(text("ALTER TABLE prediction_records ADD COLUMN llm_model VARCHAR"))
            if "llm_latency_ms" not in pred_col_names:
                conn.execute(text("ALTER TABLE prediction_records ADD COLUMN llm_latency_ms INTEGER"))

ensure_schema()

app = FastAPI(title="魔力彩票助手 API")

# Disable caching for all responses
@app.middleware("http")
async def add_no_cache_header(request: Request, call_next):
    response = await call_next(request)
    response.headers["Cache-Control"] = "no-cache, no-store, must-revalidate"
    response.headers["Pragma"] = "no-cache"
    response.headers["Expires"] = "0"
    return response

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.add_middleware(
    SessionMiddleware,
    secret_key=os.getenv("SESSION_SECRET", "dev-secret"),
    same_site="lax",
    https_only=False,
)

def get_setting(db: Session, key: str) -> Optional[str]:
    s = db.query(models.AppSettings).filter_by(key=key).first()
    return s.value if s else None

def set_setting(db: Session, key: str, value: str):
    s = db.query(models.AppSettings).filter_by(key=key).first()
    if s:
        s.value = value
    else:
        db.add(models.AppSettings(key=key, value=value))
    db.commit()

def hash_password(password: str, salt: Optional[bytes] = None, iterations: int = 120_000) -> str:
    import secrets as _secrets
    if salt is None:
        salt = _secrets.token_bytes(16)
    dk = hashlib.pbkdf2_hmac("sha256", password.encode("utf-8"), salt, iterations)
    salt_b64 = base64.urlsafe_b64encode(salt).decode("utf-8").rstrip("=")
    dk_b64 = base64.urlsafe_b64encode(dk).decode("utf-8").rstrip("=")
    return f"pbkdf2_sha256${iterations}${salt_b64}${dk_b64}"

def verify_password(password: str, stored: str) -> bool:
    import secrets as _secrets
    try:
        algo, iterations_s, salt_b64, dk_b64 = stored.split("$", 3)
        if algo != "pbkdf2_sha256":
            return False
        iterations = int(iterations_s)
        salt = base64.urlsafe_b64decode(salt_b64 + "==")
        expected = base64.urlsafe_b64decode(dk_b64 + "==")
        actual = hashlib.pbkdf2_hmac("sha256", password.encode("utf-8"), salt, iterations)
        return _secrets.compare_digest(actual, expected)
    except Exception:
        return False

def ensure_admin_password_hash(db: Session) -> str:
    stored = get_setting(db, "admin_password_hash")
    changed = get_setting(db, "admin_password_changed")
    initial = os.getenv("ADMIN_PASSWORD", "admin")
    if stored:
        if changed == "1":
            return stored
        if verify_password(initial, stored):
            return stored
        stored = hash_password(initial)
        set_setting(db, "admin_password_hash", stored)
        return stored
    stored = hash_password(initial)
    set_setting(db, "admin_password_hash", stored)
    return stored

def parse_basic_auth(request: Request) -> Optional[tuple[str, str]]:
    auth = request.headers.get("authorization") or ""
    if not auth.lower().startswith("basic "):
        return None
    token = auth.split(" ", 1)[1].strip()
    try:
        decoded = base64.b64decode(token).decode("utf-8")
    except Exception:
        return None
    if ":" not in decoded:
        return None
    username, password = decoded.split(":", 1)
    return username, password

def verify_admin_credentials(db: Session, username: str, password: str) -> bool:
    import secrets as _secrets
    admin_user = os.getenv("ADMIN_USERNAME", "admin")
    if not _secrets.compare_digest(username, admin_user):
        return False
    stored = ensure_admin_password_hash(db)
    return verify_password(password, stored)

def is_admin_session(request: Request, db: Session) -> bool:
    import secrets as _secrets
    admin_user = os.getenv("ADMIN_USERNAME", "admin")
    sess_user = request.session.get("admin")
    return isinstance(sess_user, str) and _secrets.compare_digest(sess_user, admin_user)

def require_admin_api(request: Request, db: Session) -> str:
    if is_admin_session(request, db):
        return request.session["admin"]
    basic = parse_basic_auth(request)
    if basic and verify_admin_credentials(db, basic[0], basic[1]):
        return basic[0]
    raise HTTPException(
        status_code=status.HTTP_401_UNAUTHORIZED,
        detail="Unauthorized",
        headers={"WWW-Authenticate": "Basic"},
    )

# Dependency
def get_db():
    db = database.SessionLocal()
    try:
        yield db
    finally:
        db.close()

class SettingsUpdate(BaseModel):
    llm_api_key: Optional[str] = None
    llm_base_url: Optional[str] = None
    llm_model: Optional[str] = None

class AdminPasswordChange(BaseModel):
    old_password: str
    new_password: str

def _run_command(args: list[str], timeout: int = 30) -> dict:
    try:
        p = subprocess.run(args, capture_output=True, text=True, timeout=timeout)
        return {
            "cmd": " ".join(args),
            "ok": p.returncode == 0,
            "code": p.returncode,
            "stdout": (p.stdout or "")[:2000],
            "stderr": (p.stderr or "")[:2000],
        }
    except Exception as e:
        return {"cmd": " ".join(args), "ok": False, "code": None, "stdout": "", "stderr": str(e)[:2000]}

def _which(name: str) -> bool:
    try:
        return bool(shutil.which(name))
    except Exception:
        return False

class LotteryRecordOut(BaseModel):
    lottery_type: str
    issue: str
    date: Optional[str] = None
    red_balls: str
    blue_balls: str

class PredictionOut(BaseModel):
    id: int
    lottery_type: str
    based_on_issue: str
    target_issue: Optional[str] = None
    red_balls: str
    blue_balls: str
    created_at: str
    evaluated: bool
    actual_issue: Optional[str] = None
    red_hits: Optional[int] = None
    blue_hits: Optional[int] = None
    total_hits: Optional[int] = None

@app.get("/api/history/{lottery_type}")
def get_history(lottery_type: str, limit: int = 100, db: Session = Depends(get_db)):
    records = db.query(models.LotteryRecord).filter_by(lottery_type=lottery_type).order_by(models.LotteryRecord.issue.desc()).limit(limit).all()
    items = []
    for r in records:
        items.append(
            LotteryRecordOut(
                lottery_type=r.lottery_type,
                issue=r.issue,
                date=r.date.isoformat() if r.date else None,
                red_balls=r.red_balls,
                blue_balls=r.blue_balls,
            ).model_dump()
        )
    return items

@app.get("/api/latest/{lottery_type}")
def get_latest(lottery_type: str, db: Session = Depends(get_db)):
    r = (
        db.query(models.LotteryRecord)
        .filter_by(lottery_type=lottery_type)
        .order_by(models.LotteryRecord.issue.desc())
        .first()
    )
    if not r:
        return None
    return LotteryRecordOut(
        lottery_type=r.lottery_type,
        issue=r.issue,
        date=r.date.isoformat() if r.date else None,
        red_balls=r.red_balls,
        blue_balls=r.blue_balls,
    ).model_dump()

@app.get("/api/predictions/{lottery_type}")
def get_predictions(lottery_type: str, limit: int = 5, offset: int = 0, db: Session = Depends(get_db)):
    latest = (
        db.query(models.LotteryRecord)
        .filter_by(lottery_type=lottery_type)
        .order_by(models.LotteryRecord.issue.desc())
        .first()
    )
    if not latest:
        return {"based_on_issue": None, "items": [], "llm": None}

    all_rows = (
        db.query(models.PredictionRecord)
        .filter_by(lottery_type=lottery_type, based_on_issue=latest.issue)
        .filter(models.PredictionRecord.used_llm == True)
        .all()
    )
    if all_rows:
        red_w, blue_w = build_number_weights(db, lottery_type)
        ranked = sorted(
            all_rows,
            key=lambda p: (
                -score_prediction(lottery_type, p.red_balls, p.blue_balls, red_w, blue_w),
                p.sequence,
                p.id,
            ),
        )
        top = ranked[0]
        q = ranked[offset : offset + limit]
        configured_base_url = get_setting(db, "llm_base_url") or ""
        llm = {
            "used": True,
            "model": top.llm_model or (get_setting(db, "llm_model") or None),
            "base_url": configured_base_url or None,
            "latency_ms": top.llm_latency_ms,
            "created_at": _dt_to_utc_iso(top.created_at),
        }
    else:
        q = []
        try:
            llm = get_llm_status(lottery_type, db=db)
        except Exception:
            llm = None
    items = [
        PredictionOut(
            id=p.id,
            lottery_type=p.lottery_type,
            based_on_issue=p.based_on_issue,
            target_issue=p.target_issue,
            red_balls=p.red_balls,
            blue_balls=p.blue_balls,
            created_at=_dt_to_utc_iso(p.created_at) or "",
            evaluated=bool(p.evaluated),
            actual_issue=p.actual_issue,
            red_hits=p.red_hits,
            blue_hits=p.blue_hits,
            total_hits=p.total_hits,
        ).model_dump()
        for p in q
    ]
    return {"based_on_issue": latest.issue, "items": items, "llm": llm}

@app.get("/api/hit-stats/{lottery_type}")
def get_hit_stats(lottery_type: str, db: Session = Depends(get_db), cycles_limit: int = 12):
    latest = (
        db.query(models.LotteryRecord)
        .filter_by(lottery_type=lottery_type)
        .order_by(models.LotteryRecord.issue.desc())
        .first()
    )
    latest_issue = latest.issue if latest else None

    try:
        evaluate_predictions(db, lottery_type)
    except Exception:
        pass

    def parse_nums(value: Optional[str]) -> list[str]:
        if not value:
            return []
        return [s.strip() for s in str(value).split(",") if s.strip()]

    def build_demo_payload() -> dict:
        demo_actual_issue = latest_issue or ("2026010" if lottery_type == "ssq" else "2026007")
        demo_based_on_issue = str(int(demo_actual_issue) - 1) if str(demo_actual_issue).isdigit() else demo_actual_issue
        if lottery_type == "dlt":
            actual_red_balls = "03,08,12,19,30"
            actual_blue_balls = "02,11"
            red_a = "03,07,12,21,30"
            red_b = "01,08,11,19,29"
            blue_a = "02,11"
            blue_b = "04,10"
        else:
            actual_red_balls = "02,09,16,20,24,31"
            actual_blue_balls = "12"
            red_a = "02,07,16,21,24,31"
            red_b = "01,09,12,20,27,33"
            blue_a = "12"
            blue_b = "05"

        actual_red_set = set(parse_nums(actual_red_balls))
        actual_blue_set = set(parse_nums(actual_blue_balls))

        recent_predictions = []
        for i in range(20):
            red_balls = red_a if i % 2 == 0 else red_b
            blue_balls = blue_a if i % 3 == 0 else blue_b
            red_hits = sum(1 for n in parse_nums(red_balls) if n in actual_red_set)
            blue_hits = sum(1 for n in parse_nums(blue_balls) if n in actual_blue_set)
            total_hits = red_hits + blue_hits
            recent_predictions.append(
                {
                    "id": f"demo-{lottery_type}-{i}",
                    "sequence": i,
                    "red_balls": red_balls,
                    "blue_balls": blue_balls,
                    "actual_red_balls": actual_red_balls,
                    "actual_blue_balls": actual_blue_balls,
                    "red_hits": red_hits,
                    "blue_hits": blue_hits,
                    "total_hits": total_hits,
                    "based_on_issue": demo_based_on_issue,
                    "actual_issue": demo_actual_issue,
                }
            )

        total_predictions = len(recent_predictions)
        total_red_hits = sum(p["red_hits"] for p in recent_predictions)
        total_blue_hits = sum(p["blue_hits"] for p in recent_predictions)
        total_hits = sum(p["total_hits"] for p in recent_predictions)
        total_pred_red = sum(len(parse_nums(p["red_balls"])) for p in recent_predictions)
        total_pred_blue = sum(len(parse_nums(p["blue_balls"])) for p in recent_predictions)
        total_pred_nums = total_pred_red + total_pred_blue
        max_total_hits = max((p["total_hits"] for p in recent_predictions), default=0)

        thresholds = {1: 0, 3: 0, 5: 0}
        for p in recent_predictions:
            hits = p["total_hits"]
            for k in thresholds:
                if hits >= k:
                    thresholds[k] += 1

        latest_summary = {
            "actual_issue": demo_actual_issue,
            "based_on_issue": demo_based_on_issue,
            "actual_red_balls": actual_red_balls,
            "actual_blue_balls": actual_blue_balls,
            "total_predictions": total_predictions,
            "total_red_hits": total_red_hits,
            "total_blue_hits": total_blue_hits,
            "total_hits": total_hits,
            "avg_red_hits": round(total_red_hits / total_predictions, 2) if total_predictions else 0,
            "avg_blue_hits": round(total_blue_hits / total_predictions, 2) if total_predictions else 0,
            "avg_total_hits": round(total_hits / total_predictions, 2) if total_predictions else 0,
            "max_total_hits": max_total_hits,
            "hit_ratio": round(total_hits / total_pred_nums, 4) if total_pred_nums else 0,
            "red_hit_ratio": round(total_red_hits / total_pred_red, 4) if total_pred_red else 0,
            "blue_hit_ratio": round(total_blue_hits / total_pred_blue, 4) if total_pred_blue else 0,
            "groups_ge_1": thresholds[1],
            "groups_ge_3": thresholds[3],
            "groups_ge_5": thresholds[5],
        }

        cycles = []
        base_hit_ratio = latest_summary["hit_ratio"]
        for i in range(10):
            issue = str(int(demo_actual_issue) - i) if str(demo_actual_issue).isdigit() else demo_actual_issue
            hit_ratio = max(0.0, min(1.0, float(base_hit_ratio) - i * 0.01))
            cycles.append(
                {
                    "actual_issue": issue,
                    "based_on_issue": demo_based_on_issue,
                    "actual_red_balls": actual_red_balls,
                    "actual_blue_balls": actual_blue_balls,
                    "total_predictions": total_predictions,
                    "total_hits": total_hits,
                    "avg_total_hits": latest_summary["avg_total_hits"],
                    "max_total_hits": max_total_hits,
                    "hit_ratio": round(hit_ratio, 4),
                }
            )

        best = max(recent_predictions, key=lambda p: p["total_hits"], default=None)
        cumulative_stats = {
            "evaluated_issues": 18,
            "total_predictions": total_predictions * 18,
            "total_red_hits": total_red_hits * 18,
            "total_blue_hits": total_blue_hits * 18,
            "total_hits": total_hits * 18,
            "avg_red_hits": latest_summary["avg_red_hits"],
            "avg_blue_hits": latest_summary["avg_blue_hits"],
            "avg_total_hits": latest_summary["avg_total_hits"],
            "hit_ratio": latest_summary["hit_ratio"],
            "red_hit_ratio": latest_summary["red_hit_ratio"],
            "blue_hit_ratio": latest_summary["blue_hit_ratio"],
            "hit_distribution": {},
            "best_prediction": {
                "red_balls": best["red_balls"] if best else red_a,
                "blue_balls": best["blue_balls"] if best else blue_a,
                "actual_red_balls": actual_red_balls,
                "actual_blue_balls": actual_blue_balls,
                "red_hits": best["red_hits"] if best else 0,
                "blue_hits": best["blue_hits"] if best else 0,
                "total_hits": best["total_hits"] if best else 0,
                "actual_issue": demo_actual_issue,
            },
        }

        return {
            "latest_issue": latest_issue,
            "latest_evaluated_issue": demo_actual_issue,
            "recent_predictions": recent_predictions,
            "latest_summary": latest_summary,
            "cycles": cycles,
            "cumulative_stats": cumulative_stats,
            "is_demo": True,
        }

    def cycle_summary(preds: list[models.PredictionRecord]):
        if not preds:
            return None
        total_predictions = len(preds)
        total_red_hits = sum(p.red_hits or 0 for p in preds)
        total_blue_hits = sum(p.blue_hits or 0 for p in preds)
        total_hits = sum(p.total_hits or 0 for p in preds)
        total_pred_red = sum(len(parse_nums(p.red_balls)) for p in preds)
        total_pred_blue = sum(len(parse_nums(p.blue_balls)) for p in preds)
        total_pred_nums = total_pred_red + total_pred_blue
        max_total_hits = max((p.total_hits or 0) for p in preds) if preds else 0
        thresholds = {1: 0, 3: 0, 5: 0}
        for p in preds:
            hits = p.total_hits or 0
            for k in thresholds:
                if hits >= k:
                    thresholds[k] += 1
        actual_red = preds[0].actual_red_balls
        actual_blue = preds[0].actual_blue_balls
        return {
            "actual_issue": preds[0].actual_issue,
            "based_on_issue": preds[0].based_on_issue,
            "actual_red_balls": actual_red,
            "actual_blue_balls": actual_blue,
            "total_predictions": total_predictions,
            "total_red_hits": total_red_hits,
            "total_blue_hits": total_blue_hits,
            "total_hits": total_hits,
            "avg_red_hits": round(total_red_hits / total_predictions, 2) if total_predictions else 0,
            "avg_blue_hits": round(total_blue_hits / total_predictions, 2) if total_predictions else 0,
            "avg_total_hits": round(total_hits / total_predictions, 2) if total_predictions else 0,
            "max_total_hits": max_total_hits,
            "hit_ratio": round(total_hits / total_pred_nums, 4) if total_pred_nums else 0,
            "red_hit_ratio": round(total_red_hits / total_pred_red, 4) if total_pred_red else 0,
            "blue_hit_ratio": round(total_blue_hits / total_pred_blue, 4) if total_pred_blue else 0,
            "groups_ge_1": thresholds[1],
            "groups_ge_3": thresholds[3],
            "groups_ge_5": thresholds[5],
        }

    latest_eval_row = (
        db.query(models.PredictionRecord.actual_issue)
        .filter_by(lottery_type=lottery_type, used_llm=True, evaluated=True)
        .filter(models.PredictionRecord.actual_issue.isnot(None))
        .order_by(models.PredictionRecord.actual_issue.desc())
        .first()
    )
    latest_evaluated_issue = latest_eval_row[0] if latest_eval_row else None

    if not latest_evaluated_issue:
        return build_demo_payload()

    recent_predictions = (
        db.query(models.PredictionRecord)
        .filter_by(lottery_type=lottery_type, used_llm=True, evaluated=True, actual_issue=latest_evaluated_issue)
        .order_by(models.PredictionRecord.sequence.asc())
        .limit(20)
        .all()
    )

    recent_items = [
        {
            "id": p.id,
            "sequence": p.sequence,
            "red_balls": p.red_balls,
            "blue_balls": p.blue_balls,
            "actual_red_balls": p.actual_red_balls,
            "actual_blue_balls": p.actual_blue_balls,
            "red_hits": p.red_hits,
            "blue_hits": p.blue_hits,
            "total_hits": p.total_hits,
            "based_on_issue": p.based_on_issue,
            "actual_issue": p.actual_issue,
        }
        for p in recent_predictions
    ]

    latest_summary = cycle_summary(recent_predictions)

    issue_rows = (
        db.query(models.PredictionRecord.actual_issue)
        .filter_by(lottery_type=lottery_type, used_llm=True, evaluated=True)
        .filter(models.PredictionRecord.actual_issue.isnot(None))
        .distinct()
        .order_by(models.PredictionRecord.actual_issue.desc())
        .limit(max(1, min(int(cycles_limit), 60)))
        .all()
    )
    recent_issues = [r[0] for r in issue_rows if r and r[0]]

    cycle_predictions = []
    if recent_issues:
        cycle_predictions = (
            db.query(models.PredictionRecord)
            .filter_by(lottery_type=lottery_type, used_llm=True, evaluated=True)
            .filter(models.PredictionRecord.actual_issue.in_(recent_issues))
            .order_by(models.PredictionRecord.actual_issue.desc(), models.PredictionRecord.sequence.asc())
            .all()
        )

    grouped: dict[str, list[models.PredictionRecord]] = {}
    for p in cycle_predictions:
        if not p.actual_issue:
            continue
        grouped.setdefault(p.actual_issue, []).append(p)

    cycles = []
    for issue in recent_issues:
        preds = grouped.get(issue, [])
        s = cycle_summary(preds)
        if s:
            cycles.append(s)

    all_evaluated = (
        db.query(models.PredictionRecord)
        .filter_by(lottery_type=lottery_type, used_llm=True, evaluated=True)
        .all()
    )

    if all_evaluated:
        total_predictions = len(all_evaluated)
        total_red_hits = sum(p.red_hits or 0 for p in all_evaluated)
        total_blue_hits = sum(p.blue_hits or 0 for p in all_evaluated)
        total_hits = sum(p.total_hits or 0 for p in all_evaluated)
        total_pred_red = sum(len(parse_nums(p.red_balls)) for p in all_evaluated)
        total_pred_blue = sum(len(parse_nums(p.blue_balls)) for p in all_evaluated)
        total_pred_nums = total_pred_red + total_pred_blue

        hit_distribution = {}
        for p in all_evaluated:
            hits = p.total_hits or 0
            hit_distribution[hits] = hit_distribution.get(hits, 0) + 1

        best_prediction = max(all_evaluated, key=lambda p: p.total_hits or 0)
        evaluated_issues = len({p.actual_issue for p in all_evaluated if p.actual_issue})

        cumulative_stats = {
            "evaluated_issues": evaluated_issues,
            "total_predictions": total_predictions,
            "total_red_hits": total_red_hits,
            "total_blue_hits": total_blue_hits,
            "total_hits": total_hits,
            "avg_red_hits": round(total_red_hits / total_predictions, 2) if total_predictions else 0,
            "avg_blue_hits": round(total_blue_hits / total_predictions, 2) if total_predictions else 0,
            "avg_total_hits": round(total_hits / total_predictions, 2) if total_predictions else 0,
            "hit_ratio": round(total_hits / total_pred_nums, 4) if total_pred_nums else 0,
            "red_hit_ratio": round(total_red_hits / total_pred_red, 4) if total_pred_red else 0,
            "blue_hit_ratio": round(total_blue_hits / total_pred_blue, 4) if total_pred_blue else 0,
            "hit_distribution": hit_distribution,
            "best_prediction": {
                "red_balls": best_prediction.red_balls,
                "blue_balls": best_prediction.blue_balls,
                "actual_red_balls": best_prediction.actual_red_balls,
                "actual_blue_balls": best_prediction.actual_blue_balls,
                "red_hits": best_prediction.red_hits,
                "blue_hits": best_prediction.blue_hits,
                "total_hits": best_prediction.total_hits,
                "actual_issue": best_prediction.actual_issue,
            },
        }
    else:
        cumulative_stats = None

    return {
        "latest_issue": latest_issue,
        "latest_evaluated_issue": latest_evaluated_issue,
        "recent_predictions": recent_items,
        "latest_summary": latest_summary,
        "cycles": cycles,
        "cumulative_stats": cumulative_stats,
    }


@app.post("/api/scrape/{lottery_type}")
def scrape_data(
    lottery_type: str,
    limit: int = 100,
    upsert: bool = False,
    issue: str = "",
    fill_missing: bool = True,
    db: Session = Depends(get_db),
):
    s = scraper.LotteryScraper(db)
    if lottery_type == 'ssq':
        stats = s.scrape_ssq(limit=limit, upsert=upsert, want_issue=issue or None)
    elif lottery_type == 'dlt':
        stats = s.scrape_dlt(limit=limit, upsert=upsert, want_issue=issue or None)
    else:
        raise HTTPException(status_code=400, detail="Invalid lottery type")
    added = 0
    updated = 0
    seen = 0
    if isinstance(stats, dict):
        try:
            added = int(stats.get("added") or 0)
        except Exception:
            added = 0
        try:
            updated = int(stats.get("updated") or 0)
        except Exception:
            updated = 0
        try:
            seen = int(stats.get("seen") or 0)
        except Exception:
            seen = 0

    if fill_missing and not (issue or "").strip():
        try:
            extra = _backfill_missing_issues(db, lottery_type, base_limit=limit, max_limit=3000, years=2, max_rounds=6)
            if isinstance(extra, dict):
                try:
                    added += int(extra.get("added") or 0)
                except Exception:
                    pass
                try:
                    updated += int(extra.get("updated") or 0)
                except Exception:
                    pass
        except Exception:
            pass

    evaluate_predictions(db, lottery_type)
    ensure_predictions_for_cycle(db, lottery_type, min_count=20)

    try:
        _notify_latest_issue(lottery_type)
    except Exception:
        pass
    return {"message": f"Scraped {added} new records, updated {updated} records (seen {seen}) for {lottery_type}"}

def _missing_issues_for_year(db: Session, lottery_type: str, year: int) -> list[int]:
    prefix = f"{year:02d}"
    rows = (
        db.query(models.LotteryRecord.issue)
        .filter_by(lottery_type=lottery_type)
        .filter(models.LotteryRecord.issue.like(f"{prefix}%"))
        .all()
    )
    issues: list[int] = []
    for (s,) in rows:
        iv = issue_to_int(s)
        if iv is None or iv // 1000 != year:
            continue
        issues.append(iv)
    if not issues:
        return []
    present = {iv % 1000 for iv in issues}
    max_seq = max(present)
    missing: list[int] = []
    for seq in range(1, max_seq + 1):
        if seq not in present:
            missing.append(year * 1000 + seq)
    return missing

def _backfill_missing_issues(
    db: Session,
    lottery_type: str,
    base_limit: int,
    max_limit: int = 3000,
    years: int = 2,
    max_rounds: int = 6,
) -> dict:
    latest = (
        db.query(models.LotteryRecord)
        .filter_by(lottery_type=lottery_type)
        .order_by(models.LotteryRecord.issue.desc())
        .first()
    )
    latest_int = issue_to_int(latest.issue) if latest else None
    if latest_int is None:
        return {"added": 0, "updated": 0}

    scraper_inst = scraper.LotteryScraper(db)
    cur_limit = max(1, int(base_limit or 0))
    total_added = 0
    total_updated = 0

    for _ in range(max_rounds):
        year0 = latest_int // 1000
        candidates: list[int] = []
        for i in range(max(1, int(years or 0))):
            candidates.extend(_missing_issues_for_year(db, lottery_type, year0 - i))
        if not candidates:
            break
        target_issue = str(min(candidates))
        if cur_limit >= max_limit:
            break
        cur_limit = min(max_limit, max(cur_limit * 2, cur_limit + 200))
        if lottery_type == "ssq":
            st = scraper_inst.scrape_ssq(limit=cur_limit, upsert=True, want_issue=target_issue)
        else:
            st = scraper_inst.scrape_dlt(limit=cur_limit, upsert=True, want_issue=target_issue)
        if isinstance(st, dict):
            try:
                total_added += int(st.get("added") or 0)
            except Exception:
                pass
            try:
                total_updated += int(st.get("updated") or 0)
            except Exception:
                pass

    return {"added": total_added, "updated": total_updated}

@app.post("/api/predict/{lottery_type}")
def predict_numbers(lottery_type: str, count: int = 5, db: Session = Depends(get_db)):
    latest = (
        db.query(models.LotteryRecord)
        .filter_by(lottery_type=lottery_type)
        .order_by(models.LotteryRecord.issue.desc())
        .first()
    )
    if not latest:
        raise HTTPException(status_code=400, detail="No data available. Please scrape data first.")

    if count < 1 or count > 30:
        raise HTTPException(status_code=400, detail="count must be between 1 and 30")

    created, meta = generate_predictions_for_cycle(db, lottery_type, based_on_issue=latest.issue, count=count)
    items = [
        PredictionOut(
            id=p.id,
            lottery_type=p.lottery_type,
            based_on_issue=p.based_on_issue,
            target_issue=p.target_issue,
            red_balls=p.red_balls,
            blue_balls=p.blue_balls,
            created_at=_dt_to_utc_iso(p.created_at) or "",
            evaluated=bool(p.evaluated),
            actual_issue=p.actual_issue,
            red_hits=p.red_hits,
            blue_hits=p.blue_hits,
            total_hits=p.total_hits,
        ).model_dump()
        for p in created
    ]
    llm = {
        "used": bool(meta.get("used_llm")),
        "model": meta.get("model"),
        "base_url": meta.get("base_url"),
        "latency_ms": meta.get("latency_ms"),
    }
    return {"based_on_issue": latest.issue, "items": items, "llm": llm}

def issue_to_int(issue: str) -> Optional[int]:
    if issue is None:
        return None
    s = str(issue).strip()
    if not s.isdigit():
        return None
    try:
        return int(s)
    except Exception:
        return None

def normalize_numbers(s: str) -> list[str]:
    if not s:
        return []
    return [x.strip() for x in str(s).split(",") if x.strip()]

def build_number_weights(db: Session, lottery_type: str, limit: int = 120) -> tuple[dict[str, float], dict[str, float]]:
    records = (
        db.query(models.LotteryRecord)
        .filter_by(lottery_type=lottery_type)
        .order_by(models.LotteryRecord.issue.desc())
        .limit(limit)
        .all()
    )
    red_w: dict[str, float] = {}
    blue_w: dict[str, float] = {}
    for i, r in enumerate(records):
        decay = 0.985 ** i
        for n in normalize_numbers(r.red_balls):
            red_w[n] = red_w.get(n, 0.0) + decay
        for n in normalize_numbers(r.blue_balls):
            blue_w[n] = blue_w.get(n, 0.0) + decay
    return red_w, blue_w

def score_prediction(lottery_type: str, red_balls: str, blue_balls: str, red_w: dict[str, float], blue_w: dict[str, float]) -> float:
    reds = normalize_numbers(red_balls)
    blues = normalize_numbers(blue_balls)
    score = 0.0
    for n in reds:
        score += red_w.get(n, 0.0)
    blue_factor = 1.15 if lottery_type == "ssq" else 1.1
    for n in blues:
        score += blue_w.get(n, 0.0) * blue_factor
    return score

def evaluate_predictions(db: Session, lottery_type: str):
    records = (
        db.query(models.LotteryRecord)
        .filter_by(lottery_type=lottery_type)
        .order_by(models.LotteryRecord.issue.asc())
        .all()
    )
    parsed = []
    for r in records:
        iv = issue_to_int(r.issue)
        if iv is None:
            continue
        parsed.append((iv, r))
    parsed.sort(key=lambda x: x[0])

    next_by_issue: dict[str, models.LotteryRecord] = {}
    for i in range(len(parsed) - 1):
        cur = parsed[i][1]
        nxt = parsed[i + 1][1]
        next_by_issue[cur.issue] = nxt

    pending = (
        db.query(models.PredictionRecord)
        .filter_by(lottery_type=lottery_type, evaluated=False)
        .order_by(models.PredictionRecord.created_at.asc())
        .all()
    )
    for p in pending:
        actual = next_by_issue.get(p.based_on_issue)
        if not actual:
            continue
        pred_red = set(normalize_numbers(p.red_balls))
        pred_blue = set(normalize_numbers(p.blue_balls))
        act_red = set(normalize_numbers(actual.red_balls))
        act_blue = set(normalize_numbers(actual.blue_balls))

        red_hits = len(pred_red & act_red)
        blue_hits = len(pred_blue & act_blue)
        total_hits = red_hits + blue_hits

        p.evaluated = True
        p.actual_issue = actual.issue
        p.actual_red_balls = actual.red_balls
        p.actual_blue_balls = actual.blue_balls
        p.red_hits = red_hits
        p.blue_hits = blue_hits
        p.total_hits = total_hits
    db.commit()

def generate_predictions_for_cycle(db: Session, lottery_type: str, based_on_issue: str, count: int):
    p = predictor.Predictor(db)
    result = p.predict_multi(lottery_type, count=count)
    if "error" in result:
        tail = ""
        raw = result.get("raw_response")
        if raw is not None:
            tail = "；raw=" + str(raw)[:200]
        if result.get("valid_count") is not None and result.get("want_count") is not None:
            tail = f"；valid={result.get('valid_count')}/{result.get('want_count')}" + tail
        raise HTTPException(status_code=500, detail=str(result.get("error") or "LLM failed") + tail)
    meta = result.get("meta") or {"used_llm": False}
    try:
        prompt = result.get("prompt")
        raw_content = result.get("raw_content")
        if prompt or raw_content:
            db.add(
                models.AiGenerationLog(
                    lottery_type=lottery_type,
                    based_on_issue=based_on_issue,
                    llm_model=meta.get("model"),
                    llm_base_url=meta.get("base_url"),
                    llm_latency_ms=meta.get("latency_ms"),
                    prompt=prompt,
                    raw_content=raw_content,
                )
            )
            db.commit()
    except Exception:
        try:
            db.rollback()
        except Exception:
            pass
    status_snapshot = {}
    try:
        status_snapshot = {
            "used": bool(meta.get("used_llm")),
            "model": meta.get("model"),
            "base_url": meta.get("base_url"),
            "latency_ms": meta.get("latency_ms"),
            "based_on_issue": based_on_issue,
            "updated_at": datetime.utcnow().isoformat(),
        }
        _llm_status[lottery_type] = status_snapshot
    except Exception:
        pass

    existing_max = (
        db.query(models.PredictionRecord.sequence)
        .filter_by(lottery_type=lottery_type, based_on_issue=based_on_issue)
        .order_by(models.PredictionRecord.sequence.desc())
        .first()
    )
    start_seq = (existing_max[0] + 1) if existing_max else 0

    created = []
    for idx, item in enumerate(result.get("predictions", [])):
        reds = ",".join(item.get("red_balls", []))
        blues = ",".join(item.get("blue_balls", []))
        rec = models.PredictionRecord(
            lottery_type=lottery_type,
            based_on_issue=based_on_issue,
            target_issue=item.get("target_issue"),
            sequence=start_seq + idx,
            red_balls=reds,
            blue_balls=blues,
            used_llm=bool(meta.get("used_llm")),
            llm_model=meta.get("model"),
            llm_latency_ms=meta.get("latency_ms"),
        )
        db.add(rec)
        created.append(rec)
    db.commit()
    try:
        if status_snapshot:
            set_setting(db, f"llm_status_{lottery_type}", json.dumps(status_snapshot, ensure_ascii=False))
    except Exception:
        pass
    for rec in created:
        db.refresh(rec)
    return created, meta

def ensure_predictions_for_cycle(db: Session, lottery_type: str, min_count: int = 20) -> bool:
    latest = (
        db.query(models.LotteryRecord)
        .filter_by(lottery_type=lottery_type)
        .order_by(models.LotteryRecord.issue.desc())
        .first()
    )
    if not latest:
        return False
    existing = (
        db.query(models.PredictionRecord)
        .filter_by(lottery_type=lottery_type, based_on_issue=latest.issue)
        .filter(models.PredictionRecord.used_llm == True)
        .count()
    )
    if existing >= min_count:
        try:
            api_key = get_setting(db, "llm_api_key") or ""
            model = get_setting(db, "llm_model") or ""
            raw_status = get_setting(db, f"llm_status_{lottery_type}") or ""
            if api_key and model and not raw_status:
                generate_predictions_for_cycle(db, lottery_type, based_on_issue=latest.issue, count=1)
        except Exception:
            pass
        return True
    remaining = min_count - existing
    while remaining > 0:
        batch = min(5, remaining)
        try:
            generate_predictions_for_cycle(db, lottery_type, based_on_issue=latest.issue, count=batch)
        except Exception:
            break
        remaining -= batch
    existing2 = (
        db.query(models.PredictionRecord)
        .filter_by(lottery_type=lottery_type, based_on_issue=latest.issue)
        .filter(models.PredictionRecord.used_llm == True)
        .count()
    )
    return existing2 >= min_count

@app.get("/api/llm/status/{lottery_type}")
def get_llm_status(lottery_type: str, db: Session = Depends(get_db)):
    if lottery_type not in ("ssq", "dlt"):
        raise HTTPException(status_code=400, detail="Invalid lottery type")
    cur = _llm_status.get(lottery_type) or {}
    if not cur:
        try:
            raw = get_setting(db, f"llm_status_{lottery_type}") or ""
            if raw:
                cur = json.loads(raw)
                if isinstance(cur, dict):
                    _llm_status[lottery_type] = cur
                else:
                    cur = {}
        except Exception:
            cur = {}

    configured_model = get_setting(db, "llm_model") or ""
    configured_base_url = get_setting(db, "llm_base_url") or ""
    configured_api_key = bool(get_setting(db, "llm_api_key") or "")

    if not cur:
        return {
            "used": False,
            "unknown": True,
            "configured": configured_api_key and bool(configured_model),
            "configured_model": configured_model,
            "configured_base_url": configured_base_url,
        }

    cur_out = dict(cur)
    cur_out["configured"] = configured_api_key and bool(configured_model)
    cur_out["configured_model"] = configured_model
    cur_out["configured_base_url"] = configured_base_url
    return cur_out

@app.post("/api/ensure_predictions/{lottery_type}")
async def ensure_predictions_endpoint(lottery_type: str, min_count: int = 20):
    if lottery_type not in ("ssq", "dlt"):
        raise HTTPException(status_code=400, detail="Invalid lottery type")
    if min_count < 1 or min_count > 30:
        raise HTTPException(status_code=400, detail="min_count must be between 1 and 30")

    def run():
        db = database.SessionLocal()
        try:
            ensure_predictions_for_cycle(db, lottery_type, min_count=min_count)
        finally:
            db.close()

    asyncio.create_task(asyncio.to_thread(run))
    return {"queued": True}

LOTTERY_TZ = ZoneInfo("Asia/Shanghai")
LOTTERY_SCHEDULE = {
    "ssq": {"days": {1, 3, 6}, "draw_time": dtime(21, 15), "delay_min": 8},
    "dlt": {"days": {0, 2, 5}, "draw_time": dtime(21, 25), "delay_min": 8},
}
try:
    SCRAPE_WINDOW_MINUTES = int(os.getenv("SCRAPE_WINDOW_MINUTES", "360"))
except Exception:
    SCRAPE_WINDOW_MINUTES = 360
_auto_scrape_state: dict[str, dict] = {}
_latest_issue_events: dict[str, asyncio.Event] = {"ssq": asyncio.Event(), "dlt": asyncio.Event()}

def in_scrape_window(now_cn: datetime, lottery_type: str) -> Optional[tuple[datetime, datetime]]:
    cfg = LOTTERY_SCHEDULE.get(lottery_type)
    if not cfg:
        return None
    if now_cn.weekday() not in cfg["days"]:
        return None
    base = datetime.combine(now_cn.date(), cfg["draw_time"], tzinfo=LOTTERY_TZ) + timedelta(minutes=cfg["delay_min"])
    return (base, base + timedelta(minutes=SCRAPE_WINDOW_MINUTES))

def _notify_latest_issue(lottery_type: str):
    ev = _latest_issue_events.get(lottery_type)
    if not ev:
        return
    if not ev.is_set():
        ev.set()

async def smart_scrape_loop():
    """基于实际开奖时间的智能抓取循环
    
    双色球：每周二、四、日 21:15开奖，21:23后开始抓取
    大乐透：每周一、三、六 21:25开奖，21:33后开始抓取
    抓取窗口：90分钟，期间每45秒检查一次
    """
    logger.info("智能抓取循环已启动 - 将根据开奖时间自动抓取数据")
    while True:
        try:
            now_cn = get_now_cn()
            for lt in ("ssq", "dlt"):
                lottery_name = "双色球" if lt == "ssq" else "大乐透"
                
                win = in_scrape_window(now_cn, lt)
                if not win:
                    _auto_scrape_state.pop(lt, None)
                    continue
                    
                win_start, win_end = win
                if now_cn < win_start:
                    _auto_scrape_state.pop(lt, None)
                    continue
                if now_cn > win_end:
                    st = _auto_scrape_state.get(lt)
                    if st and not st.get("done") and st.get("win_start") == win_start:
                        logger.error(f"[{lottery_name}] ❌ 抓取失败：抓取窗口已结束，未获取到新期数据")
                    _auto_scrape_state.pop(lt, None)
                    continue
                    
                st = _auto_scrape_state.get(lt)
                if not st or st.get("win_start") != win_start:
                    logger.info(f"[{lottery_name}] 检测到开奖日且在抓取窗口内（{win_start.strftime('%H:%M')} - {win_end.strftime('%H:%M')}），开始智能抓取...")
                    _auto_scrape_state[lt] = {"win_start": win_start, "baseline_issue": None, "done": False}
                    st = _auto_scrape_state[lt]

                if st.get("done"):
                    continue

                try:
                    db = database.SessionLocal()
                    try:
                        if st.get("baseline_issue") is None:
                            latest_before = (
                                db.query(models.LotteryRecord)
                                .filter_by(lottery_type=lt)
                                .order_by(models.LotteryRecord.issue.desc())
                                .first()
                            )
                            st["baseline_issue"] = latest_before.issue if latest_before else None
                            if st["baseline_issue"]:
                                logger.info(f"[{lottery_name}] 基准期号: {st['baseline_issue']}")
                            else:
                                logger.warning(f"[{lottery_name}] 暂无历史数据")

                        s = scraper.LotteryScraper(db)
                        if lt == "ssq":
                            stats = s.scrape_ssq(limit=1)
                        else:
                            stats = s.scrape_dlt(limit=1)
                        count = 0
                        if isinstance(stats, dict):
                            try:
                                count = int(stats.get("added") or 0)
                            except Exception:
                                count = 0

                        latest_after = (
                            db.query(models.LotteryRecord)
                            .filter_by(lottery_type=lt)
                            .order_by(models.LotteryRecord.issue.desc())
                            .first()
                        )
                        latest_issue = latest_after.issue if latest_after else None

                        if latest_issue and latest_issue != st.get("baseline_issue"):
                            logger.info(f"[{lottery_name}] ✅ 抓取成功！新期号: {latest_issue}")
                            logger.info(f"[{lottery_name}] 开始生成推算...")
                            evaluate_predictions(db, lt)
                            ensure_predictions_for_cycle(db, lt, min_count=20)
                            logger.info(f"[{lottery_name}] 推算生成完成，共20组")
                            st["done"] = True
                            _notify_latest_issue(lt)
                        else:
                            remaining_minutes = (win_end - now_cn).total_seconds() / 60
                            if remaining_minutes < 10 and count == 0:
                                logger.warning(f"[{lottery_name}] 抓取窗口即将结束（剩余{remaining_minutes:.0f}分钟），但尚未获取到新期数据")
                    finally:
                        db.close()
                except Exception as e:
                    logger.error(f"[{lottery_name}] 智能抓取任务异常: {e}")
                            
        except Exception as e:
            logger.error(f"智能抓取循环异常: {e}")
        await asyncio.sleep(45)

@app.get("/api/stream/latest/{lottery_type}")
async def stream_latest(lottery_type: str, db: Session = Depends(get_db)):
    if lottery_type not in ("ssq", "dlt"):
        raise HTTPException(status_code=400, detail="Invalid lottery type")

    ev = _latest_issue_events.get(lottery_type)
    if not ev:
        raise HTTPException(status_code=500, detail="Event not available")

    initial = (
        db.query(models.LotteryRecord)
        .filter_by(lottery_type=lottery_type)
        .order_by(models.LotteryRecord.issue.desc())
        .first()
    )
    initial_issue = initial.issue if initial else ""

    async def gen():
        yield f"event: init\ndata: {initial_issue}\n\n"
        deadline = asyncio.get_event_loop().time() + 3_600
        while True:
            remaining = deadline - asyncio.get_event_loop().time()
            if remaining <= 0:
                yield "event: timeout\ndata: \n\n"
                return
            try:
                await asyncio.wait_for(ev.wait(), timeout=min(15.0, remaining))
                break
            except asyncio.TimeoutError:
                yield ": ping\n\n"
                continue
        ev.clear()

        latest = (
            db.query(models.LotteryRecord)
            .filter_by(lottery_type=lottery_type)
            .order_by(models.LotteryRecord.issue.desc())
            .first()
        )
        issue = latest.issue if latest else ""
        yield f"event: update\ndata: {issue}\n\n"

    headers = {
        "Cache-Control": "no-cache",
        "Connection": "keep-alive",
        "X-Accel-Buffering": "no",
    }
    return StreamingResponse(gen(), media_type="text/event-stream", headers=headers)

@app.on_event("startup")
async def on_startup():
    def init_time_offset():
        db = database.SessionLocal()
        try:
            _load_time_offset_from_db(db)
        finally:
            db.close()

    await asyncio.to_thread(init_time_offset)

    def warm_scrape(lt: str):
        db = database.SessionLocal()
        try:
            existing_count = db.query(models.LotteryRecord).filter_by(lottery_type=lt).count()
            if existing_count < 100:
                s = scraper.LotteryScraper(db)
                if lt == "ssq":
                    s.scrape_ssq(limit=100)
                else:
                    s.scrape_dlt(limit=100)
            evaluate_predictions(db, lt)
            ensure_predictions_for_cycle(db, lt, min_count=20)
        finally:
            db.close()

    asyncio.create_task(asyncio.to_thread(warm_scrape, "ssq"))
    asyncio.create_task(asyncio.to_thread(warm_scrape, "dlt"))
    asyncio.create_task(smart_scrape_loop())

@app.get("/api/settings")
def get_settings(request: Request, db: Session = Depends(get_db)):
    require_admin_api(request, db)
    settings = db.query(models.AppSettings).all()
    out = {}
    for s in settings:
        if not (s.key or "").startswith("llm_"):
            continue
        if s.key == "llm_api_key":
            continue
        out[s.key] = s.value
    return out

@app.post("/api/settings")
def update_settings(settings: SettingsUpdate, request: Request, db: Session = Depends(get_db)):
    require_admin_api(request, db)
    try:
        payload = settings.model_dump(exclude_unset=True)
    except Exception:
        payload = settings.dict(exclude_unset=True)
    for key, value in payload.items():
        if value is None:
            continue
        if key in ("llm_api_key", "llm_model") and isinstance(value, str) and value.strip() == "":
            continue
        setting = db.query(models.AppSettings).filter_by(key=key).first()
        if setting:
            setting.value = value
        else:
            setting = models.AppSettings(key=key, value=value)
            db.add(setting)
    db.commit()
    return {"message": "Settings updated"}

@app.post("/api/admin/test-llm")
def test_llm(settings: SettingsUpdate, request: Request, db: Session = Depends(get_db)):
    require_admin_api(request, db)

    saved_base_url = get_setting(db, "llm_base_url") or ""
    saved_model = get_setting(db, "llm_model") or ""

    if settings.llm_api_key is None or not (settings.llm_api_key or "").strip():
        raise HTTPException(status_code=400, detail="LLM API Key 为空")
    api_key = settings.llm_api_key
    base_url = settings.llm_base_url if settings.llm_base_url is not None else saved_base_url
    model = settings.llm_model if settings.llm_model is not None else saved_model

    if not model:
        raise HTTPException(status_code=400, detail="LLM 模型名为空")

    t0 = time.monotonic()
    try:
        base = (base_url or "https://api.siliconflow.cn/v1").strip()
        if not base:
            base = "https://api.siliconflow.cn/v1"
        base = base.rstrip("/")
        parsed_base = urlparse(base)
        path = (parsed_base.path or "").rstrip("/")
        if path.endswith("/chat/completions") or "/chat/completions" in path:
            url = base
        else:
            segs = [s for s in path.split("/") if s]
            if segs and segs[-1] == "v1":
                new_path = path + "/chat/completions"
            elif "v1" in segs:
                new_path = path + "/chat/completions"
            else:
                new_path = path + "/v1/chat/completions"
            url = urlunparse(parsed_base._replace(path=new_path))
        payload = {
            "model": model,
            "messages": [{"role": "user", "content": "只回复 JSON：{\"ok\":true}"}],
            "max_tokens": 32,
            "temperature": 0,
        }
        headers = {"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"}
        r = requests.post(url, headers=headers, json=payload, timeout=30)
        if r.status_code < 200 or r.status_code >= 300:
            raise RuntimeError(f"HTTP {r.status_code}: {(r.text or '')[:200]}")
        data = r.json() if r.content else {}
    except Exception as e:
        raise HTTPException(status_code=502, detail=f"调用失败：{str(e)}")
    dt_ms = int((time.monotonic() - t0) * 1000)

    content = ""
    try:
        content = ((((data.get("choices") or [{}])[0]).get("message") or {}).get("content") or "").strip()
    except Exception:
        content = ""

    return {
        "ok": True,
        "model": model,
        "base_url": base_url,
        "latency_ms": dt_ms,
        "reply": content[:200],
    }

@app.post("/api/admin/change-password")
def change_admin_password(payload: AdminPasswordChange, request: Request, db: Session = Depends(get_db)):
    if not is_admin_session(request, db):
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Unauthorized")
    if not payload.new_password or len(payload.new_password) < 8:
        raise HTTPException(status_code=400, detail="新密码至少 8 位")
    if not verify_admin_credentials(db, request.session["admin"], payload.old_password):
        raise HTTPException(status_code=400, detail="原密码不正确")
    set_setting(db, "admin_password_hash", hash_password(payload.new_password))
    set_setting(db, "admin_password_changed", "1")
    return {"message": "密码修改成功"}

@app.post("/api/admin/sync-time")
def sync_server_time(request: Request, db: Session = Depends(get_db)):
    require_admin_api(request, db)
    steps = []
    local_utc = datetime.now(timezone.utc)
    steps.append({"cmd": "local_utc", "ok": True, "stdout": local_utc.isoformat(), "stderr": "", "code": 0})
    steps.append({"cmd": "local_cn", "ok": True, "stdout": local_utc.astimezone(LOTTERY_TZ).isoformat(), "stderr": "", "code": 0})

    urls = [
        "https://www.baidu.com",
        "https://www.qq.com",
        "https://www.aliyun.com",
        "https://www.cloudflare.com",
    ]
    net_utc = None
    used_url = ""
    last_err = ""
    for url in urls:
        try:
            r = requests.head(url, timeout=10, allow_redirects=True, headers={"User-Agent": "Mozilla/5.0"})
            date_hdr = (r.headers.get("Date") or r.headers.get("date") or "").strip()
            if not date_hdr:
                steps.append({"cmd": f"HEAD {url}", "ok": False, "stdout": "", "stderr": "missing Date header", "code": r.status_code})
                continue
            dt = parsedate_to_datetime(date_hdr)
            if dt.tzinfo is None:
                dt = dt.replace(tzinfo=timezone.utc)
            net_utc = dt.astimezone(timezone.utc)
            used_url = url
            steps.append({"cmd": f"HEAD {url}", "ok": True, "stdout": f"{date_hdr} -> {net_utc.isoformat()}", "stderr": "", "code": r.status_code})
            break
        except Exception as e:
            last_err = str(e)
            steps.append({"cmd": f"HEAD {url}", "ok": False, "stdout": "", "stderr": last_err[:2000], "code": None})

    reason = []
    tz_ok = True
    synced = False
    offset_seconds = 0
    if net_utc is None:
        reason.append("无法获取网络时间（需要可访问外网）")
    else:
        offset = (net_utc - datetime.now(timezone.utc)).total_seconds()
        offset_seconds = int(round(offset))
        set_setting(db, "time_offset_seconds", str(offset_seconds))
        set_setting(db, "time_offset_source", used_url)
        set_setting(db, "time_offset_updated_at", datetime.now(timezone.utc).isoformat())
        _load_time_offset_from_db(db)
        synced = True
        steps.append({"cmd": "time_offset_seconds", "ok": True, "stdout": str(offset_seconds), "stderr": "", "code": 0})
        steps.append({"cmd": "app_now_cn", "ok": True, "stdout": get_now_cn().isoformat(), "stderr": "", "code": 0})

    return {
        "ok": bool(tz_ok) and bool(synced),
        "timezone": "Asia/Shanghai",
        "tz_ok": bool(tz_ok),
        "synced": bool(synced),
        "offset_seconds": offset_seconds,
        "source": used_url,
        "reason": "；".join(reason) if reason else "",
        "steps": steps,
    }

@app.post("/api/admin/regenerate-predictions")
def regenerate_predictions(request: Request, db: Session = Depends(get_db)):
    """管理员手动触发重新推算"""
    require_admin_api(request, db)
    
    try:
        # 获取最新开奖记录
        ssq_latest = db.query(models.LotteryRecord).filter(
            models.LotteryRecord.lottery_type == "ssq"
        ).order_by(models.LotteryRecord.issue.desc()).first()
        
        dlt_latest = db.query(models.LotteryRecord).filter(
            models.LotteryRecord.lottery_type == "dlt"
        ).order_by(models.LotteryRecord.issue.desc()).first()
        
        if not ssq_latest and not dlt_latest:
            raise HTTPException(status_code=400, detail="暂无开奖数据，无法推算")
        
        # 清除现有推算记录
        deleted_count = 0
        if ssq_latest:
            deleted = db.query(models.PredictionRecord).filter(
                models.PredictionRecord.lottery_type == "ssq",
                models.PredictionRecord.based_on_issue == ssq_latest.issue
            ).delete()
            deleted_count += deleted
            
        if dlt_latest:
            deleted = db.query(models.PredictionRecord).filter(
                models.PredictionRecord.lottery_type == "dlt",
                models.PredictionRecord.based_on_issue == dlt_latest.issue
            ).delete()
            deleted_count += deleted
            
        db.commit()
        
        # 触发新的推算
        results = []
        
        if ssq_latest:
            created, meta = generate_predictions_for_cycle(db, "ssq", based_on_issue=ssq_latest.issue, count=20)
            if not bool((meta or {}).get("used_llm")):
                raise HTTPException(status_code=500, detail="推算未调用AI大模型")
            model = (meta or {}).get("model") or ""
            results.append(f"双色球: {len(created)}组" + (f" · {model}" if model else ""))
                
        if dlt_latest:
            created, meta = generate_predictions_for_cycle(db, "dlt", based_on_issue=dlt_latest.issue, count=20)
            if not bool((meta or {}).get("used_llm")):
                raise HTTPException(status_code=500, detail="推算未调用AI大模型")
            model = (meta or {}).get("model") or ""
            results.append(f"大乐透: {len(created)}组" + (f" · {model}" if model else ""))
        
        message = f"已清除 {deleted_count} 条旧记录，重新生成推算：" + "，".join(results)
        return {"message": message}
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Regenerate predictions error: {str(e)}")
        raise HTTPException(status_code=500, detail=f"重新推算失败: {str(e)}")

@app.post("/admin/action/sync-time")
def admin_action_sync_time(request: Request, db: Session = Depends(get_db)):
    if not is_admin_session(request, db):
        return RedirectResponse(url="/admin/login", status_code=303)
    try:
        data = sync_server_time(request, db)
        ok = bool((data or {}).get("ok"))
        if ok:
            steps = (data or {}).get("steps") or []
            last = steps[-4:] if isinstance(steps, list) else []
            offset_seconds = (data or {}).get("offset_seconds")
            source = (data or {}).get("source") or ""
            extra = ""
            try:
                extra = f"{int(offset_seconds)}秒" if offset_seconds is not None else ""
            except Exception:
                extra = str(offset_seconds or "")
            text = "校准成功：已记录与网络时间偏移" + (f" {extra}" if extra else "") + "（应用内生效，不修改系统时间）"
            if source:
                text += f"\n来源：{source}"
            if last:
                text += "\n\n" + json.dumps(last, ensure_ascii=False, indent=2)
            request.session["admin_flash"] = {"kind": "msg", "text": text}
        else:
            reason = (data or {}).get("reason") or "执行失败"
            steps = (data or {}).get("steps") or []
            last = steps[-4:] if isinstance(steps, list) else []
            text = "校准失败：" + str(reason)
            if last:
                text += "\n\n" + json.dumps(last, ensure_ascii=False, indent=2)
            request.session["admin_flash"] = {"kind": "err", "text": text}
    except HTTPException as e:
        request.session["admin_flash"] = {"kind": "err", "text": "校准失败：" + str(e.detail or "")}
    except Exception as e:
        request.session["admin_flash"] = {"kind": "err", "text": "校准失败：" + str(e)}
    return RedirectResponse(url="/admin", status_code=303)

@app.post("/admin/action/regenerate-predictions")
def admin_action_regenerate_predictions(request: Request, db: Session = Depends(get_db)):
    if not is_admin_session(request, db):
        return RedirectResponse(url="/admin/login", status_code=303)
    try:
        data = regenerate_predictions(request, db)
        message = ""
        if isinstance(data, dict):
            message = str(data.get("message") or "")
        request.session["admin_flash"] = {"kind": "msg", "text": "重新推算成功：" + message}
    except HTTPException as e:
        request.session["admin_flash"] = {"kind": "err", "text": "重新推算失败：" + str(e.detail or "")}
    except Exception as e:
        request.session["admin_flash"] = {"kind": "err", "text": "重新推算失败：" + str(e)}
    return RedirectResponse(url="/admin", status_code=303)

def _admin_backfill_one(request: Request, db: Session, lottery_type: str, backfill_issue: str = ""):
    backfill_issue = (backfill_issue or "").strip()
    before_latest = (
        db.query(models.LotteryRecord)
        .filter_by(lottery_type=lottery_type)
        .order_by(models.LotteryRecord.issue.desc())
        .first()
    )
    before_issue = before_latest.issue if before_latest else ""

    limit = 2000 if not backfill_issue else 2000
    data = scrape_data(lottery_type, limit=limit, upsert=True, issue=backfill_issue, db=db)
    message = ""
    if isinstance(data, dict):
        message = str(data.get("message") or "")

    if backfill_issue:
        got = (
            db.query(models.LotteryRecord)
            .filter_by(lottery_type=lottery_type, issue=backfill_issue)
            .first()
        )
        if not got:
            request.session["admin_flash"] = {
                "kind": "err",
                "text": f"补抓完成但未找到期号 {backfill_issue}（已尝试抓取最近 {limit} 期）\n{message}".strip(),
            }
            return
        got_date = got.date.isoformat() if getattr(got, "date", None) else ""
        request.session["admin_flash"] = {
            "kind": "msg",
            "text": f"补抓完成：已获取期号 {backfill_issue}" + (f"（{got_date}）" if got_date else "") + "\n" + (message or ""),
        }
        return

    after_latest = (
        db.query(models.LotteryRecord)
        .filter_by(lottery_type=lottery_type)
        .order_by(models.LotteryRecord.issue.desc())
        .first()
    )
    after_issue = after_latest.issue if after_latest else ""
    after_date = after_latest.date.isoformat() if getattr(after_latest, "date", None) else ""

    extra = ""
    if after_issue:
        extra = f"\n最新期：{after_issue}" + (f"（{after_date}）" if after_date else "")
        if before_issue and after_issue != before_issue:
            extra = "\n新增最新期：" + after_issue + (f"（{after_date}）" if after_date else "")
    request.session["admin_flash"] = {"kind": "msg", "text": "补抓完成：" + (message or "已执行") + extra}

@app.post("/admin/action/scrape-ssq")
def admin_action_scrape_ssq(request: Request, backfill_issue: str = Form(""), db: Session = Depends(get_db)):
    if not is_admin_session(request, db):
        return RedirectResponse(url="/admin/login", status_code=303)
    try:
        _admin_backfill_one(request, db, "ssq", backfill_issue=backfill_issue)
    except HTTPException as e:
        request.session["admin_flash"] = {"kind": "err", "text": "补抓失败：" + str(e.detail or "")}
    except Exception as e:
        request.session["admin_flash"] = {"kind": "err", "text": "补抓失败：" + str(e)}
    return RedirectResponse(url="/admin", status_code=303)

@app.post("/admin/action/scrape-dlt")
def admin_action_scrape_dlt(request: Request, backfill_issue: str = Form(""), db: Session = Depends(get_db)):
    if not is_admin_session(request, db):
        return RedirectResponse(url="/admin/login", status_code=303)
    try:
        _admin_backfill_one(request, db, "dlt", backfill_issue=backfill_issue)
    except HTTPException as e:
        request.session["admin_flash"] = {"kind": "err", "text": "补抓失败：" + str(e.detail or "")}
    except Exception as e:
        request.session["admin_flash"] = {"kind": "err", "text": "补抓失败：" + str(e)}
    return RedirectResponse(url="/admin", status_code=303)

@app.post("/admin/action/test-llm")
def admin_action_test_llm(
    request: Request,
    llm_api_key: str = Form(""),
    llm_base_url: str = Form(""),
    llm_model: str = Form(""),
    db: Session = Depends(get_db),
):
    if not is_admin_session(request, db):
        return RedirectResponse(url="/admin/login", status_code=303)
    try:
        llm_api_key = (llm_api_key or "").strip()
        llm_model = (llm_model or "").strip()
        payload = SettingsUpdate(
            llm_api_key=(llm_api_key if llm_api_key else None),
            llm_base_url=(llm_base_url or ""),
            llm_model=(llm_model if llm_model else None),
        )
        data = test_llm(payload, request, db)
        model = ""
        latency_ms = 0
        if isinstance(data, dict):
            model = str(data.get("model") or "")
            try:
                latency_ms = int(data.get("latency_ms") or 0)
            except Exception:
                latency_ms = 0
        request.session["admin_flash"] = {"kind": "msg", "text": f"测试成功：{model} · {latency_ms}ms".strip()}
    except HTTPException as e:
        request.session["admin_flash"] = {"kind": "err", "text": "测试失败：" + str(e.detail or "")}
    except Exception as e:
        request.session["admin_flash"] = {"kind": "err", "text": "测试失败：" + str(e)}
    return RedirectResponse(url="/admin", status_code=303)

@app.post("/admin/action/save-settings")
def admin_action_save_settings(
    request: Request,
    llm_api_key: str = Form(""),
    llm_base_url: str = Form(""),
    llm_model: str = Form(""),
    db: Session = Depends(get_db),
):
    if not is_admin_session(request, db):
        return RedirectResponse(url="/admin/login", status_code=303)
    try:
        llm_api_key = (llm_api_key or "").strip()
        llm_model = (llm_model or "").strip()
        payload = SettingsUpdate(
            llm_api_key=(llm_api_key if llm_api_key else None),
            llm_base_url=llm_base_url or "",
            llm_model=(llm_model if llm_model else None),
        )
        update_settings(payload, request, db)
        request.session["admin_flash"] = {"kind": "msg", "text": "保存成功"}
    except HTTPException as e:
        request.session["admin_flash"] = {"kind": "err", "text": "保存失败：" + str(e.detail or "")}
    except Exception as e:
        request.session["admin_flash"] = {"kind": "err", "text": "保存失败：" + str(e)}
    return RedirectResponse(url="/admin", status_code=303)

@app.post("/admin/action/change-password")
def admin_action_change_password(
    request: Request,
    old_password: str = Form(""),
    new_password: str = Form(""),
    db: Session = Depends(get_db),
):
    if not is_admin_session(request, db):
        return RedirectResponse(url="/admin/login", status_code=303)
    try:
        payload = AdminPasswordChange(old_password=old_password or "", new_password=new_password or "")
        change_admin_password(payload, request, db)
        request.session["admin_flash"] = {"kind": "msg", "text": "密码修改成功，请牢记新密码"}
    except HTTPException as e:
        request.session["admin_flash"] = {"kind": "err", "text": "修改失败：" + str(e.detail or "")}
    except Exception as e:
        request.session["admin_flash"] = {"kind": "err", "text": "修改失败：" + str(e)}
    return RedirectResponse(url="/admin", status_code=303)

@app.post("/admin/action/restart-backend")
def admin_action_restart_backend(request: Request, db: Session = Depends(get_db)):
    if not is_admin_session(request, db):
        return RedirectResponse(url="/admin/login", status_code=303)

    in_docker = False
    try:
        in_docker = os.path.exists("/.dockerenv") or os.path.exists("/run/.containerenv") or bool(os.getenv("DOCKER"))
    except Exception:
        in_docker = False
    if in_docker or os.getpid() == 1:
        request.session["admin_flash"] = {"kind": "msg", "text": "已执行重载（Docker 环境不进行进程自重启）"}
        return RedirectResponse(url="/admin", status_code=303)

    project_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    port = 8888
    try:
        port = int(os.getenv("PORT") or os.getenv("APP_PORT") or "8888")
    except Exception:
        port = 8888
    log_path = os.path.join(project_dir, f"uvicorn-{port}.log")
    pid_path = os.path.join(project_dir, f"uvicorn-{port}.pid")
    host = os.getenv("HOST") or "0.0.0.0"

    helper = r"""
import os, sys, time, subprocess, signal, socket
pid = int(sys.argv[1])
project_dir = sys.argv[2]
host = sys.argv[3]
port = int(sys.argv[4])
log_path = sys.argv[5]
pid_path = sys.argv[6]
delay_s = float(sys.argv[7])
time.sleep(delay_s)
try:
    os.kill(pid, signal.SIGTERM)
except Exception:
    pass
deadline = time.time() + 12.0
while time.time() < deadline:
    try:
        os.kill(pid, 0)
        time.sleep(0.2)
    except Exception:
        break
else:
    try:
        os.kill(pid, signal.SIGKILL)
    except Exception:
        pass
    time.sleep(0.4)
def port_open() -> bool:
    try:
        s = socket.create_connection(("127.0.0.1", port), timeout=0.5)
        try:
            s.close()
        except Exception:
            pass
        return True
    except Exception:
        return False

deadline = time.time() + 12.0
while time.time() < deadline:
    if port_open():
        raise SystemExit(0)
    time.sleep(0.3)

out = open(log_path, "a", encoding="utf-8", errors="ignore")
cmd = [
    sys.executable,
    "-m",
    "uvicorn",
    "backend.main:app",
    "--app-dir",
    project_dir,
    "--host",
    host,
    "--port",
    str(port),
]
p = subprocess.Popen(cmd, stdout=out, stderr=out, start_new_session=True)
try:
    with open(pid_path, "w", encoding="utf-8", errors="ignore") as f:
        f.write(str(p.pid))
except Exception:
    pass
"""
    try:
        subprocess.Popen(
            [sys.executable, "-c", helper, str(os.getpid()), project_dir, host, str(port), log_path, pid_path, "1.6"],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            start_new_session=True,
            close_fds=True,
        )
    except Exception:
        threading.Thread(target=lambda: os._exit(0), daemon=True).start()
    return HTMLResponse(
        """
<!doctype html>
<html lang="zh-CN">
  <head>
    <meta charset="UTF-8" />
    <meta name="viewport" content="width=device-width, initial-scale=1" />
    <title>重启中…</title>
    <style>
      :root { color-scheme: dark; }
      body { margin:0; font-family: ui-sans-serif,system-ui,-apple-system,Segoe UI,Roboto,Helvetica,Arial; background: radial-gradient(1200px 600px at 20% 0%, #1f2937 0%, #0b1220 55%, #060913 100%); color: #e5e7eb; }
      .wrap { max-width: 780px; margin: 0 auto; padding: 24px; }
      .card { background: rgba(255,255,255,.06); border: 1px solid rgba(255,255,255,.10); border-radius: 18px; padding: 18px; box-shadow: 0 30px 80px rgba(0,0,0,.35); backdrop-filter: blur(10px); }
      h1 { font-size: 18px; margin: 0 0 10px; }
      p { margin: 0; color: rgba(229,231,235,.80); font-size: 13px; line-height: 1.6; }
      a { color: rgba(147,197,253,.95); text-decoration: none; font-weight: 700; }
      .hint { margin-top: 12px; font-size: 12px; color: rgba(229,231,235,.62); }
    </style>
  </head>
  <body>
    <div class="wrap">
      <div class="card">
        <h1>已触发重启</h1>
        <p>后端服务正在重启中。页面短暂不可用属于正常现象，约数秒后恢复。</p>
        <div class="hint">恢复后将自动返回后台管理页：<a href="/admin">/admin</a></div>
      </div>
    </div>
    <script>
      const maxTry = 90
      let tries = 0
      const tick = async () => {
        tries += 1
        try {
          const r = await fetch('/api/latest/dlt', { cache: 'no-store' })
          if (r && r.ok) {
            location.href = '/admin'
            return
          }
        } catch (e) {}
        if (tries < maxTry) setTimeout(tick, 700)
      }
      setTimeout(tick, 900)
    </script>
  </body>
</html>
        """.strip()
    )


@app.post("/api/admin/restart-backend")
def api_admin_restart_backend(request: Request, db: Session = Depends(get_db)):
    require_admin_api(request, db)
    in_docker = False
    try:
        in_docker = os.path.exists("/.dockerenv") or os.path.exists("/run/.containerenv") or bool(os.getenv("DOCKER"))
    except Exception:
        in_docker = False
    if in_docker or os.getpid() == 1:
        return {"ok": True, "mode": "reload"}
    admin_action_restart_backend(request, db)
    return {"ok": True, "mode": "restart"}


@app.get("/admin/login", response_class=HTMLResponse)
def admin_login_page():
    return """
<!doctype html>
<html lang="zh-CN">
  <head>
    <meta charset="UTF-8" />
    <meta name="viewport" content="width=device-width, initial-scale=1" />
    <title>魔力彩票助手 - 后台登录</title>
    <style>
      :root { color-scheme: dark; }
      body { margin:0; font-family: ui-sans-serif,system-ui,-apple-system,Segoe UI,Roboto,Helvetica,Arial; background: radial-gradient(1200px 600px at 20% 0%, #1f2937 0%, #0b1220 55%, #060913 100%); color: #e5e7eb; }
      .wrap { max-width: 560px; margin: 0 auto; padding: 28px 18px; }
      .card { background: rgba(255,255,255,.06); border: 1px solid rgba(255,255,255,.10); border-radius: 18px; padding: 18px; box-shadow: 0 30px 80px rgba(0,0,0,.35); backdrop-filter: blur(10px); }
      h1 { font-size: 20px; margin: 0 0 6px; }
      p { margin: 0; color: rgba(229,231,235,.75); font-size: 13px; }
      label { display:block; font-size: 12px; color: rgba(229,231,235,.85); margin: 14px 0 6px; }
      input { width: 100%; box-sizing: border-box; padding: 10px 12px; border-radius: 12px; border: 1px solid rgba(255,255,255,.14); background: rgba(255,255,255,.06); color: #e5e7eb; outline: none; }
      input:focus { border-color: rgba(255,255,255,.25); box-shadow: 0 0 0 4px rgba(255,255,255,.08); }
      .row { display:flex; gap: 10px; align-items:center; margin-top: 16px; }
      button { border: 0; padding: 10px 14px; border-radius: 12px; font-weight: 800; cursor: pointer; background: #ffffff; color: #0b1220; width: 100%; }
      .hint { margin-top: 12px; font-size: 12px; color: rgba(229,231,235,.62); }
      .mono { font-family: ui-monospace, SFMono-Regular, Menlo, Monaco, Consolas, 'Liberation Mono', 'Courier New', monospace; }
    </style>
  </head>
  <body>
    <div class="wrap">
      <div class="card">
        <h1>魔力彩票助手：后台登录</h1>
        <p>登录后才可进行 AI 配置。</p>
        <form method="post" action="/admin/login">
          <label>用户名</label>
          <input name="username" autocomplete="username" value="admin" />
          <label>密码</label>
          <input name="password" type="password" autocomplete="current-password" />
          <div class="row">
            <button type="submit">登录</button>
          </div>
        </form>
        <div class="hint">默认账号：<span class="mono">admin</span> / <span class="mono">admin</span>（建议首次登录后修改密码）</div>
      </div>
    </div>
  </body>
</html>
    """

@app.post("/admin/login")
def admin_login(request: Request, username: str = Form(...), password: str = Form(...), db: Session = Depends(get_db)):
    if verify_admin_credentials(db, username, password):
        request.session["admin"] = username
        return RedirectResponse(url="/admin", status_code=303)
    return RedirectResponse(url="/admin/login", status_code=303)

@app.get("/admin/logout")
def admin_logout(request: Request):
    request.session.pop("admin", None)
    return RedirectResponse(url="/admin/login", status_code=303)

@app.get("/admin", response_class=HTMLResponse)
def admin_page(request: Request, db: Session = Depends(get_db)):
    if not is_admin_session(request, db):
        return RedirectResponse(url="/admin/login", status_code=303)
    html = """
<!doctype html>
<html lang="zh-CN">
  <head>
    <meta charset="UTF-8" />
    <meta name="viewport" content="width=device-width, initial-scale=1" />
    <title>魔力彩票助手 - 后台管理</title>
    <style>
      :root { color-scheme: dark; }
      body { margin:0; font-family: ui-sans-serif,system-ui,-apple-system,Segoe UI,Roboto,Helvetica,Arial; background: radial-gradient(1200px 600px at 20% 0%, #1f2937 0%, #0b1220 55%, #060913 100%); color: #e5e7eb; }
      .wrap { max-width: 1080px; margin: 0 auto; padding: 24px; }
      .card { background: rgba(255,255,255,.06); border: 1px solid rgba(255,255,255,.10); border-radius: 18px; padding: 18px; box-shadow: 0 30px 80px rgba(0,0,0,.35); backdrop-filter: blur(10px); }
      h1 { font-size: 22px; margin: 0 0 6px; }
      p { margin: 0; color: rgba(229,231,235,.75); font-size: 13px; }
      .head { display:flex; justify-content: space-between; align-items: flex-start; gap: 14px; }
      .head .rt { display:flex; align-items:center; gap: 10px; flex-wrap: wrap; justify-content: flex-end; }
      .layout { display:grid; grid-template-columns: 1fr; gap: 12px; margin-top: 14px; }
      @media (min-width: 920px) { .layout { grid-template-columns: 1.1fr 0.9fr; } }
      .stack { display:flex; flex-direction: column; gap: 12px; }
      .panel { background: rgba(255,255,255,.05); border: 1px solid rgba(255,255,255,.10); border-radius: 16px; padding: 14px; }
      .panel-title { font-size: 14px; font-weight: 900; margin: 0; color: rgba(255,255,255,.92); display:flex; align-items:center; gap: 10px; }
      .panel-title .dot { width: 8px; height: 8px; border-radius: 99px; background: rgba(255,255,255,.25); }
      .panel-desc { margin: 6px 0 0; font-size: 12px; color: rgba(229,231,235,.70); }
      .field-grid { display:grid; grid-template-columns: 1fr; gap: 10px; margin-top: 12px; }
      @media (min-width: 720px) { .field-grid.cols-2 { grid-template-columns: 1fr 1fr; } }
      label { display:block; font-size: 12px; color: rgba(229,231,235,.85); margin: 10px 0 6px; }
      input { width: 100%; box-sizing: border-box; padding: 10px 12px; border-radius: 12px; border: 1px solid rgba(255,255,255,.14); background: rgba(255,255,255,.06); color: #e5e7eb; outline: none; }
      input:focus { border-color: rgba(255,255,255,.25); box-shadow: 0 0 0 4px rgba(255,255,255,.08); }
      .row { display:flex; gap: 10px; flex-wrap: wrap; margin-top: 12px; }
      button { border: 0; padding: 10px 14px; border-radius: 12px; font-weight: 700; cursor: pointer; }
      .primary { background: #ffffff; color: #0b1220; }
      .ghost { background: rgba(255,255,255,.08); color: #e5e7eb; border: 1px solid rgba(255,255,255,.12); }
      .danger { background: rgba(244,63,94,.16); color: rgba(253,164,175,.95); border: 1px solid rgba(244,63,94,.28); }
      .pill { display:inline-flex; align-items:center; gap: 8px; padding: 10px 14px; border-radius: 12px; font-weight: 800; border: 1px solid rgba(255,255,255,.12); text-decoration:none; }
      .msg { margin-top: 10px; font-size: 13px; color: rgba(167,243,208,.95); display:none; }
      .err { margin-top: 10px; font-size: 13px; color: rgba(253,164,175,.95); display:none; white-space: pre-wrap; }
      .hint { margin-top: 12px; font-size: 12px; color: rgba(229,231,235,.62); }
      .mono { font-family: ui-monospace, SFMono-Regular, Menlo, Monaco, Consolas, 'Liberation Mono', 'Courier New', monospace; }
    </style>
  </head>
  <body>
    <div class="wrap">
      <div class="card">
        <div class="head">
          <div>
            <h1>魔力彩票助手：后台管理</h1>
            <p>AI 配置 · 数据运维 · 安全设置</p>
          </div>
          <div class="rt">
            <a href="/" class="pill ghost" target="_blank" rel="noopener noreferrer" title="打开前台首页">前台首页</a>
            <form method="post" action="/admin/action/restart-backend" style="margin:0">
              <button type="submit" class="pill danger" id="restartBackendTop" title="重启后端服务">重启应用</button>
            </form>
            <a href="/admin/logout" class="pill ghost" id="logout" title="退出后台">退出</a>
          </div>
        </div>

        <div class="msg" id="msg"></div>
        <div class="err" id="err"></div>

        <div class="layout">
          <div class="stack">
            <div class="panel">
              <div class="panel-title"><span class="dot"></span>AI 配置</div>
              <div class="panel-desc">用于生成每期开奖后的 20 组预测号码</div>
              <form method="post" action="/admin/action/save-settings" style="margin: 0;">
                <div class="field-grid cols-2">
                  <div>
                    <label>LLM API Key</label>
                    <input id="k" name="llm_api_key" type="password" placeholder="sk-..." />
                  </div>
                  <div>
                    <label>LLM Base URL（可选）</label>
                    <input id="b" name="llm_base_url" placeholder="https://api.siliconflow.cn/v1" />
                  </div>
                </div>
                <div style="margin-top: 10px;">
                  <label>LLM 模型名</label>
                  <input id="m" name="llm_model" placeholder="deepseek-ai/DeepSeek-V3" />
                  <div class="hint">示范：<span class="mono">Base URL=https://api.siliconflow.cn/v1</span>，<span class="mono">Model=deepseek-ai/DeepSeek-V3</span></div>
                </div>
                <div class="row">
                  <button type="submit" class="ghost" id="test" formaction="/admin/action/test-llm">测试配置</button>
                  <button type="submit" class="primary" id="save" formaction="/admin/action/save-settings">保存</button>
                </div>
              </form>
            </div>

            <div class="panel">
              <div class="panel-title"><span class="dot"></span>数据运维</div>
              <div class="panel-desc">补抓开奖 · 校准时间 · 重新推算</div>
              <form method="post" action="/admin/action/scrape-dlt" style="margin: 0;">
                <div class="field-grid cols-2">
                  <div>
                    <label>补抓期号（可选）</label>
                    <input name="backfill_issue" placeholder="如 26012" />
                    <div class="hint">留空表示按近期自动补齐</div>
                  </div>
                  <div>
                    <label>快捷动作</label>
                    <div class="row" style="margin-top: 0;">
                      <button type="submit" class="ghost" id="syncTime" formaction="/admin/action/sync-time">校准时间(上海)</button>
                      <button type="submit" class="ghost" id="regenerate" formaction="/admin/action/regenerate-predictions">重新推算</button>
                    </div>
                  </div>
                </div>
                <div class="row">
                  <button type="submit" class="ghost" id="scrapeSsq" formaction="/admin/action/scrape-ssq">补抓双色球开奖</button>
                  <button type="submit" class="ghost" id="scrapeDlt" formaction="/admin/action/scrape-dlt">补抓大乐透开奖</button>
                </div>
              </form>
            </div>
          </div>

          <div class="stack">
            <div class="panel">
              <div class="panel-title"><span class="dot"></span>安全设置</div>
              <div class="panel-desc">建议首次登录后修改密码</div>
              <form method="post" action="/admin/action/change-password" style="margin: 0;">
                <div class="field-grid">
                  <div>
                    <label>原密码</label>
                    <input id="oldpw" name="old_password" type="password" autocomplete="current-password" />
                  </div>
                  <div>
                    <label>新密码（至少 8 位）</label>
                    <input id="newpw" name="new_password" type="password" autocomplete="new-password" />
                  </div>
                </div>
                <div class="row">
                  <button type="submit" class="primary" id="changepw">修改密码</button>
                </div>
              </form>
              <div class="hint">
                默认账号：<span class="mono">admin</span> / <span class="mono">admin</span>（建议通过环境变量修改用户名：<span class="mono">ADMIN_USERNAME</span>；会话密钥：<span class="mono">SESSION_SECRET</span>）
              </div>
            </div>
          </div>
        </div>

      </div>
    </div>

    <script>
      const $ = (id) => document.getElementById(id)
      const showPanel = (kind, text) => {
        try {
          const el = $(kind)
          const other = kind === 'msg' ? $('err') : $('msg')
          if (!el || !other) {
            alert(String(text || ''))
            return
          }
          el.textContent = text
          el.style.display = 'block'
          other.style.display = 'none'
          try {
            if (typeof el.scrollIntoView === 'function') el.scrollIntoView()
          } catch (e3) {}
        } catch (e) {
          try { alert(String(text || '')) } catch (e2) {}
        }
      }
      const msg = (t) => showPanel('msg', t)
      const err = (t) => showPanel('err', t)

      async function pushNotice(title, body) {
        try {
          if (!('Notification' in window)) return
          if (Notification.permission === 'granted') {
            new Notification(title, { body })
            return
          }
          if (Notification.permission === 'default') {
            const perm = await Notification.requestPermission()
            if (perm === 'granted') new Notification(title, { body })
          }
        } catch (e) {}
      }

      async function loadSettings() {
        try {
          const res = await fetch('/api/settings', { credentials: 'same-origin' })
          const data = await res.json()
          if (!res.ok) throw new Error(data.detail || ('HTTP ' + res.status))
          $('k').value = ''
          try {
            $('k').placeholder = '不会回显，留空表示不修改'
          } catch (e) {}
          $('b').value = data.llm_base_url || ''
          $('m').value = data.llm_model || ''
          const hasFlash =
            (($('msg') && $('msg').style.display === 'block' && ($('msg').textContent || '').trim()) ||
             ($('err') && $('err').style.display === 'block' && ($('err').textContent || '').trim()))
          if (!hasFlash) msg('读取成功')
        } catch (e) {
          err('读取失败：' + (e && e.message ? e.message : String(e)))
        }
      }

      async function testSettings() {
        try {
          const payload = {
            llm_base_url: $('b').value || '',
          }
          if (($('k').value || '').trim()) payload.llm_api_key = $('k').value
          if (($('m').value || '').trim()) payload.llm_model = $('m').value
          const res = await fetch('/api/admin/test-llm', {
            method: 'POST',
            credentials: 'same-origin',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify(payload)
          })
          const data = await res.json()
          if (!res.ok) throw new Error(data.detail || ('HTTP ' + res.status))
          msg('测试成功：' + (data.model || '') + ' · ' + (data.latency_ms || 0) + 'ms')
        } catch (e) {
          err('测试失败：' + (e && e.message ? e.message : String(e)))
        }
      }

      async function saveSettings() {
        try {
          const payload = {
            llm_base_url: $('b').value || '',
          }
          if (($('k').value || '').trim()) payload.llm_api_key = $('k').value
          if (($('m').value || '').trim()) payload.llm_model = $('m').value
          const res = await fetch('/api/settings', {
            method: 'POST',
            credentials: 'same-origin',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify(payload)
          })
          const data = await res.json()
          if (!res.ok) throw new Error(data.detail || ('HTTP ' + res.status))
          await loadSettings()
          msg('保存成功')
        } catch (e) {
          err('保存失败：' + (e && e.message ? e.message : String(e)))
        }
      }

      async function changePassword() {
        try {
          const oldpw = $('oldpw').value || ''
          const newpw = $('newpw').value || ''
          const res = await fetch('/api/admin/change-password', {
            method: 'POST',
            credentials: 'same-origin',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ old_password: oldpw, new_password: newpw })
          })
          const data = await res.json()
          if (!res.ok) throw new Error(data.detail || ('HTTP ' + res.status))
          $('oldpw').value = ''
          $('newpw').value = ''
          msg('密码修改成功，请牢记新密码')
        } catch (e) {
          err('修改失败：' + (e && e.message ? e.message : String(e)))
        }
      }

      async function regeneratePredictions() {
        try {
          const btn = $('regenerate')
          btn.disabled = true
          btn.textContent = '重新推算中...'
          msg('正在重新推算，请稍候...')
          const res = await fetch('/api/admin/regenerate-predictions', {
            method: 'POST',
            credentials: 'same-origin',
            headers: { 'Content-Type': 'application/json' }
          })
          const data = await res.json()
          if (!res.ok) throw new Error(data.detail || ('HTTP ' + res.status))
          const text = '重新推算成功：' + (data.message || '')
          msg(text)
          pushNotice('重新推算完成', data.message || '')
        } catch (e) {
          const text = '重新推算失败：' + (e && e.message ? e.message : String(e))
          err(text)
          pushNotice('重新推算失败', e && e.message ? e.message : String(e))
        } finally {
          const btn = $('regenerate')
          if (btn) {
            btn.disabled = false
            btn.textContent = '重新推算'
          }
        }
      }

      async function syncTime() {
        try {
          const btn = $('syncTime')
          btn.disabled = true
          btn.textContent = '校准中...'
          msg('正在校准服务器时间，请稍候...')
          const res = await fetch('/api/admin/sync-time', {
            method: 'POST',
            credentials: 'same-origin',
            headers: { 'Content-Type': 'application/json' }
          })
          const data = await res.json().catch(() => ({}))
          if (!res.ok) throw new Error(data.detail || ('HTTP ' + res.status))
          if (!data.ok) {
            const reason = data.reason || '执行失败'
            const last = Array.isArray(data.steps) ? data.steps.slice(-4) : []
            err('校准失败：' + reason + (last.length ? ('\n\n' + JSON.stringify(last, null, 2)) : ''))
            pushNotice('时间校准失败', reason)
            return
          }
          const last = Array.isArray(data.steps) ? data.steps.slice(-4) : []
          msg('校准成功：Asia/Shanghai 已生效，且已执行时间同步' + (last.length ? ('\n\n' + JSON.stringify(last, null, 2)) : ''))
          pushNotice('时间校准完成', '已设置为 Asia/Shanghai，并执行时间同步')
        } catch (e) {
          const text = '校准失败：' + (e && e.message ? e.message : String(e))
          err(text)
          pushNotice('时间校准失败', e && e.message ? e.message : String(e))
        } finally {
          const btn = $('syncTime')
          if (btn) {
            btn.disabled = false
            btn.textContent = '校准时间(上海)'
          }
        }
      }

      const bindClick = (id, fn) => {
        const el = $(id)
        if (!el) return
        try {
          if ((el.tagName || '').toUpperCase() === 'BUTTON') {
            const t = String(el.getAttribute('type') || '').toLowerCase()
            if (t === 'submit') return
          }
        } catch (e) {}
        el.addEventListener('click', (evt) => {
          try { evt.preventDefault() } catch (e) {}
          try { evt.stopPropagation() } catch (e) {}
          fn(evt)
        })
      }
      bindClick('test', () => testSettings())
      bindClick('save', () => saveSettings())
      bindClick('changepw', () => changePassword())
      bindClick('syncTime', () => syncTime())
      bindClick('regenerate', () => regeneratePredictions())
      bindClick('logout', () => { window.location.href = '/admin/logout' })
      loadSettings()
    </script>
  </body>
</html>
    """
    try:
        import html as _html
        flash = request.session.pop("admin_flash", None)
        if isinstance(flash, dict):
            kind = flash.get("kind")
            text = flash.get("text")
            if isinstance(text, str) and text.strip():
                safe_text = _html.escape(text)
                if kind == "msg":
                    html = html.replace(
                        '<div class="msg" id="msg"></div>',
                        f'<div class="msg" id="msg" style="display:block; white-space: pre-wrap">{safe_text}</div>',
                        1,
                    )
                elif kind == "err":
                    html = html.replace(
                        '<div class="err" id="err"></div>',
                        f'<div class="err" id="err" style="display:block">{safe_text}</div>',
                        1,
                    )
    except Exception:
        pass
    return html

# Serve Frontend Static Files
frontend_dist = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "frontend", "dist")
if os.path.exists(frontend_dist):
    app.mount("/", StaticFiles(directory=frontend_dist, html=True), name="static")
else:
    print(f"Frontend dist not found at {frontend_dist}")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8888)
