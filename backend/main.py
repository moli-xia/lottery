from fastapi import FastAPI, Depends, HTTPException, Request, status, Form
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse, RedirectResponse, StreamingResponse
from starlette.middleware.sessions import SessionMiddleware
from sqlalchemy.orm import Session
from sqlalchemy import text
from pydantic import BaseModel
from typing import Optional
from datetime import datetime, timedelta, time as dtime
from zoneinfo import ZoneInfo
import asyncio
import base64
import hashlib
import json
import logging
import os
import sys
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

DEFAULT_LLM_BASE_URL = "https://api.siliconflow.cn/v1"
DEFAULT_LLM_MODEL = "deepseek-ai/DeepSeek-R1"

def ensure_default_llm_settings():
    db = database.SessionLocal()
    try:
        base_url = get_setting(db, "llm_base_url")
        model = get_setting(db, "llm_model")
        if not base_url:
            set_setting(db, "llm_base_url", DEFAULT_LLM_BASE_URL)
        if not model:
            set_setting(db, "llm_model", DEFAULT_LLM_MODEL)
    finally:
        db.close()

def hash_password(password: str, salt: Optional[bytes] = None, iterations: int = 120_000) -> str:
    if salt is None:
        salt = secrets.token_bytes(16)
    dk = hashlib.pbkdf2_hmac("sha256", password.encode("utf-8"), salt, iterations)
    salt_b64 = base64.urlsafe_b64encode(salt).decode("utf-8").rstrip("=")
    dk_b64 = base64.urlsafe_b64encode(dk).decode("utf-8").rstrip("=")
    return f"pbkdf2_sha256${iterations}${salt_b64}${dk_b64}"

def verify_password(password: str, stored: str) -> bool:
    try:
        algo, iterations_s, salt_b64, dk_b64 = stored.split("$", 3)
        if algo != "pbkdf2_sha256":
            return False
        iterations = int(iterations_s)
        salt = base64.urlsafe_b64decode(salt_b64 + "==")
        expected = base64.urlsafe_b64decode(dk_b64 + "==")
        actual = hashlib.pbkdf2_hmac("sha256", password.encode("utf-8"), salt, iterations)
        return secrets.compare_digest(actual, expected)
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
    admin_user = os.getenv("ADMIN_USERNAME", "admin")
    if not secrets.compare_digest(username, admin_user):
        return False
    stored = ensure_admin_password_hash(db)
    return verify_password(password, stored)

def is_admin_session(request: Request, db: Session) -> bool:
    admin_user = os.getenv("ADMIN_USERNAME", "admin")
    sess_user = request.session.get("admin")
    return isinstance(sess_user, str) and secrets.compare_digest(sess_user, admin_user)

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

    q = (
        db.query(models.PredictionRecord)
        .filter_by(lottery_type=lottery_type, based_on_issue=latest.issue)
        .filter(models.PredictionRecord.used_llm == True)
        .order_by(models.PredictionRecord.sequence.asc())
        .offset(offset)
        .limit(limit)
        .all()
    )
    items = [
        PredictionOut(
            id=p.id,
            lottery_type=p.lottery_type,
            based_on_issue=p.based_on_issue,
            target_issue=p.target_issue,
            red_balls=p.red_balls,
            blue_balls=p.blue_balls,
            created_at=p.created_at.isoformat() if p.created_at else "",
            evaluated=bool(p.evaluated),
            actual_issue=p.actual_issue,
            red_hits=p.red_hits,
            blue_hits=p.blue_hits,
            total_hits=p.total_hits,
        ).model_dump()
        for p in q
    ]
    if q:
        configured_base_url = get_setting(db, "llm_base_url") or ""
        llm = {
            "used": True,
            "model": q[0].llm_model or (get_setting(db, "llm_model") or None),
            "base_url": configured_base_url or None,
            "latency_ms": q[0].llm_latency_ms,
        }
    else:
        try:
            llm = get_llm_status(lottery_type, db=db)
        except Exception:
            llm = None
    return {"based_on_issue": latest.issue, "items": items, "llm": llm}

@app.post("/api/scrape/{lottery_type}")
def scrape_data(lottery_type: str, db: Session = Depends(get_db)):
    s = scraper.LotteryScraper(db)
    if lottery_type == 'ssq':
        count = s.scrape_ssq(limit=100)
    elif lottery_type == 'dlt':
        count = s.scrape_dlt(limit=100)
    else:
        raise HTTPException(status_code=400, detail="Invalid lottery type")
    evaluate_predictions(db, lottery_type)
    ensure_predictions_for_cycle(db, lottery_type, min_count=20)
    return {"message": f"Scraped {count} new records for {lottery_type}"}

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
            created_at=p.created_at.isoformat() if p.created_at else "",
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
        raise HTTPException(status_code=500, detail=result["error"])
    meta = result.get("meta") or {"used_llm": False}
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

def ensure_predictions_for_cycle(db: Session, lottery_type: str, min_count: int = 20):
    latest = (
        db.query(models.LotteryRecord)
        .filter_by(lottery_type=lottery_type)
        .order_by(models.LotteryRecord.issue.desc())
        .first()
    )
    if not latest:
        return
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
        return
    remaining = min_count - existing
    while remaining > 0:
        batch = 2 if remaining >= 2 else 1
        try:
            generate_predictions_for_cycle(db, lottery_type, based_on_issue=latest.issue, count=batch)
        except Exception:
            break
        remaining -= batch

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
_auto_scrape_state: dict[str, dict] = {}
_latest_issue_events: dict[str, asyncio.Event] = {"ssq": asyncio.Event(), "dlt": asyncio.Event()}

def in_scrape_window(now_cn: datetime, lottery_type: str) -> Optional[tuple[datetime, datetime]]:
    cfg = LOTTERY_SCHEDULE.get(lottery_type)
    if not cfg:
        return None
    if now_cn.weekday() not in cfg["days"]:
        return None
    base = datetime.combine(now_cn.date(), cfg["draw_time"], tzinfo=LOTTERY_TZ) + timedelta(minutes=cfg["delay_min"])
    return (base, base + timedelta(minutes=90))

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
            now_cn = datetime.now(LOTTERY_TZ)
            for lt in ("ssq", "dlt"):
                cfg = LOTTERY_SCHEDULE.get(lt)
                lottery_name = "双色球" if lt == "ssq" else "大乐透"
                
                win = in_scrape_window(now_cn, lt)
                if not win:
                    _auto_scrape_state.pop(lt, None)
                    continue
                    
                win_start, win_end = win
                if not (win_start <= now_cn <= win_end):
                    _auto_scrape_state.pop(lt, None)
                    continue
                    
                st = _auto_scrape_state.get(lt)
                if not st or st.get("win_start") != win_start:
                    logger.info(f"[{lottery_name}] 检测到开奖日且在抓取窗口内（{win_start.strftime('%H:%M')} - {win_end.strftime('%H:%M')}），开始智能抓取...")
                    _auto_scrape_state[lt] = {"win_start": win_start, "baseline_issue": None, "done": False}
                    st = _auto_scrape_state[lt]

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

                    if st.get("done"):
                        continue

                    # 执行抓取
                    s = scraper.LotteryScraper(db)
                    if lt == "ssq":
                        count = s.scrape_ssq(limit=1)
                    else:
                        count = s.scrape_dlt(limit=1)

                    latest_after = (
                        db.query(models.LotteryRecord)
                        .filter_by(lottery_type=lt)
                        .order_by(models.LotteryRecord.issue.desc())
                        .first()
                    )
                    latest_issue = latest_after.issue if latest_after else None
                    
                    # 检查是否抓到新数据
                    if latest_issue and latest_issue != st.get("baseline_issue"):
                        logger.info(f"[{lottery_name}] ✅ 抓取成功！新期号: {latest_issue}")
                        logger.info(f"[{lottery_name}] 开始生成推算...")
                        evaluate_predictions(db, lt)
                        ensure_predictions_for_cycle(db, lt, min_count=20)
                        logger.info(f"[{lottery_name}] 推算生成完成，共20组")
                        st["done"] = True
                        _notify_latest_issue(lt)
                    else:
                        # 检查是否即将超时
                        remaining_minutes = (win_end - now_cn).total_seconds() / 60
                        if remaining_minutes < 10 and count == 0:
                            logger.warning(f"[{lottery_name}] 抓取窗口即将结束（剩余{remaining_minutes:.0f}分钟），但尚未获取到新期数据")
                finally:
                    db.close()
                    
            # 在窗口结束时检查是否有失败的任务
            for lt in list(_auto_scrape_state.keys()):
                st = _auto_scrape_state.get(lt)
                if st and not st.get("done"):
                    win_start = st.get("win_start")
                    if win_start:
                        win_end = win_start + timedelta(minutes=90)
                        if now_cn > win_end:
                            lottery_name = "双色球" if lt == "ssq" else "大乐透"
                            logger.error(f"[{lottery_name}] ❌ 抓取失败：抓取窗口已结束，未获取到新期数据")
                            _auto_scrape_state.pop(lt, None)
                            
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
        try:
            await asyncio.wait_for(ev.wait(), timeout=3_600)
        except asyncio.TimeoutError:
            yield "event: timeout\ndata: \n\n"
            return
        finally:
            ev.clear()

        latest = (
            db.query(models.LotteryRecord)
            .filter_by(lottery_type=lottery_type)
            .order_by(models.LotteryRecord.issue.desc())
            .first()
        )
        issue = latest.issue if latest else ""
        yield f"event: update\ndata: {issue}\n\n"

    return StreamingResponse(gen(), media_type="text/event-stream")

@app.on_event("startup")
async def on_startup():
    await asyncio.to_thread(ensure_default_llm_settings)

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
    return {s.key: s.value for s in settings if s.key.startswith("llm_")}

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

    saved_api_key = get_setting(db, "llm_api_key") or ""
    saved_base_url = get_setting(db, "llm_base_url") or ""
    saved_model = get_setting(db, "llm_model") or ""

    api_key = settings.llm_api_key if settings.llm_api_key is not None else saved_api_key
    base_url = settings.llm_base_url if settings.llm_base_url is not None else saved_base_url
    model = settings.llm_model if settings.llm_model is not None else saved_model

    if not api_key:
        raise HTTPException(status_code=400, detail="LLM API Key 为空")
    if not model:
        raise HTTPException(status_code=400, detail="LLM 模型名为空")

    t0 = time.monotonic()
    try:
        base = (base_url or "https://api.siliconflow.cn/v1").rstrip("/")
        url = f"{base}/chat/completions"
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
    return """
<!doctype html>
<html lang="zh-CN">
  <head>
    <meta charset="UTF-8" />
    <meta name="viewport" content="width=device-width, initial-scale=1" />
    <title>魔力彩票助手 - 后台管理</title>
    <style>
      :root { color-scheme: dark; }
      body { margin:0; font-family: ui-sans-serif,system-ui,-apple-system,Segoe UI,Roboto,Helvetica,Arial; background: radial-gradient(1200px 600px at 20% 0%, #1f2937 0%, #0b1220 55%, #060913 100%); color: #e5e7eb; }
      .wrap { max-width: 900px; margin: 0 auto; padding: 24px; }
      .card { background: rgba(255,255,255,.06); border: 1px solid rgba(255,255,255,.10); border-radius: 18px; padding: 18px; box-shadow: 0 30px 80px rgba(0,0,0,.35); backdrop-filter: blur(10px); }
      h1 { font-size: 22px; margin: 0 0 6px; }
      p { margin: 0; color: rgba(229,231,235,.75); font-size: 13px; }
      .head { display:flex; justify-content: space-between; align-items: flex-start; gap: 14px; }
      .head .rt { display:flex; align-items:center; gap: 10px; }
      .grid { display: grid; grid-template-columns: 1fr; gap: 12px; margin-top: 14px; }
      @media (min-width: 720px) { .grid { grid-template-columns: 1fr 1fr; } }
      label { display:block; font-size: 12px; color: rgba(229,231,235,.85); margin: 10px 0 6px; }
      input { width: 100%; box-sizing: border-box; padding: 10px 12px; border-radius: 12px; border: 1px solid rgba(255,255,255,.14); background: rgba(255,255,255,.06); color: #e5e7eb; outline: none; }
      input:focus { border-color: rgba(255,255,255,.25); box-shadow: 0 0 0 4px rgba(255,255,255,.08); }
      .row { display:flex; gap: 10px; flex-wrap: wrap; margin-top: 14px; }
      button { border: 0; padding: 10px 14px; border-radius: 12px; font-weight: 700; cursor: pointer; }
      .primary { background: #ffffff; color: #0b1220; }
      .ghost { background: rgba(255,255,255,.08); color: #e5e7eb; border: 1px solid rgba(255,255,255,.12); }
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
            <p>该页面用于配置大模型 API（需先登录）。</p>
          </div>
          <div class="rt">
            <button class="ghost" id="logout" title="退出后台">退出</button>
          </div>
        </div>

        <div class="grid">
          <div>
            <label>LLM API Key</label>
            <input id="k" type="password" placeholder="sk-..." />
          </div>
          <div>
            <label>LLM Base URL（可选）</label>
            <input id="b" placeholder="https://api.siliconflow.cn/v1" />
          </div>
        </div>

        <div>
          <label>LLM 模型名</label>
          <input id="m" placeholder="deepseek-ai/DeepSeek-R1" />
          <div class="hint">示范配置：<span class="mono">Base URL=https://api.siliconflow.cn/v1</span>，<span class="mono">Model=deepseek-ai/DeepSeek-R1</span></div>
        </div>

        <div class="row">
          <button class="ghost" id="test">测试配置</button>
          <button class="primary" id="save">保存</button>
        </div>

        <div class="msg" id="msg"></div>
        <div class="err" id="err"></div>

        <div class="grid" style="margin-top: 18px;">
          <div>
            <label>原密码</label>
            <input id="oldpw" type="password" autocomplete="current-password" />
          </div>
          <div>
            <label>新密码（至少 8 位）</label>
            <input id="newpw" type="password" autocomplete="new-password" />
          </div>
        </div>
        <div class="row">
          <button class="primary" id="changepw">修改密码</button>
        </div>

        <div class="hint">
          默认账号：<span class="mono">admin</span> / <span class="mono">admin</span>（建议通过环境变量修改用户名：<span class="mono">ADMIN_USERNAME</span>；会话密钥：<span class="mono">SESSION_SECRET</span>）
        </div>
      </div>
    </div>

    <script>
      const $ = (id) => document.getElementById(id)
      const msg = (t) => { $('msg').textContent = t; $('msg').style.display = 'block'; $('err').style.display = 'none'; }
      const err = (t) => { $('err').textContent = t; $('err').style.display = 'block'; $('msg').style.display = 'none'; }

      async function loadSettings() {
        try {
          const res = await fetch('/api/settings')
          const data = await res.json()
          if (!res.ok) throw new Error(data.detail || ('HTTP ' + res.status))
          $('k').value = data.llm_api_key || ''
          $('b').value = data.llm_base_url || ''
          $('m').value = data.llm_model || ''
          msg('读取成功')
        } catch (e) {
          err('读取失败：' + (e && e.message ? e.message : String(e)))
        }
      }

      async function testSettings() {
        try {
          const payload = {
            llm_api_key: $('k').value || '',
            llm_base_url: $('b').value || '',
            llm_model: $('m').value || ''
          }
          const res = await fetch('/api/admin/test-llm', {
            method: 'POST',
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
            llm_api_key: $('k').value || '',
            llm_base_url: $('b').value || '',
            llm_model: $('m').value || ''
          }
          const res = await fetch('/api/settings', {
            method: 'POST',
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

      $('test').addEventListener('click', testSettings)
      $('save').addEventListener('click', saveSettings)
      $('changepw').addEventListener('click', changePassword)
      $('logout').addEventListener('click', () => { window.location.href = '/admin/logout' })
      loadSettings()
    </script>
  </body>
</html>
    """

# Serve Frontend Static Files
frontend_dist = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "frontend", "dist")
if os.path.exists(frontend_dist):
    app.mount("/", StaticFiles(directory=frontend_dist, html=True), name="static")
else:
    print(f"Frontend dist not found at {frontend_dist}")
