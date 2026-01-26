from sqlalchemy import Column, Integer, String, Date, DateTime, Text, UniqueConstraint, Boolean
from datetime import datetime
from database import Base

class LotteryRecord(Base):
    __tablename__ = "lottery_records"
    __table_args__ = (UniqueConstraint("lottery_type", "issue", name="uix_lottery_type_issue"),)

    id = Column(Integer, primary_key=True, index=True)
    lottery_type = Column(String, index=True)  # 'ssq' (Double Color Ball) or 'dlt' (Super Lotto)
    issue = Column(String, index=True)
    date = Column(Date)
    red_balls = Column(String)  # Comma separated
    blue_balls = Column(String) # Comma separated
    sales = Column(String, nullable=True)
    pool = Column(String, nullable=True)

class AppSettings(Base):
    __tablename__ = "app_settings"
    
    id = Column(Integer, primary_key=True, index=True)
    key = Column(String, unique=True, index=True)
    value = Column(Text)

class PredictionRecord(Base):
    __tablename__ = "prediction_records"
    __table_args__ = (
        UniqueConstraint("lottery_type", "based_on_issue", "sequence", name="uix_pred_cycle_seq"),
    )

    id = Column(Integer, primary_key=True, index=True)
    lottery_type = Column(String, index=True)

    based_on_issue = Column(String, index=True)
    target_issue = Column(String, index=True, nullable=True)
    sequence = Column(Integer, nullable=False, default=0)

    red_balls = Column(String)
    blue_balls = Column(String)

    used_llm = Column(Boolean, default=False, index=True)
    llm_model = Column(String, nullable=True)
    llm_latency_ms = Column(Integer, nullable=True)

    created_at = Column(DateTime, default=datetime.utcnow, index=True)
    evaluated = Column(Boolean, default=False, index=True)
    actual_issue = Column(String, nullable=True)
    actual_red_balls = Column(String, nullable=True)
    actual_blue_balls = Column(String, nullable=True)
    red_hits = Column(Integer, nullable=True)
    blue_hits = Column(Integer, nullable=True)
    total_hits = Column(Integer, nullable=True)
