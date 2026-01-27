from sqlalchemy import create_engine
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker
import os

_env_url = os.getenv("DATABASE_URL")
if _env_url:
    SQLALCHEMY_DATABASE_URL = _env_url
else:
    db_path = os.getenv("DB_PATH", "./lottery.db")
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
