from sqlalchemy import create_engine
from sqlalchemy.orm import Session, sessionmaker

from app.config import settings

engine = create_engine(settings.database_url, pool_pre_ping=True)
# Use Session directly as context manager: `with SessionLocal() as db:`
SessionLocal = sessionmaker(bind=engine, autocommit=False, autoflush=False, class_=Session)
