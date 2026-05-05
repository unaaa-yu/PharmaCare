import os
from dotenv import load_dotenv
from sqlalchemy import create_engine

load_dotenv()
from sqlalchemy.orm import sessionmaker, DeclarativeBase

database_url = os.environ["DATABASE_URL"]
# Cloud databases (Neon/Supabase) require SSL; local Docker does not
connect_args = {}
if "sslmode" not in database_url and not any(
    h in database_url for h in ["localhost", "127.0.0.1", "db"]
):
    connect_args["sslmode"] = "require"

engine = create_engine(
    database_url,
    connect_args=connect_args,
    pool_pre_ping=True,
)
Session = sessionmaker(bind=engine)


class Base(DeclarativeBase):
    pass


def get_db():
    db = Session()
    try:
        yield db
    finally:
        db.close()
