import os
from pathlib import Path

from dotenv import load_dotenv

BASE_DIR = Path(__file__).resolve().parent

load_dotenv(BASE_DIR / ".env")

def resolve_path(value: str) -> Path:
    path = Path(value).expanduser()
    return path if path.is_absolute() else BASE_DIR / path

PDF_PATH = resolve_path(
    os.getenv("PDF_PATH", "data/tbank_tariff.pdf")
)

DB_DIR = resolve_path(
    os.getenv("DB_DIR", "vector_db")
)

EMBEDDING_MODEL = os.getenv(
    "EMBEDDING_MODEL", "cointegrated/rubert-tiny2"
)

GIGACHAT_MODEL = os.getenv("GIGACHAT_MODEL", "GigaChat")
GIGACHAT_CREDENTIALS = os.getenv("GIGACHAT_CREDENTIALS")