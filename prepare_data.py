
import fitz

from pathlib import Path

from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_core.documents import Document

from config import PDF_SOURCES, DB_DIR, EMBEDDING_MODEL


def load_pdf_with_metadata(
    file_path: Path,
    bank: str
) -> list[Document]:

    docs = []

    print(f"Читаем файл: {file_path.name}")

    if not file_path.is_file():
        raise FileNotFoundError(
            f"PDF-файл не найден: {file_path}"
        )

    with fitz.open(str(file_path)) as pdf:
        for page_num, page in enumerate(pdf, start=1):
            text = page.get_text("text", sort=True)

            text = text.strip()

            if text.strip():
                metadata = {
                    "source": file_path.name,
                    "bank": bank,
                    "page": page_num,
                }

                docs.append(
                    Document(
                        page_content=text,
                        metadata=metadata
                    )
                )

    return docs


def build_chunks(raw_docs: list[Document]) -> list[Document]:
    """Разбивает страницы, сохраняя метаданные источника."""
    splitter = RecursiveCharacterTextSplitter.from_tiktoken_encoder(
        chunk_size=600, chunk_overlap=100,
        separators=["\n\n", "\n", ".", " ", ""],
        encoding_name="cl100k_base",
    )
    return splitter.split_documents(raw_docs)



def main():
    db_dir = Path(DB_DIR)

    if db_dir.exists():
        raise FileExistsError(
            f"Векторная база уже существует: {db_dir}. "
            "Удалите её вручную или укажите другой DB_DIR."
        )

    raw_docs = []

    for bank, pdf_path in PDF_SOURCES.items():
        pdf_path = Path(pdf_path)

        documents = load_pdf_with_metadata(
            file_path=pdf_path,
            bank=bank
        )

        if not documents:
            raise ValueError(
                f"PDF не содержит извлекаемого текста: {pdf_path}"
            )

        raw_docs.extend(documents)

        print(
            f"{bank}: загружено страниц с текстом: "
            f"{len(documents)}"
        )

    print(f"Всего загружено страниц: {len(raw_docs)}")

    chunks = build_chunks(raw_docs)

    print(f"Создано чанков: {len(chunks)}")

    if not chunks:
        raise ValueError("Не удалось создать чанки")

    print("Загрузка модели эмбеддингов")

    embeddings = HuggingFaceEmbeddings(
        model_name=EMBEDDING_MODEL,
        model_kwargs={"device": "cpu"},
        encode_kwargs={"normalize_embeddings": True}
    )

    print(f"Создаем векторное хранилище в {db_dir}")

    Chroma.from_documents(
        documents=chunks,
        embedding=embeddings,
        persist_directory=str(db_dir)
    )

    print("Векторная база готова")


if __name__ == "__main__":
    main()
