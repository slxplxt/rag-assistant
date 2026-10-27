import fitz
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_core.documents import Document
from pathlib import Path

from config import PDF_PATH, DB_DIR, EMBEDDING_MODEL

def load_pdf_with_metadata(file_path: Path) -> list[Document]:
    docs = []

    print(f"Читаем файл: {file_path}")

    with fitz.open(str(file_path)) as pdf:
        for page_num, page in enumerate(pdf):
            text = page.get_text("text", sort=True)

            if text.strip():
                metadata = {"source": str(file_path), "page": page_num + 1}
                docs.append(Document(page_content=text, metadata=metadata))

    return docs

def main():
    pdf_path = Path(PDF_PATH)
    db_dir = Path(DB_DIR)

    if not pdf_path.is_file():
        raise FileNotFoundError(
            f"PDF-файл не найден: {pdf_path}"
        )

    if db_dir.exists():
        raise FileExistsError(
            f"Векторная база уже существует: {db_dir}. "
            "Удалите её вручную или укажите другой DB_DIR."
        )

    raw_docs = load_pdf_with_metadata(pdf_path)

    if not raw_docs:
        raise ValueError(
            f"PDF не содержит извлекаемого текста: {pdf_path}"
        )
    
    print((f"Загружено страниц с текстом: {len(raw_docs)}"))

    text_splitter = RecursiveCharacterTextSplitter.from_tiktoken_encoder(
        chunk_size=600,
        chunk_overlap=100,
        separators=["\n\n", "\n", ".", " ", ""],
        encoding_name="cl100k_base"
    )

    chunks = text_splitter.split_documents(raw_docs)

    print(f"Создано чанков: {len(chunks)}")

    print("Загрузка модели эмбеддингов")
    model_kwargs = {'device': 'cpu'}
    encode_kwargs = {'normalize_embeddings': True}

    embeddings = HuggingFaceEmbeddings(
        model_name=EMBEDDING_MODEL,
        model_kwargs=model_kwargs,
        encode_kwargs=encode_kwargs
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

