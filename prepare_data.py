import fitz
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_core.documents import Document
import os


def load_pdf_with_metadata(file_path):
    docs = []
    print(f"Читаем файл: {file_path}")

    with fitz.open(file_path) as doc:
        for page_num, page in enumerate(doc):
            text = page.get_text()
            if text.strip():
                # Создаем объект документа LangChain
                metadata = {"source": file_path, "page": page_num + 1}
                docs.append(Document(page_content=text, metadata=metadata))
    return docs

def main():
    pdf_path = r"C:\Projects\rag_project\data\tbank_tariff.pdf"
    db_dir = "vector_db"

    #1. Загрузка данных
    raw_docs = load_pdf_with_metadata(pdf_path)
    print((f"Загружено страниц: {len(raw_docs)}"))

    # 2. Разбивка на чанки (через токены)
    text_splitter = RecursiveCharacterTextSplitter.from_tiktoken_encoder(
        chunk_size=600,
        chunk_overlap=100,
        separators=["\n\n", "\n", ".", " ", ""],
        encoding_name="cl100k_base"
    )
    chunks = text_splitter.split_documents(raw_docs)
    print(f"Создано чанков: {len(chunks)}")

    # 3. Эмбеддинги
    print("Загрузка модели эмбеддингов")
    model_kwargs = {'device': 'cpu'}
    encode_kwargs = {'normalize_embeddings': True}

    embeddings = HuggingFaceEmbeddings(
        model_name="cointegrated/rubert-tiny2",
        model_kwargs=model_kwargs,
        encode_kwargs=encode_kwargs
    )

    # 4. Сохранение в векторную базу
    if os.path.exists(db_dir):
        print("Старая база найдена, обновляем")
        
    print(f"Создаем векторное хранилище в {db_dir}")
    vector_db = Chroma.from_documents(
        documents=chunks,
        embedding=embeddings,
        persist_directory=db_dir
    )
    print("Векторная база готова")

if __name__ == "__main__":
    main()

