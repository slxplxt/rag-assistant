import os
from pathlib import Path

from dotenv import load_dotenv
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_gigachat.chat_models import GigaChat
from langchain_core.prompts import PromptTemplate

from config import DB_DIR, EMBEDDING_MODEL, GIGACHAT_MODEL, GIGACHAT_CREDENTIALS

def init_vector_db() -> Chroma:
    db_dir = Path(DB_DIR)

    if not db_dir.is_dir():
        raise FileNotFoundError(
            f"Векторная база не найдена: {db_dir}. "
            "Сначала запустите prepare_data.py"
        )

    embeddings = HuggingFaceEmbeddings(
        model_name=EMBEDDING_MODEL,
        model_kwargs={"device": "cpu"},
        encode_kwargs={"normalize_embeddings": True}
    )

    vector_db = Chroma(
        persist_directory=str(db_dir),
        embedding_function=embeddings
    )

    if not vector_db.get(limit=1)["ids"]:
        raise RuntimeError(
            "Векторная коллекция пуста. "
            "Проверьте путь и выполните индексацию."
        )

    return vector_db

def init_llm() -> GigaChat:
    credentials = GIGACHAT_CREDENTIALS

    if not credentials:
        raise ValueError(
            "GIGACHAT_CREDENTIALS не задан. "
            "Добавьте credentials в .env"
        )

    return GigaChat(
        credentials=credentials,
        verify_ssl_certs=False,
        model=GIGACHAT_MODEL,
        timeout=120
    )

def init_rag_system():
    vector_db = init_vector_db()
    llm = init_llm()

    prompt_template = PromptTemplate.from_template(
        """
        Ты банковский ассистент.

        Отвечай только на основе предоставленного контекста.
        Не придумывай тарифы, комиссии и условия.

        Если в контексте недостаточно данных для ответа,
        скажи: "В предоставленном тарифе недостаточно информации".

        Контекст:
        {context}

        Вопрос:
        {question}

        Ответ:
        """
    )

    return vector_db, llm, prompt_template

def main_chat():
    vector_db, llm, prompt_template = init_rag_system()

    print("\n" + "="*50)
    print("✅ Ассистент готов! Задавайте вопросы по тарифу.")
    print("Для выхода напишите 'выход', 'exit' или 'q'.")
    print("="*50 + "\n")

    while True:
        query = input("Ты:")

        if query.lower() in ['выход', 'exit', 'q', 'й']:
            print("Ассистент: До свидания!")
            break

        if not(query.strip()):
            continue

        print("Ассистент: Ищу в документах и формулирую ответ...")

        #Поиск и генерация
        results = vector_db.similarity_search(query, k=3)

        if not results:
            print("Ассистент: Не удалось найти информацию в документах.")
            continue

        context_text = "\n\n".join(
            f"[Источник {i}, страница {doc.metadata.get('page', '?')}]\n"
            f"{doc.page_content}"
            for i, doc in enumerate(results, start=1)
        )

        prompt = prompt_template.format(context=context_text, question=query)
        response = llm.invoke(prompt)

        print(f"\nОтвет: {response.content}\n")
        print("-" * 50)

if __name__ == "__main__":
    main_chat()