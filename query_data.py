import os
from dotenv import load_dotenv
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_gigachat.chat_models import GigaChat
from langchain_core.prompts import PromptTemplate

# 1. Загружаем секретный ключ из файла .env
load_dotenv()

def init_rag_system():
    db_dir = "vector_db"

    print("Подключение к базе данных")
    embeddings = HuggingFaceEmbeddings(
        model_name="cointegrated/rubert-tiny2",
        model_kwargs={"device": "cpu"},
        encode_kwargs={'normalize_embeddings': True}
    )
    import warnings
    warnings.filterwarnings("ignore", category=DeprecationWarning)

    vector_db = Chroma(persist_directory=db_dir, embedding_function=embeddings)

    print("Подключение к GigaChat")
    llm = GigaChat(
        credentials=os.environ.get("GIGACHAT_CREDENTIALS"),
        verify_ssl_certs=False,
        model="GigaChat"
    )

    prompt_template = PromptTemplate.from_template(
        "Ты умный банковский ассистент. Ответь на вопрос пользователя, опираясь ТОЛЬКО на предоставленный текст тарифа.\n\n"
        "ПРАВИЛО ЛОГИКИ: В банковских тарифах пункт 'в прочих случаях' применяется ко всем ситуациям, которые не подходят под условия бесплатного обслуживания.\n\n"
        "Контекст тарифа:\n{context}\n\n"
        "Вопрос: {question}\n\n"
        "Ответ:"
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
        context_text = "\n\n".join([doc.page_content for doc in results])

        prompt = prompt_template.format(context=context_text, question=query)
        response = llm.invoke(prompt)

        print(f"\nОтвет: {response.content}\n")
        print("-" * 50)

if __name__ == "__main__":
    main_chat()