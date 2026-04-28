import streamlit as st
import os 
from dotenv import load_dotenv
from query_data import init_rag_system

load_dotenv()

st.set_page_config(page_title="Bank AI assistant", page_icon="🏦")
st.title("🏦 Ассистент по тарифам")

# Инициализация системы (кешируем, чтобы не загружать модель при каждом клике)
@st.cache_resource

def get_rag_sources():
    return init_rag_system()

vector_db, llm, prompt_template = get_rag_sources()

# История чата в сессии streamlit
if "messages" not in st.session_state:
    st.session_state.messages = []

if st.button("Очистить историю чата"):
    st.session_state.messages = []
    st.rerun()

# Отображение истории сообщений
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# Поле ввода вопроса
if prompt := st.chat_input("Задайте вопрос по тарифу"):
    # Добавляем вопрос пользователя в чат
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)
    
    with st.chat_message("assistant"):
        message_placeholder = st.empty()
        message_placeholder.markdown("⏳ *Ищу информацию...*")

        try:
            results = vector_db.similarity_search(prompt, k=3)
            context_text = "\n\n".join([doc.page_content for doc in results])

            full_prompt = prompt_template.format(context=context_text, question=prompt)
            response = llm.invoke(full_prompt)

            final_answer = response.content
            message_placeholder.markdown(final_answer)

            # Сохраняем ответ в историю
            st.session_state.messages.append({"role": "assistant", "content": final_answer})
            
            # Показываем источники в выпадающем списке
            with st.expander("Посмотреть найденные фрагменты тарифа"):
                for i, doc in enumerate(results):
                    st.write(f"**Фрагмент {i+1} (стр. {doc.metadata.get('page')})**")
                    st.text(doc.page_content)
                    st.divider()
                    
        except Exception as e:
            st.error(f"Произошла ошибка: {e}")
