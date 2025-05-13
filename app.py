import streamlit as st
import weaviate
import os
import requests
import json
import sys
import logging
import traceback
from dotenv import load_dotenv
import pandas as pd
from functools import lru_cache
import re
from nltk.stem import SnowballStemmer
import importlib.util
from bs4 import BeautifulSoup
import tempfile
import shutil  # Для удаления временных файлов
from weaviate.classes.config import Property, DataType # Добавлено для схем Weaviate v4
from weaviate.exceptions import UnexpectedStatusCodeError # Добавлено для обработки ошибок Weaviate

# Загрузка переменных окружения из .env файла или из Streamlit secrets
load_dotenv()

# --- Функции для работы с переменными окружения ---
def get_env_vars():
    """Получение переменных окружения и вывод их значений в лог"""
    # Настройка логирования
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
    logger = logging.getLogger(__name__)
    
    # Проверяем, доступны ли секреты Streamlit
    if hasattr(st, 'secrets'):
        logger.info("Загрузка переменных из Streamlit secrets")
        env_vars = {
            # Weaviate настройки
            "WEAVIATE_HOST": st.secrets.get("weaviate", {}).get("host") or os.getenv("WEAVIATE_HOST"),
            "WEAVIATE_PORT": st.secrets.get("weaviate", {}).get("port") or os.getenv("WEAVIATE_PORT"),
            "WEAVIATE_GRPC_PORT": st.secrets.get("weaviate", {}).get("grpc_port") or os.getenv("WEAVIATE_GRPC_PORT"),
            
            # API ключи
            "OPENROUTER_API_KEY": st.secrets.get("api", {}).get("openrouter_key") or os.getenv("OPENROUTER_API_KEY"),
            "OPENROUTER_MODEL": st.secrets.get("api", {}).get("openrouter_model") or os.getenv("OPENROUTER_MODEL"),
            "OPENROUTER_API_URL": st.secrets.get("api", {}).get("openrouter_url") or os.getenv("OPENROUTER_API_URL"),
            "HUGGINGFACE_TOKEN": st.secrets.get("api", {}).get("huggingface_token") or os.getenv("HUGGINGFACE_TOKEN"),
            
            # Google PSE
            "GOOGLE_PSE_API_KEY": st.secrets.get("google", {}).get("pse_api_key") or os.getenv("GOOGLE_PSE_API_KEY"),
            "GOOGLE_PSE_ID": st.secrets.get("google", {}).get("pse_id") or os.getenv("GOOGLE_PSE_ID")
        }
    else:
        logger.info("Streamlit secrets не найдены, загрузка из переменных окружения")
        env_vars = {
            "WEAVIATE_HOST": os.getenv("WEAVIATE_HOST"),
            "WEAVIATE_PORT": os.getenv("WEAVIATE_PORT"),
            "WEAVIATE_GRPC_PORT": os.getenv("WEAVIATE_GRPC_PORT"),
            "OPENROUTER_API_KEY": os.getenv("OPENROUTER_API_KEY"),
            "OPENROUTER_MODEL": os.getenv("OPENROUTER_MODEL"),
            "OPENROUTER_API_URL": os.getenv("OPENROUTER_API_URL"),
            "HUGGINGFACE_TOKEN": os.getenv("HUGGINGFACE_TOKEN"),
            "GOOGLE_PSE_API_KEY": os.getenv("GOOGLE_PSE_API_KEY"),
            "GOOGLE_PSE_ID": os.getenv("GOOGLE_PSE_ID")
        }
    
    logger.info("Текущие значения переменных окружения:")
    for key, value in env_vars.items():
        if key.endswith("_KEY") or key.endswith("_TOKEN"):
            display_value = "***" + value[-4:] if value else None
        else:
            display_value = value
        logger.info(f"{key}: {display_value}")
    
    return env_vars

env_vars = get_env_vars()

# --- Константы из .env файла ---
WEAVIATE_HOST = env_vars["WEAVIATE_HOST"]
WEAVIATE_PORT = int(env_vars["WEAVIATE_PORT"]) if env_vars["WEAVIATE_PORT"] else 8080
WEAVIATE_GRPC_PORT = int(env_vars["WEAVIATE_GRPC_PORT"]) if env_vars["WEAVIATE_GRPC_PORT"] else 50051
CLASS_NAME = "DocumentChunkV4"
EMBEDDING_MODEL_NAME = 'sentence-transformers/all-MiniLM-L6-v2'
OPENROUTER_API_KEY = env_vars["OPENROUTER_API_KEY"]
OPENROUTER_MODEL = env_vars["OPENROUTER_MODEL"] or "anthropic/claude-3-haiku"  # Значение по умолчанию
OPENROUTER_API_URL = env_vars["OPENROUTER_API_URL"] or "https://openrouter.ai/api/v1"
HF_TOKEN = env_vars["HUGGINGFACE_TOKEN"]
GOOGLE_PSE_API_KEY = env_vars["GOOGLE_PSE_API_KEY"]
GOOGLE_PSE_ID = env_vars["GOOGLE_PSE_ID"]

# Настройка страницы Streamlit
st.set_page_config(
    page_title="OpenSource RAG",
    page_icon="🔍",
    layout="wide"
)

# Импортируем SentenceTransformer только при необходимости,
# чтобы избежать конфликтов с watchdog Streamlit
sentence_transformer_loaded = False

# --- Mock User Store (для демонстрации) ---
MOCK_USERS = {
    "user1": "pass1",
    "admin": "adminpass"
}

# --- Session State Initialization ---
if 'logged_in' not in st.session_state:
    st.session_state.logged_in = False
if 'username' not in st.session_state:
    st.session_state.username = None
if 'user_collection_name' not in st.session_state:
    st.session_state.user_collection_name = None

# Настройка логирования
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# --- Подключение к Weaviate ---
@st.cache_resource
def connect_to_weaviate():
    logger.info(f"Подключение к Weaviate: {WEAVIATE_HOST}:{WEAVIATE_PORT} (gRPC: {WEAVIATE_GRPC_PORT})")
    try:
        client = weaviate.connect_to_local(
            host=WEAVIATE_HOST,
            port=WEAVIATE_PORT,
            grpc_port=WEAVIATE_GRPC_PORT
        )
        
        if not client.is_ready():
            st.error("Weaviate не готов к работе. Проверьте, запущен ли сервер Weaviate.")
            return None
            
        logger.info("Успешно подключено к Weaviate и клиент готов.")
        return client
    except Exception as e:
        logger.error(f"Ошибка подключения к Weaviate: {e}")
        logger.error(traceback.format_exc())
        st.error(f"Ошибка подключения к Weaviate: {e}")
        return None

# --- Функция для создания коллекции пользователя, если она не существует ---
def create_user_collection_if_not_exists(client, collection_name):
    """Создает коллекцию Weaviate с определенной схемой, если она еще не существует."""
    try:
        if not client.collections.exists(collection_name):
            logger.info(f"Коллекция '{collection_name}' не найдена. Создание новой коллекции...")
            client.collections.create(
                name=collection_name,
                properties=[
                    Property(name="text", data_type=DataType.TEXT),
                    Property(name="filename", data_type=DataType.TEXT),
                    Property(name="chunk_index", data_type=DataType.INT),
                    Property(name="page_number", data_type=DataType.INT),
                    Property(name="element_types", data_type=DataType.TEXT) # Храним как JSON строку
                ],
                # Опционально: настройте векторизатор, если он не глобальный
                # vectorizer_config=weaviate.classes.config.Configure.Vectorizer.text2vec_transformers()
            )
            logger.info(f"Коллекция '{collection_name}' успешно создана.")
        else:
            logger.info(f"Коллекция '{collection_name}' уже существует.")
    except UnexpectedStatusCodeError as e:
        if e.status_code == 422: # Unprocessable Entity - часто означает, что коллекция уже существует с другой конфигурацией
            logger.warning(f"Коллекция '{collection_name}' уже существует или возник конфликт конфигурации: {e}")
            # Можно добавить более детальную проверку схемы, если необходимо
        else:
            logger.error(f"Ошибка при создании/проверке коллекции '{collection_name}': {e}")
            st.error(f"Ошибка Weaviate при работе с коллекцией '{collection_name}': {e.message}")
    except Exception as e:
        logger.error(f"Непредвиденная ошибка при создании/проверке коллекции '{collection_name}': {e}")
        logger.error(traceback.format_exc())
        st.error(f"Непредвиденная ошибка при работе с коллекцией '{collection_name}'.")

# --- Authentication Functions ---
def login(username, password):
    if username in MOCK_USERS and MOCK_USERS[username] == password:
        st.session_state.logged_in = True
        st.session_state.username = username
        st.session_state.user_collection_name = f"Collection_{username.replace(' ', '_')}" # Генерируем имя коллекции
        # Убедимся, что коллекция пользователя существует или будет создана при первом использовании
        client = connect_to_weaviate() # Получаем клиент
        if client:
            create_user_collection_if_not_exists(client, st.session_state.user_collection_name)
        else:
            st.error("Не удалось подключиться к Weaviate для создания/проверки коллекции пользователя.")
        st.success(f"Добро пожаловать, {username}!")
        st.rerun() # Перезагружаем страницу, чтобы обновить UI
        return True
    else:
        st.error("Неверное имя пользователя или пароль")
        return False

def logout():
    st.session_state.logged_in = False
    st.session_state.username = None
    st.session_state.user_collection_name = None
    st.info("Вы вышли из системы.")
    st.rerun() # Перезагружаем страницу, чтобы обновить UI

# --- User-specific Collection Name ---
def get_user_collection_name():
    if st.session_state.logged_in and st.session_state.user_collection_name:
        return st.session_state.user_collection_name
    # Если пользователь не залогинен, не возвращаем коллекцию
    return None

# --- UI for Login/Logout ---
if not st.session_state.logged_in:
    with st.sidebar.form("login_form"):
        st.sidebar.title("Вход")
        username_input = st.text_input("Имя пользователя", key="login_username")
        password_input = st.text_input("Пароль", type="password", key="login_password")
        login_button = st.form_submit_button("Войти")
        if login_button:
            login(username_input, password_input)
else:
    st.sidebar.success(f"Вы вошли как: {st.session_state.username}")
    if st.sidebar.button("Выйти"):
        logout()

# --- Инициализация модели для эмбеддингов ---
@st.cache_resource
def load_embedder():
    logger.info(f"Загрузка модели эмбеддингов: {EMBEDDING_MODEL_NAME}...")
    try:
        # Импортируем SentenceTransformer здесь, чтобы избежать
        # конфликтов с Streamlit
        from sentence_transformers import SentenceTransformer
        global sentence_transformer_loaded
        sentence_transformer_loaded = True
        
        embedder = SentenceTransformer(EMBEDDING_MODEL_NAME, use_auth_token=HF_TOKEN)
        logger.info("Модель эмбеддингов успешно загружена.")
        return embedder
    except Exception as e:
        logger.error(f"Не удалось загрузить модель SentenceTransformer: {e}")
        logger.error(traceback.format_exc())
        st.error(f"Ошибка загрузки модели эмбеддингов: {e}")
        return None

# --- Препроцессинг текста: очистка, лемматизация/стемминг ---
stemmer = SnowballStemmer("russian")
def preprocess_text(text: str) -> str:
    text = text.lower()
    text = re.sub(r'[^а-яА-ЯёЁ\s]', '', text)  # Удаление пунктуации
    words = text.split()
    return ' '.join([stemmer.stem(word) for word in words])

# --- Функция поиска ---
def search_chunks(query, client, embedder, n_results=5, alpha=0.5):
    try:
        # Создаем эмбеддинг для запроса
        query_embedding = get_embedding(query, embedder)
        
        collection_name = get_user_collection_name() # Получаем имя коллекции пользователя
        if not collection_name:
            st.error("Вы должны войти в систему для поиска по своей коллекции.")
            return {
                "documents": [],
                "metadatas": [],
                "distances": [],
                "sources": []
            }
        logger.info(f"Попытка поиска в коллекции: {collection_name}")

        if not client.collections.exists(collection_name):
            st.error(f"Коллекция '{collection_name}' не найдена. Загрузите документы для создания коллекции.")
            return {
                "documents": [],
                "metadatas": [],
                "distances": [],
                "sources": []
            }
        
        collection = client.collections.get(collection_name)
        logger.info(f"Поиск будет выполнен в коллекции: {collection.name}")
        
        if alpha == 1.0:
            # Чисто векторный поиск
            logger.info("Выполняем чисто векторный поиск")
            result = collection.query.near_vector(
                near_vector=query_embedding,
                limit=n_results,
                return_properties=["text", "filename", "chunk_index", "page_number", "element_types"],
                return_metadata=["distance"]
            )
        elif alpha == 0.0:
            # Чисто BM25 поиск
            logger.info("Выполняем чисто BM25 поиск")
            result = collection.query.bm25(
                query=query,
                limit=n_results,
                return_properties=["text", "filename", "chunk_index", "page_number", "element_types"]
            )
        else:
            # Гибридный поиск
            logger.info(f"Выполняем гибридный поиск с alpha={alpha}")
            result = collection.query.hybrid(
                query=query,  # Текстовый запрос для BM25
                vector=query_embedding,  # Векторный запрос
                limit=n_results,  # Ограничение по количеству результатов
                alpha=alpha,  # Балансировка между BM25 (ближе к 0) и векторным поиском (ближе к 1)
                return_properties=["text", "filename", "chunk_index", "page_number", "element_types"],
                return_metadata=["distance"]
            )
        
        # Форматируем результаты в удобную структуру
        formatted_results = {
            "documents": [],  # Тексты документов
            "metadatas": [],  # Метаданные
            "distances": [],   # Расстояния (для сортировки/фильтрации)
            "sources": []     # Добавляем список источников
        }
        
        # Проверяем, есть ли результаты
        if result.objects:
            logger.info(f"Найдено {len(result.objects)} результатов")
            for obj in result.objects:
                formatted_results["documents"].append(obj.properties.get("text", ""))
                
                # Получаем типы элементов, убедимся что это всегда список
                element_types_str = obj.properties.get("element_types", "[]")
                try:
                    element_types_list = json.loads(element_types_str) if isinstance(element_types_str, str) else element_types_str
                except (json.JSONDecodeError, TypeError):
                    element_types_list = ["parsing_error"]

                # Получаем расстояние из метаданных
                distance = None
                if hasattr(obj, 'metadata') and obj.metadata is not None:
                    if hasattr(obj.metadata, 'distance'):
                        distance = obj.metadata.distance
                    elif hasattr(obj.metadata, 'certainty'):
                        distance = 1.0 - obj.metadata.certainty

                logger.info(f"UUID: {obj.uuid}, Distance: {distance}, Type: {type(distance)}")
                
                metadata = {
                    "filename": obj.properties.get("filename", "N/A"),
                    "chunk_index": obj.properties.get("chunk_index", -1),
                    "page_number": obj.properties.get("page_number", 0),
                    "element_types": element_types_list,
                    "uuid": str(obj.uuid) # UUID для отладки
                }
                
                formatted_results["metadatas"].append(metadata)
                formatted_results["distances"].append(distance)
                
                # Добавляем информацию об источнике
                formatted_results["sources"].append({
                    "title": f"Документ: {obj.properties.get('filename', 'N/A')}",
                    "link": f"Чанк {obj.properties.get('chunk_index', -1)}, страница {obj.properties.get('page_number', 0)}",
                    "type": "document"
                })
                
        return formatted_results
    except Exception as e:
        logger.error(f"Ошибка во время поиска: {e}")
        logger.error(traceback.format_exc())
        return None

# --- Функция для поиска в интернете
def search_web(query, num_results=3, follow_links=True, max_depth=1, use_llm_summary=True):
    """
    Выполняет поиск в интернете по запросу с использованием Google Programmable Search Engine.
    Также может переходить по найденным ссылкам для извлечения более полного контекста.
    
    Args:
        query (str): Поисковый запрос
        num_results (int): Количество результатов для возврата
        follow_links (bool): Если True, переходит по найденным ссылкам для извлечения содержимого
        max_depth (int): Максимальная глубина перехода по ссылкам
        use_llm_summary (bool): Если True, использует LLM для создания выжимки
        
    Returns:
        tuple: (formatted_results, sources_list) - форматированные результаты и список источников
    """
    try:
        # Проверяем наличие ключей API
        if not GOOGLE_PSE_API_KEY or not GOOGLE_PSE_ID:
            logger.warning("API ключ Google PSE или ID поисковой системы не настроен в .env файле")
            return "Для поиска в интернете необходимо настроить Google PSE в .env файле", []
            
        # Формируем URL для запроса к API Google Custom Search
        search_url = f"https://www.googleapis.com/customsearch/v1"
        params = {
            "key": GOOGLE_PSE_API_KEY,
            "cx": GOOGLE_PSE_ID,
            "q": query,
            "num": min(num_results, 10),  # API ограничивает до 10 результатов
            "hl": "ru",   # Язык результатов (русский)
            "lr": "lang_ru"  # Ограничение по языку (русский)
        }
        
        response = requests.get(search_url, params=params)
        
        if response.status_code != 200:
            logger.error(f"Ошибка поиска в Google PSE: {response.status_code}")
            return f"Не удалось выполнить поиск в интернете. Код ошибки: {response.status_code}", []
        
        search_data = response.json()
        search_results = []
        sources_list = []
        
        # Проверяем наличие результатов
        if "items" not in search_data or not search_data["items"]:
            return "Поиск в интернете не дал результатов.", []
        
        # Обрабатываем результаты поиска
        for item in search_data["items"]:
            title = item.get("title", "Без заголовка")
            link = item.get("link", "#")
            snippet = item.get("snippet", "Нет описания")
            
            search_result = {
                "title": title,
                "link": link,
                "snippet": snippet,
                "content": "",  # Для полного текста страницы, если follow_links=True
                "summary": ""   # Для выжимки из текста
            }
            
            # Добавляем источник в отдельный список
            sources_list.append({
                "title": title,
                "link": link,
                "type": "web"
            })
            
            # Если нужно, переходим по ссылке и извлекаем содержимое
            if follow_links and link != "#":
                try:
                    logger.info(f"Извлечение содержимого страницы: {link}")
                    
                    # Заголовки для имитации браузера
                    headers = {
                        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36",
                        "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
                        "Accept-Language": "ru,en-US;q=0.9,en;q=0.8",
                        "Accept-Encoding": "gzip, deflate, br",
                        "Connection": "keep-alive",
                        "Upgrade-Insecure-Requests": "1",
                        "Cache-Control": "max-age=0"
                    }
                    
                    # Запрашиваем страницу
                    page_response = requests.get(link, headers=headers, timeout=10)
                    
                    if page_response.status_code == 200:
                        # Определяем кодировку страницы
                        page_response.encoding = page_response.apparent_encoding
                        
                        # Извлекаем текст с помощью BeautifulSoup
                        soup = BeautifulSoup(page_response.text, 'html.parser')
                        
                        # Удаляем ненужные элементы
                        for tag in soup(["script", "style", "meta", "head", "footer", "nav", "iframe", "noscript"]):
                            tag.extract()
                        
                        # Получаем основной текст страницы
                        page_text = ""
                        
                        # Ищем основной контент - приоритет отдаем article, main, content
                        main_content = soup.find(["article", "main", "div"], class_=lambda x: x and ("content" in x.lower() or "article" in x.lower()))
                        
                        if main_content:
                            page_text = main_content.get_text(separator="\n", strip=True)
                        else:
                            # Если не найден основной блок, берем текст всей страницы
                            page_text = soup.get_text(separator="\n", strip=True)
                            
                        # Очищаем текст от лишних пробелов и переносов строк
                        page_text = re.sub(r'\n+', '\n', page_text)
                        page_text = re.sub(r'\s+', ' ', page_text)
                        
                        # Сохраняем оригинальный текст
                        search_result["content"] = page_text
                        
                        # Создаем выжимку
                        if use_llm_summary and OPENROUTER_API_KEY:
                            logger.info(f"Создание выжимки из текста страницы: {link}")
                            # Очищаем текст от ссылок и сносок перед созданием выжимки
                            cleaned_text = clean_text_for_summary(page_text)
                            summary = summarize_with_llm(cleaned_text, query)
                            search_result["summary"] = summary
                        
                        logger.info(f"Успешно извлечено содержимое страницы: {link} ({len(page_text)} символов)")
                    else:
                        logger.warning(f"Не удалось получить содержимое страницы {link}. Код ответа: {page_response.status_code}")
                except Exception as e:
                    logger.warning(f"Ошибка при извлечении содержимого страницы {link}: {str(e)}")
                    
            search_results.append(search_result)
            
        # Форматируем результаты для контекста
        formatted_results = "\n\nРезультаты поиска в интернете:\n\n"
        for idx, result in enumerate(search_results, 1):
            formatted_results += f"{idx}. {result['title']}\n{result['snippet']}\n{result['link']}\n\n"
            
            # Если есть выжимка, добавляем её
            if result["summary"]:
                formatted_results += f"Содержимое страницы {idx} (выжимка):\n{result['summary']}\n\n"
            # Иначе, если есть контент, добавляем его частично
            elif result["content"]:
                formatted_results += f"Содержимое страницы {idx}:\n{result['content'][:1000]}...\n\n"
        
        return formatted_results, sources_list
        
    except Exception as e:
        logger.error(f"Ошибка при поиске в интернете: {e}")
        logger.error(traceback.format_exc())
        return "Произошла ошибка при поиске в интернете.", []

# --- Функция для запроса к LLM через OpenRouter ---
def query_llm(query, context, model=None, use_model_knowledge=False, use_web_search=False, document_sources=None, follow_links=False, use_llm_summary=True):
    # Проверяем/обновляем значение OPENROUTER_MODEL перед каждым запросом
    global OPENROUTER_MODEL
    if os.getenv("OPENROUTER_MODEL") and os.getenv("OPENROUTER_MODEL") != OPENROUTER_MODEL:
        OPENROUTER_MODEL = os.getenv("OPENROUTER_MODEL")
        logger.info(f"Обновлена модель OPENROUTER_MODEL: {OPENROUTER_MODEL}")
        
    if not OPENROUTER_API_KEY:
        st.error("API ключ OpenRouter не настроен в .env файле")
        return "Для использования LLM необходимо настроить API ключ в .env файле", []
    
    # Используем модель из параметров или из переменной окружения
    current_model = model if model is not None else OPENROUTER_MODEL
    logger.info(f"Используется модель для запроса: {current_model}")
    
    try:
        # Подготовим список всех источников
        all_sources = []
        if document_sources:
            all_sources.extend(document_sources)
        
        # Если включен поиск в интернете, выполняем его и добавляем к контексту
        web_context = ""
        web_sources = []
        if use_web_search:
            logger.info(f"Выполнение поиска в интернете для запроса: {query}")
            web_search_results, web_sources = search_web(query, follow_links=follow_links, use_llm_summary=use_llm_summary)
            web_context = web_search_results
            all_sources.extend(web_sources)
            logger.info("Поиск в интернете выполнен успешно")
        
        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {OPENROUTER_API_KEY}",
            "HTTP-Referer": "https://github.com/yourusername/opensourcerag", # Замените на ваш адрес
            "X-Title": "OpenSource RAG"
        }
        
        # Выбираем системную инструкцию в зависимости от режима
        if use_model_knowledge:
            system_message = """Ты — полезный ассистент, который отвечает на вопросы о пенсиях, пенсионном обеспечении и социальной защите.
            
Основывайся в первую очередь на предоставленном контексте из документов. Если информации в контексте недостаточно,
ты можешь дополнить ответ из своих знаний или предоставленных результатов поиска в интернете. 
Явно указывай, когда ты дополняешь информацию из контекста своими знаниями или данными из интернета.

Когда цитируешь факты, НЕ используй формат 'Источник[N]' в тексте ответа. Источники будут отображены отдельно под твоим ответом.

Отвечай детально и с пониманием темы, учитывая, что пользователь может не разбираться в пенсионном законодательстве.
Если тебе не хватает информации, спокойно признай это и предложи уточнить вопрос."""
        else:
            system_message = """Ты — полезный ассистент, который отвечает на вопросы на основе предоставленного контекста.
Используй только информацию из контекста и результатов поиска в интернете (если они предоставлены).
Если в контексте и результатах поиска нет ответа на вопрос, скажи, что не можешь ответить.

Когда цитируешь факты, НЕ используй формат 'Источник[N]' в тексте ответа. Источники будут отображены отдельно под твоим ответом."""
        
        # Формируем полный контекст, включая результаты поиска в интернете, если они есть
        full_context = context
        if web_context:
            full_context += web_context
        
        data = {
            "model": current_model,
            "messages": [
                {"role": "system", "content": system_message},
                {"role": "user", "content": f"Контекст: {full_context}\n\nВопрос: {query}"}
            ]
        }
        
        response = requests.post(f"{OPENROUTER_API_URL}/chat/completions", headers=headers, json=data)
        
        if response.status_code == 200:
            result = response.json()
            return result['choices'][0]['message']['content'], all_sources
        else:
            logger.error(f"Ошибка запроса к LLM API: {response.status_code}")
            logger.error(response.text)
            return f"Ошибка запроса к LLM API: {response.status_code}", []
    except Exception as e:
        logger.error(f"Ошибка при запросе к LLM: {e}")
        return f"Произошла ошибка: {str(e)}", []

# --- Получение нормализованного эмбеддинга ---
@lru_cache(maxsize=1024)
def get_embedding(text: str, embedder) -> list[float]:
    cleaned_text = preprocess_text(text)
    embedding = embedder.encode(cleaned_text, normalize_embeddings=True)
    return embedding.tolist()

# --- Импорт функции process_file из main_fixed.py ---
spec = importlib.util.spec_from_file_location("main_fixed", os.path.join(os.path.dirname(__file__), "main_fixed.py"))
main_fixed = importlib.util.module_from_spec(spec)
spec.loader.exec_module(main_fixed)

# --- Функция для создания выжимки из текста страницы с помощью LLM ---
def summarize_with_llm(text, query, max_length=1500):
    """
    Создает информативную выжимку из текста страницы с помощью LLM.
    
    Args:
        text (str): Исходный текст страницы
        query (str): Исходный запрос пользователя для контекста
        max_length (int): Максимальная желаемая длина выжимки
        
    Returns:
        str: Информативная выжимка из текста
    """
    # Если текст уже короче максимальной длины, возвращаем как есть
    if len(text) <= max_length:
        return text
        
    try:
        if not OPENROUTER_API_KEY:
            logger.warning("API ключ OpenRouter не настроен. Используем алгоритмическую выжимку.")
            # Если нет API ключа, используем простой алгоритм обрезки текста
            return text[:max_length] + "..."
        
        logger.info(f"Создание выжимки с помощью LLM для текста длиной {len(text)} символов")
        
        # Ограничиваем размер текста для запроса к LLM (чтобы избежать превышения контекста)
        if len(text) > 10000:
            # Берем начало, середину и конец текста
            first_part = text[:5000]
            mid_index = len(text) // 2
            middle_part = text[mid_index-1000:mid_index+1000]
            last_part = text[-3000:]
            truncated_text = f"{first_part}\n\n[...сокращено...]\n\n{middle_part}\n\n[...сокращено...]\n\n{last_part}"
        else:
            truncated_text = text
            
        # Дополнительная очистка текста от ссылок и сносок (на случай, если не была выполнена ранее)
        truncated_text = clean_text_for_summary(truncated_text)
        
        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {OPENROUTER_API_KEY}",
            "HTTP-Referer": "https://github.com/yourusername/opensourcerag",
            "X-Title": "OpenSource RAG Summary"
        }
        
        system_prompt = """Ты — эксперт по созданию кратких и информативных выжимок из текста. 
Твоя задача — создать информативное резюме на основе предоставленного текста веб-страницы.
Выжимка должна:
1. Сохранять ключевую информацию, факты, определения и важные детали
2. Быть релевантной запросу пользователя
3. Быть краткой, но содержательной (до 1500 символов)
4. Не содержать субъективных оценок или интерпретаций
5. Сохранять важные данные, даты, цифры и названия
6. Игнорировать ссылки на источники вида [1], [2] и т.д.
7. Не включать ненужные технические элементы (сноски, метаданные)

Пожалуйста, создай только выжимку, без введений и метатекста."""

        user_prompt = f"""Запрос пользователя: {query}

Текст веб-страницы для резюмирования:
{truncated_text}

Создай информативную выжимку из этого текста, сохраняя наиболее важную информацию, которая может быть релевантна запросу."""
        
        data = {
            "model": OPENROUTER_MODEL,
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt}
            ]
        }
        
        response = requests.post(f"{OPENROUTER_API_URL}/chat/completions", headers=headers, json=data, timeout=30)
        
        if response.status_code == 200:
            result = response.json()
            summary = result['choices'][0]['message']['content']
            logger.info(f"Создана выжимка длиной {len(summary)} символов")
            return summary
        else:
            logger.error(f"Ошибка при обращении к LLM API: {response.status_code}")
            logger.error(response.text)
            # Если не удалось получить выжимку через LLM, возвращаем часть исходного текста
            return text[:max_length] + "..."
    
    except Exception as e:
        logger.error(f"Ошибка при создании выжимки с помощью LLM: {e}")
        logger.error(traceback.format_exc())
        # В случае ошибки возвращаем часть исходного текста
        return text[:max_length] + "..."

# --- Функция для очистки текста от ссылок и сносок ---
def clean_text_for_summary(text):
    """
    Очищает текст от ссылок, сносок и других технических элементов.
    
    Args:
        text (str): Исходный текст
        
    Returns:
        str: Очищенный текст
    """
    # Удаляем ссылки вида [1], [2], [3] и т.д.
    cleaned_text = re.sub(r'\[\d+\]', '', text)
    
    # Удаляем ссылки вида [a], [b], [c] и т.д.
    cleaned_text = re.sub(r'\[[a-zA-Z]\]', '', cleaned_text)
    
    # Удаляем мусорные символы, которые часто встречаются при парсинге веб-страниц
    cleaned_text = re.sub(r'[\xa0\u200b\u200c\u200d]', ' ', cleaned_text)
    
    # Удаляем повторяющиеся пробелы
    cleaned_text = re.sub(r'\s+', ' ', cleaned_text)
    
    # Удаляем повторяющиеся переносы строк
    cleaned_text = re.sub(r'\n+', '\n', cleaned_text)
    
    return cleaned_text.strip()

# --- Функция для обновления переменных окружения ---
def update_env_vars():
    """Перезагружает переменные окружения из .env файла"""
    # Перезагрузка .env файла
    load_dotenv(override=True)
    
    # Получение обновленных переменных
    new_env_vars = get_env_vars()
    
    # Обновление глобальных переменных
    global WEAVIATE_HOST, WEAVIATE_PORT, WEAVIATE_GRPC_PORT, OPENROUTER_API_KEY
    global OPENROUTER_MODEL, OPENROUTER_API_URL, HF_TOKEN, GOOGLE_PSE_API_KEY, GOOGLE_PSE_ID
    
    WEAVIATE_HOST = new_env_vars["WEAVIATE_HOST"]
    WEAVIATE_PORT = int(new_env_vars["WEAVIATE_PORT"]) if new_env_vars["WEAVIATE_PORT"] else 8080
    WEAVIATE_GRPC_PORT = int(new_env_vars["WEAVIATE_GRPC_PORT"]) if new_env_vars["WEAVIATE_GRPC_PORT"] else 50051
    OPENROUTER_API_KEY = new_env_vars["OPENROUTER_API_KEY"]
    OPENROUTER_MODEL = new_env_vars["OPENROUTER_MODEL"] or "anthropic/claude-3-haiku"  # Значение по умолчанию
    OPENROUTER_API_URL = new_env_vars["OPENROUTER_API_URL"] or "https://openrouter.ai/api/v1"
    HF_TOKEN = new_env_vars["HUGGINGFACE_TOKEN"]
    GOOGLE_PSE_API_KEY = new_env_vars["GOOGLE_PSE_API_KEY"]
    GOOGLE_PSE_ID = new_env_vars["GOOGLE_PSE_ID"]
    
    logger.info(f"Переменные окружения обновлены. Используемая модель LLM: {OPENROUTER_MODEL}")
    
    return {
        "WEAVIATE_HOST": WEAVIATE_HOST,
        "WEAVIATE_PORT": WEAVIATE_PORT,
        "WEAVIATE_GRPC_PORT": WEAVIATE_GRPC_PORT,
        "OPENROUTER_MODEL": OPENROUTER_MODEL,
        "OPENROUTER_API_URL": OPENROUTER_API_URL
    }

# --- Кастомные стили для всего приложения ---
st.markdown('''
    <style>
    /* Общий фон */
    .stApp { background-color: #f7fafd; }
    
    /* Сайдбар */
    section[data-testid="stSidebar"] { background: linear-gradient(180deg, #e3f2fd 0%, #f7fafd 100%); }
    
    /* Кнопки */
    .stButton>button { 
        background-color: #1976d2; 
        color: white; 
        border-radius: 6px; 
        border: none;
        padding: 0.5rem 1rem;
        transition: all 0.3s;
    }
    .stButton>button:hover { 
        background-color: #1565c0; 
        box-shadow: 0 2px 5px rgba(0,0,0,0.2);
        transform: translateY(-1px);
    }
    
    /* Чат-контейнер */
    .chat-container {
        display: flex;
        flex-direction: column;
        gap: 16px;
        margin-bottom: 20px;
    }
    
    /* Общие стили сообщений */
    .chat-message {
        display: flex;
        align-items: flex-start;
        padding: 0.5rem 0;
    }
    
    /* Стили для аватара */
    .chat-avatar {
        display: flex;
        justify-content: center;
        align-items: center;
        width: 36px;
        height: 36px;
        border-radius: 50%;
        margin-right: 12px;
        flex-shrink: 0;
    }
    
    .user-avatar {
        background: #1976d2;
        color: white;
    }
    
    .assistant-avatar {
        background: #43a047;
        color: white;
    }
    
    /* Пузыри сообщений */
    .message-bubble {
        padding: 12px 16px;
        border-radius: 12px;
        max-width: 85%;
        position: relative;
        line-height: 1.5;
    }
    
    .user-message .message-bubble {
        background: #e3f2fd;
        border: 1px solid #bbdefb;
        border-top-left-radius: 2px;
    }
    
    .assistant-message .message-bubble {
        background: #f1f8e9;
        border: 1px solid #dcedc8;
        border-top-right-radius: 2px;
    }
    
    /* Карточки документов */
    .doc-card { 
        border: 1px solid #90caf9; 
        border-radius: 10px; 
        padding: 15px; 
        margin-bottom: 10px; 
        background: #f5faff;
        transition: all 0.2s;
    }
    
    .doc-card:hover {
        box-shadow: 0 3px 8px rgba(0,0,0,0.1);
    }
    
    .doc-title { 
        color: #1976d2; 
        font-weight: bold; 
    }
    
    .doc-meta { 
        color: #555; 
        font-size: 0.95em; 
        margin: 4px 0;
    }
    
    .doc-content { 
        color: #333; 
        margin-top: 10px; 
        padding: 8px;
        background: rgba(255,255,255,0.5);
        border-radius: 4px;
    }
    
    /* Вкладки */
    .stTabs [data-baseweb="tab-list"] { 
        background: #e3f2fd; 
        border-radius: 8px; 
        padding: 4px;
    }
    
    .stTabs [data-baseweb="tab"] { 
        border-radius: 6px;
    }
    
    .stTabs [aria-selected="true"] {
        background: #bbdefb;
        font-weight: bold;
    }
    
    /* Анимация загрузки для чата */
    @keyframes pulse {
        0% { opacity: 0.4; }
        50% { opacity: 0.8; }
        100% { opacity: 0.4; }
    }
    
    .loading-animation {
        display: flex;
        padding: 12px 16px;
    }
    
    .loading-dot {
        width: 8px;
        height: 8px;
        border-radius: 50%;
        background-color: #90caf9;
        margin: 0 3px;
        animation: pulse 1.5s infinite;
    }
    
    .loading-dot:nth-child(2) {
        animation-delay: 0.2s;
    }
    
    .loading-dot:nth-child(3) {
        animation-delay: 0.4s;
    }
    
    /* Форматирование Markdown в сообщениях */
    .message-content p {
        margin-bottom: 8px;
    }
    
    .message-content h1, 
    .message-content h2, 
    .message-content h3 {
        margin-top: 12px;
        margin-bottom: 8px;
    }
    
    .message-content ul, 
    .message-content ol {
        padding-left: 20px;
    }
    
    /* Стили для кода */
    .message-content code {
        background: rgba(0,0,0,0.05);
        padding: 2px 4px;
        border-radius: 3px;
        font-family: monospace;
    }
    
    .message-content pre {
        background: #2d2d2d;
        color: #f9f9f9;
        padding: 12px;
        border-radius: 6px;
        overflow-x: auto;
    }
    </style>
''', unsafe_allow_html=True)

# Функция для рендеринга чат-сообщений
def render_chat_message(role, content, is_loading=False, feedback_key=None):
    if role == "user":
        avatar_icon = "👤"
        avatar_class = "user-avatar"
        message_class = "user-message"
    else:  # assistant
        avatar_icon = "🤖"
        avatar_class = "assistant-avatar"
        message_class = "assistant-message"
    
    message_html = f"""
    <div class="chat-message {message_class}">
        <div class="chat-avatar {avatar_class}">
            {avatar_icon}
        </div>
        <div class="message-bubble">
    """
    
    if is_loading:
        message_html += """
            <div class="loading-animation">
                <div class="loading-dot"></div>
                <div class="loading-dot"></div>
                <div class="loading-dot"></div>
            </div>
        """
    else:
        message_html += f'<div class="message-content">{content}</div>'
    
    message_html += "</div></div>"
    
    message_container = st.container()
    message_container.markdown(message_html, unsafe_allow_html=True)
    
    # Добавляем кнопку обратной связи для сообщений ассистента
    if role == "assistant" and feedback_key:
        feedback_col = message_container.columns([6, 1])[1]
        with feedback_col:
            st.feedback("thumbs", key=feedback_key)
    
    return message_container

# --- Streamlit интерфейс ---
def main():
    # --- Вкладки для навигации ---
    tabs = st.tabs(["💬 Чат", "📤 Загрузка", "👤 Профиль"])
    
    # Инициализация необходимых компонентов и параметров
    with st.spinner('Загрузка моделей и подключение к базе данных...'):
        embedder = load_embedder()
        client = connect_to_weaviate()
    if not embedder or not client:
        st.error("Не удалось инициализировать необходимые компоненты")
        return
        
    # Настройки поиска - определяем параметры
    alpha = 0.5  # Баланс BM25/Векторный поиск (по умолчанию 0.5)
    n_results = 5  # Количество результатов (по умолчанию 5)
    
    # Настройки LLM
    use_model_knowledge = True
    use_web_search = True if (GOOGLE_PSE_API_KEY and GOOGLE_PSE_ID) else False
    
    # Если пользователь активировал веб-поиск в предыдущей сессии, сохраняем это состояние
    if "follow_links" not in st.session_state:
        st.session_state.follow_links = True
    if "use_llm_summary" not in st.session_state: 
        st.session_state.use_llm_summary = True
            
    # --- Главный заголовок с логотипом ---
    with tabs[0]:
        col1, col2 = st.columns([1, 6])
        with col1:
            st.markdown("<h1 style='text-align: center; margin-bottom: 0px;'>🔍</h1>", unsafe_allow_html=True)
        with col2:
            st.markdown("<h1 style='margin-bottom: 0px;'>OpenSource RAG</h1>", unsafe_allow_html=True)
            st.markdown("<p style='margin-top: 0px;'>Семантический поиск и генерация ответов с Weaviate и OpenRouter</p>", unsafe_allow_html=True)

        # --- Новый чат-интерфейс ---
        st.header("💬 Чат с ассистентом")
        if "messages" not in st.session_state:
            st.session_state.messages = []
        if "last_search" not in st.session_state:
            st.session_state.last_search = None
        if "is_generating" not in st.session_state:
            st.session_state.is_generating = False

        # Создаем контейнер для всех сообщений
        st.markdown('<div class="chat-container">', unsafe_allow_html=True)
        
        # История сообщений
        messages_container = st.container()
        with messages_container:
            for message in st.session_state.messages:
                render_chat_message(
                    role=message["role"], 
                    content=message["content"],
                    feedback_key=message.get("feedback_key")
                )
                
            # Показываем индикатор загрузки при генерации
            if st.session_state.is_generating:
                render_chat_message(role="assistant", content="", is_loading=True)
        
        st.markdown('</div>', unsafe_allow_html=True)

        # Ввод нового запроса
        if prompt := st.chat_input("Введите ваш вопрос или запрос для поиска...", key="main_chat_input", disabled=st.session_state.is_generating):
            # Добавляем сообщение пользователя
            st.session_state.messages.append({"role": "user", "content": prompt})
            render_chat_message(role="user", content=prompt)
            
            # Устанавливаем флаг генерации
            st.session_state.is_generating = True
            st.rerun()  # Перезапускаем страницу, чтобы показать индикатор загрузки
            
        # Если в процессе генерации, выполняем генерацию ответа
        if st.session_state.is_generating:
            # Получаем последний запрос пользователя
            prompt = st.session_state.messages[-1]["content"]
            
            # Поиск релевантных документов
            search_results = search_chunks(prompt, client, embedder, n_results=n_results, alpha=alpha)
            
            # Проверяем, нужно ли использовать поиск в интернете, если нет релевантных документов
            if (not search_results or not search_results['documents']) and use_web_search:
                web_context = "В документах не найдено релевантной информации. Выполняется поиск в интернете..."
                follow_links_enabled = "follow_links" in st.session_state and st.session_state.follow_links
                use_llm_summary_enabled = "use_llm_summary" in st.session_state and st.session_state.use_llm_summary
                
                answer, sources = query_llm(prompt, "", use_model_knowledge=use_model_knowledge, 
                                          use_web_search=True, follow_links=follow_links_enabled,
                                          use_llm_summary=use_llm_summary_enabled)
                
                # Форматируем источники
                sources_md = ""
                if sources:
                    web_sources = [source for source in sources if source["type"] == "web"]
                    if web_sources:
                        sources_md = "\n\n### Источники:\n"
                        for i, source in enumerate(web_sources, 1):
                            sources_md += f"[{i}. {source['title']}]({source['link']})\n"
                
                # Объединяем ответ и источники
                full_answer = answer + sources_md
                
                # Сохраняем ответ с источниками
                feedback_key = f"feedback_{len(st.session_state.messages)}"
                st.session_state.messages.append({
                    "role": "assistant", 
                    "content": full_answer, 
                    "feedback_key": feedback_key, 
                    "sources": sources
                })
                
            elif not search_results or not search_results['documents']:
                # Нет документов и нет веб-поиска
                response = "По вашему запросу не найдены релевантные документы."
                st.session_state.messages.append({"role": "assistant", "content": response})
                
            else:
                # Есть документы, генерируем ответ
                context = "\n\n".join(search_results['documents'])
                document_sources = search_results.get("sources", [])
                
                follow_links_enabled = "follow_links" in st.session_state and st.session_state.follow_links
                use_llm_summary_enabled = "use_llm_summary" in st.session_state and st.session_state.use_llm_summary
                
                answer, sources = query_llm(prompt, context, use_model_knowledge=use_model_knowledge, 
                                          use_web_search=use_web_search, document_sources=document_sources,
                                          follow_links=follow_links_enabled, use_llm_summary=use_llm_summary_enabled)
                
                # Форматируем источники
                sources_md = ""
                if sources:
                    web_sources = [source for source in sources if source["type"] == "web"]
                    if web_sources:
                        sources_md = "\n\n### Источники:\n"
                        for i, source in enumerate(web_sources, 1):
                            sources_md += f"[{i}. {source['title']}]({source['link']})\n"
                
                # Объединяем ответ и источники
                full_answer = answer + sources_md
                
                # Сохраняем ответ с источниками
                feedback_key = f"feedback_{len(st.session_state.messages)}"
                st.session_state.messages.append({
                    "role": "assistant", 
                    "content": full_answer, 
                    "feedback_key": feedback_key, 
                    "sources": sources
                })
                
                # Сохраняем результаты поиска
                st.session_state.last_search = search_results
            
            # Снимаем флаг генерации
            st.session_state.is_generating = False
            st.rerun()  # Перезапускаем страницу, чтобы показать новое сообщение
        
        # Красивые карточки найденных документов
        if st.session_state.get("last_search") and st.session_state.last_search['documents']:
            with st.expander("📚 Найденные документы (последний запрос)", expanded=False):
                for i, (doc, meta, dist) in enumerate(zip(st.session_state.last_search['documents'], 
                                                      st.session_state.last_search['metadatas'], 
                                                      st.session_state.last_search['distances'])):
                    if dist is not None:
                        try:
                            relevance = f"{1-float(dist):.2f}" if isinstance(dist, (float, int)) or (isinstance(dist, str) and dist.replace('.', '', 1).isdigit()) else "N/A"
                        except (ValueError, TypeError):
                            relevance = "N/A"
                    else:
                        relevance = "N/A"
                    st.markdown(f"""
                    <div class='doc-card'>
                        <div class='doc-title'>Документ {i+1}: {meta['filename']}</div>
                        <div class='doc-meta'>Релевантность: {relevance} | Чанк: {meta['chunk_index']}, Страница: {meta['page_number']} | UUID: {meta['uuid']}</div>
                        <div class='doc-meta'>Типы элементов: {', '.join(meta['element_types'])}</div>
                        <div class='doc-content'>{doc[:1500]}{'...' if len(doc) > 1500 else ''}</div>
                    </div>
                    """, unsafe_allow_html=True)
                    
        # Кнопка очистки истории чата
        if st.session_state.messages:
            if st.button("Очистить историю чата", type="primary", help="Удалить все сообщения и начать заново"):
                st.session_state.messages = []
                st.session_state.last_search = None
                st.rerun()

    # --- Вкладка загрузки ---
    with tabs[1]:
        upload_page(on_back=lambda: st.session_state.update({"page": "chat"}))

    # --- Вкладка профиля ---
    with tabs[2]:
        st.header("👤 Профиль пользователя")
        if not st.session_state.logged_in:
            st.warning("Войдите в систему, чтобы просматривать профиль и управлять коллекцией.")
        else:
            st.markdown(f"**Пользователь:** `{st.session_state.username}`")
            user_collection_name = get_user_collection_name()
            st.markdown(f"**Ваша коллекция:** `{user_collection_name}`")
            client = connect_to_weaviate()
            if client and client.collections.exists(user_collection_name):
                collection = client.collections.get(user_collection_name)
                doc_count = collection.aggregate.over_all(total_count=True).total_count
                st.info(f"В коллекции {doc_count} документов.")
                # Список документов с возможностью удаления
                st.markdown("### Ваши документы:")
                docs = collection.query.bm25(query="*", limit=20, return_properties=["text", "filename", "chunk_index", "page_number"])
                for obj in docs.objects:
                    st.markdown(f"<div class='doc-card'><b>{obj.properties.get('filename','N/A')}</b> | Чанк: {obj.properties.get('chunk_index','-')}, Стр: {obj.properties.get('page_number','-')}<br><span class='doc-content'>{obj.properties.get('text','')[:300]}{'...' if len(obj.properties.get('text',''))>300 else ''}</span></div>", unsafe_allow_html=True)
                    if st.button(f"Удалить чанк {obj.uuid}", key=f"del_{obj.uuid}", help="Удалить этот чанк из коллекции"):
                        try:
                            collection.data.delete_by_id(obj.uuid)
                            st.success(f"Чанк {obj.uuid} удалён!")
                            st.rerun()
                        except Exception as e:
                            st.error(f"Ошибка при удалении: {e}")
            else:
                st.info("Ваша коллекция пуста или не создана.")

# --- Страница загрузки файлов ---
def upload_page(on_back=None):
    st.title("📤 Загрузка и индексация документов")
    
    # Добавляем кнопку "Назад" если передана функция on_back
    if on_back:
        if st.button("← Вернуться к чату", type="secondary"):
            on_back()
            st.rerun()
    
    # Интерактивная инструкция по загрузке
    with st.expander("ℹ️ Инструкция по загрузке файлов", expanded=False):
        st.markdown("""
        ### Поддерживаемые форматы:
        - PDF (`.pdf`)
        - Microsoft Word (`.docx`)
        - Текстовые файлы (`.txt`)
        - Markdown (`.md`)
        - HTML (`.html`)
        - JSON (`.json`)
        - Email (`.eml`)
        
        ### Процесс индексации:
        1. Файлы разбиваются на семантические чанки
        2. Создаются эмбеддинги для каждого чанка
        3. Данные сохраняются в Weaviate
        
        ### Рекомендации:
        - Загружайте документы с качественным текстом
        - Для больших документов процесс может занять некоторое время
        - Индексированные документы можно использовать сразу после загрузки
        """)
    
    # Показываем информацию о состоянии Weaviate
    embedder = load_embedder()
    client = connect_to_weaviate()
    
    if client and client.is_ready():
        try:
            user_collection_name = get_user_collection_name()
            if not user_collection_name:
                st.error("Вы должны войти в систему для загрузки и просмотра своих документов.")
                return
            doc_count = 0 # Инициализируем количество документов
            if client.collections.exists(user_collection_name):
                collection = client.collections.get(user_collection_name)
                doc_count = collection.aggregate.over_all(total_count=True).total_count
                st.info(f"База данных Weaviate подключена. Активная коллекция: '{user_collection_name}'. Индексировано документов: {doc_count}", icon="✅")
            else:
                st.info(f"База данных Weaviate подключена. Ваша коллекция '{user_collection_name}' не создана или пуста.", icon="ℹ️")
        except Exception as e:
            st.warning(f"База данных Weaviate подключена, но возникла ошибка при запросе информации о коллекциях: {e}")
    else:
        st.error("Не удалось подключиться к базе данных Weaviate. Проверьте настройки и запустите сервер.")
        return
    
    # Форма загрузки файлов
    uploaded_files = st.file_uploader("Загрузите файлы для индексации", 
                                      type=["pdf", "docx", "txt", "md", "html", "json", "eml"], 
                                      accept_multiple_files=True)
    
    # Кнопка загрузки с отображением прогресса
    if uploaded_files:
        st.write(f"Выбрано файлов: {len(uploaded_files)}")
        if st.button("Начать индексацию", type="primary", use_container_width=True):
            progress_bar = st.progress(0)
            status_text = st.empty()
            
            # Создаем временную директорию
            temp_dir = tempfile.mkdtemp()
            try:
                for i, uploaded_file in enumerate(uploaded_files):
                    progress = i / len(uploaded_files)
                    progress_bar.progress(progress)
                    status_text.text(f"Обработка файла {i+1} из {len(uploaded_files)}: {uploaded_file.name}")
                    
                    with st.spinner(f"Обработка файла {uploaded_file.name}..."):
                        temp_path = os.path.join(temp_dir, uploaded_file.name)
                        with open(temp_path, "wb") as f:
                            f.write(uploaded_file.getbuffer())
                        
                        current_collection_name = get_user_collection_name()
                        logger.info(f"Загрузка файла '{uploaded_file.name}' в коллекцию '{current_collection_name}'.")
                        
                        # Убедимся, что коллекция существует перед добавлением данных
                        if client and not client.collections.exists(current_collection_name):
                            logger.info(f"Коллекция '{current_collection_name}' не существует. Создаем перед загрузкой файла.")
                            create_user_collection_if_not_exists(client, current_collection_name)
                        
                        # Проверяем еще раз, на случай если создание не удалось
                        if client and client.collections.exists(current_collection_name):
                            success = main_fixed.process_file(temp_path, uploaded_file.name, current_collection_name, client, embedder)
                        else:
                            logger.error(f"Не удалось создать или найти коллекцию '{current_collection_name}' для загрузки файла.")
                            st.error(f"Ошибка: не удалось подготовить коллекцию '{current_collection_name}' для файла {uploaded_file.name}.")
                            success = False
                        
                        if success:
                            st.success(f"Файл {uploaded_file.name} успешно обработан и загружен в Weaviate!")
                        else:
                            st.error(f"Ошибка при обработке файла {uploaded_file.name}.")
            finally:
                # Удаляем временную директорию и все файлы в ней
                shutil.rmtree(temp_dir, ignore_errors=True)
            
            progress_bar.progress(1.0)
            status_text.text("Индексация завершена!")
            
            # Добавляем кнопку для возврата к чату после индексации
            if on_back:
                if st.button("Вернуться к чату после индексации", type="primary"):
                    on_back()
                    st.rerun()

# --- Навигация и запуск приложения ---
if __name__ == "__main__":
    if sys.version_info >= (3, 10):
        import asyncio
        asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())
    main()
