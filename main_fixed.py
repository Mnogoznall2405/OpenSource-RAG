import os
import weaviate
from dotenv import load_dotenv
# Import configuration classes from Weaviate v4
from weaviate.classes.config import Configure, Property, DataType, VectorDistances
from weaviate.classes.query import MetadataQuery # For specifying metadata return
from weaviate.util import generate_uuid5
from unstructured.partition.auto import partition
from unstructured.chunking.title import chunk_by_title
from unstructured.documents.elements import Table, ListItem, Text # Keep specific imports
from sentence_transformers import SentenceTransformer
import logging
import pandas as pd
import json
import traceback # For detailed error logging
from weaviate.exceptions import WeaviateConnectionError
import time 
# from weaviate import WeaviateErrorRetryConf  # Удалить/закомментировать
# import signal  # Удалить/закомментировать

# Загружаем переменные окружения из .env файла
load_dotenv()

# Настройка логирования
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Фильтр для подавления предупреждений о CropBox
class CropBoxWarningFilter(logging.Filter):
    def filter(self, record):
        return not (record.levelno == logging.WARNING and "CropBox missing from /Page, defaulting to MediaBox" in record.getMessage())

# Применяем фильтр к корневому логгеру для подавления предупреждений о CropBox
logging.getLogger().addFilter(CropBoxWarningFilter())

# --- Константы и Конфигурация ---
FOLDER_PATH = "files"  # Путь к папке с файлами
# Weaviate Configuration
WEAVIATE_HOST = "localhost"
WEAVIATE_PORT = 8080
WEAVIATE_GRPC_PORT = 50051
CLASS_NAME = "DocumentChunkV4" # Имя класса (коллекции), можно изменить
# Model Configuration
EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL")
logger.info(f"Загрузка из переменных окружения: EMBEDDING_MODEL='{EMBEDDING_MODEL}'")

# Проверяем и корректируем значение, если оно не установлено или неверно
if not EMBEDDING_MODEL or "L6-v2" in EMBEDDING_MODEL:
    EMBEDDING_MODEL = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
    logger.info(f"Принудительно установлено EMBEDDING_MODEL='{EMBEDDING_MODEL}'")

HF_TOKEN = os.getenv("HUGGINGFACE_TOKEN")
# Chunking Configuration
CHUNK_MAX_CHARS = 3000
CHUNK_OVERLAP = 300



# --- Инициализация Weaviate Client (v4) ---
logger.info(f"Подключение к Weaviate: {WEAVIATE_HOST}:{WEAVIATE_PORT} (gRPC: {WEAVIATE_GRPC_PORT})")
try:
    client = weaviate.connect_to_local(
        host=WEAVIATE_HOST,
        port=WEAVIATE_PORT,
        grpc_port=WEAVIATE_GRPC_PORT
    )

    # Улучшенная проверка готовности с ретраями
    retries = 5
    for i in range(retries):
        if client.is_ready():
            break
        logger.warning(f"Ожидание готовности Weaviate ({i+1}/{retries})")
        time.sleep(3)
    else:
        raise WeaviateConnectionError("Не удалось подключиться к Weaviate после {retries} попыток")

    logger.info("Успешное подключение к Weaviate")

    # Используем connect_to_local, самый простой способ для локального запуска
    # Убираем ЛЮБЫЕ аргументы, связанные с таймаутом, которых нет в v4 API connect_to_local
    client = weaviate.connect_to_local(
        host=WEAVIATE_HOST,
        port=WEAVIATE_PORT,
        grpc_port=WEAVIATE_GRPC_PORT
    )

    # Проверка готовности после попытки подключения
    # Дадим клиенту немного времени на установку соединения и проверку статуса
    retries = 3
    ready = False
    for i in range(retries):
        if client.is_ready():
            ready = True
            break
        logger.warning(f"Клиент Weaviate еще не готов, ожидание... ({i+1}/{retries})")
        time.sleep(5) # Ждем 5 секунд перед следующей проверкой

    if not ready:
        raise WeaviateConnectionError("Клиент Weaviate подключился, но не перешел в состояние готовности.")

    logger.info("Успешно подключено к Weaviate и клиент готов.")

# Перехватываем специфичную ошибку подключения Weaviate
except WeaviateConnectionError as e:
    logger.error(f"Ошибка подключения к Weaviate (WeaviateConnectionError): {e}")
    logger.error(traceback.format_exc())
    if "connection refused" in str(e).lower():
        logger.error("Совет: Убедитесь, что Weaviate запущен и доступен по указанному адресу/порту.")
    elif "timed out" in str(e).lower():
         logger.error("Совет: Weaviate не ответил вовремя. Проверьте статус Weaviate, системные ресурсы и сетевое соединение.")
    raise
except Exception as e:
    # Ловим другие возможные ошибки при инициализации клиента
    logger.error(f"Не удалось инициализировать клиент Weaviate: {e}")
    logger.error(traceback.format_exc())
    raise

# --- Создание или проверка схемы/коллекции Weaviate (v4 API Style) ---
try:
    logger.info(f"Проверка существования коллекции '{CLASS_NAME}'...")
    if not client.collections.exists(CLASS_NAME):
        logger.info(f"Коллекция '{CLASS_NAME}' не найдена. Создание новой коллекции...")
        client.collections.create(
            name=CLASS_NAME,
            # Определяем, что векторизация будет внешней (мы сами предоставляем векторы)
            vectorizer_config=Configure.Vectorizer.none(),
            # Конфигурируем индекс HNSW (часто используется по умолчанию)
            vector_index_config=Configure.VectorIndex.hnsw(
                distance_metric=VectorDistances.COSINE
            ),
            # Определяем свойства/поля документа
            properties=[
                Property(name="text", data_type=DataType.TEXT, description="Содержимое чанка"),
                Property(name="filename", data_type=DataType.TEXT, description="Имя исходного файла"),
                Property(name="chunk_index", data_type=DataType.INT, description="Порядковый номер чанка в файле"),
                Property(name="page_number", data_type=DataType.INT, description="Номер страницы (если доступно)"),
                Property(name="element_types", data_type=DataType.TEXT, description="JSON строка со списком типов элементов unstructured"),
            ],
            # Можно добавить конфигурацию для генеративных модулей, если нужно
            # generative_config=Configure.Generative.openai() # Пример
        )
        logger.info(f"Коллекция '{CLASS_NAME}' успешно создана.")
    else:
        logger.info(f"Коллекция '{CLASS_NAME}' уже существует. Пропускаем создание.")

except Exception as e:
    logger.error(f"Ошибка при создании/проверке коллекции '{CLASS_NAME}': {e}")
    logger.error(traceback.format_exc())
    if client.is_connected():
        try:
            client.close()
            logger.info("Соединение с Weaviate безопасно закрыто")
        except Exception as e:
            logger.error(f"Ошибка при закрытии соединения: {e}")
    raise

def process_file(file_path, filename, collection_name, client, embedder):
    """
    Обрабатывает один файл: извлекает, чанкует, векторизует и сохраняет в Weaviate.

    Args:
        file_path (str): Полный путь к файлу.
        filename (str): Имя файла.
        collection_name (str): Имя коллекции Weaviate для сохранения данных.
        client (weaviate.Client): Клиент Weaviate для выполнения операций.
        embedder (SentenceTransformer): Модель для генерации эмбеддингов.

    Returns:
        bool: True, если файл успешно обработан и загружен, False в случае ошибки.
    """
    global CHUNK_MAX_CHARS, CHUNK_OVERLAP

    try:
        logger.info(f"Начало обработки файла: {filename}")
        # 1. Извлечение элементов с помощью unstructured
        try:
            logger.debug(f"Вызов partition для {filename} со стратегией 'hi_res'")
            # Убедимся, что необходимые зависимости для partition установлены (например, pdf)
            elements = partition(
                filename=file_path,
                strategy="hi_res", # Используем hi_res для лучшего качества на PDF
                languages=['rus', 'eng'] # Можно указать языки для улучшения OCR
            )
            logger.info(f"Извлечено {len(elements)} элементов из {filename}.")
        except ImportError as ie:
             logger.error(f"Ошибка импорта для обработки {filename}. Установите нужные зависимости: 'pip install unstructured[pdf,docx,...]'. Ошибка: {ie}")
             return False # Не удалось обработать
        except Exception as e:
            logger.error(f"Ошибка partition для файла {filename}: {e}")
            elements = [] # Продолжаем, но без элементов

        if not elements:
             logger.warning(f"Не найдено элементов в файле {filename} после partition.")
             # Считаем файл обработанным (пустым), если не было ошибок на этапе partition
             # Можно вернуть True, если пустые файлы не считать ошибкой
             return True

        # 2. Разделение на текстовые элементы и таблицы
        text_based_elements = [el for el in elements if hasattr(el, 'text') and not isinstance(el, Table)]
        table_elements = [el for el in elements if isinstance(el, Table)]
        # list_item_elements = [el for el in elements if isinstance(el, ListItem)] # Пример для списков

        logger.debug(f"Найдено текстовых элементов: {len(text_based_elements)}, таблиц: {len(table_elements)}")

        # 3. Чанкинг текстовых элементов
        text_chunks = []
        if text_based_elements:
            try:
                logger.debug(f"Чанкинг {len(text_based_elements)} текстовых элементов...")
                text_chunks = chunk_by_title(
                    elements=text_based_elements,
                    max_characters=CHUNK_MAX_CHARS,
                    new_after_n_chars=int(CHUNK_MAX_CHARS * 0.95), # Начинать новый чанк чуть раньше max_characters
                    combine_text_under_n_chars=200, # Объединять мелкие элементы
                    overlap=CHUNK_OVERLAP
                    # include_metadata=True # <-- УДАЛЕНО, т.к. вызывало ошибку
                )
                logger.info(f"Создано {len(text_chunks)} текстовых чанков для {filename}.")
            except Exception as e:
                logger.error(f"Ошибка chunk_by_title для файла {filename}: {e}")
                logger.warning("Используется фоллбэк: 1 элемент = 1 чанк")
                # В качестве фоллбэка используем исходные текстовые элементы как чанки
                # Убедимся, что они имеют атрибут 'text'
                text_chunks = [el for el in text_based_elements if hasattr(el, 'text')]

        # 4. Подготовка данных для Weaviate (объединение чанков и таблиц)
        data_to_embed = [] # Список текстов для векторизации
        data_objects = []  # Список объектов для Weaviate

        global_chunk_index = 0

        # Обработка текстовых чанков
        for chunk in text_chunks:
            # Безопасно получаем текст чанка
            chunk_text = getattr(chunk, 'text', None)
            if not chunk_text or not chunk_text.strip():
                logger.warning(f"Пропущен пустой текстовый чанк в {filename} (индекс до обработки: {global_chunk_index})")
                continue

            # Безопасно получаем объект метаданных
            metadata = getattr(chunk, 'metadata', None)
            page_number = None
            # Определяем тип самого "чанка" (может быть CompositeElement или исходный тип при фоллбэке)
            element_types = [type(chunk).__name__]

            if metadata:
                # ИСПОЛЬЗУЕМ GETATTR: Безопасный доступ к атрибутам ElementMetadata
                page_number = getattr(metadata, 'page_number', None)

                # Получение типов исходных элементов (если это CompositeElement)
                orig_elements = getattr(metadata, 'orig_elements', [])
                if orig_elements:
                    # Собираем типы исходных элементов
                    element_types.extend(type(el).__name__ for el in orig_elements)
                # Если это не композитный элемент (фоллбэк), но есть метаданные
                # и номер страницы, тип элемента уже определен как type(chunk).__name__

            data_to_embed.append(chunk_text)
            data_objects.append({
                "text": chunk_text,
                "filename": filename,
                "chunk_index": global_chunk_index,
                "page_number": page_number if page_number is not None else 0, # Ставим 0, если номер страницы недоступен
                "element_types": json.dumps(list(set(element_types))), # Уникальные типы в JSON
            })
            global_chunk_index += 1

        # Обработка таблиц (каждая таблица как отдельный "чанк")
        for table in table_elements:
            table_text_raw = getattr(table, 'text', '')
            table_text_processed = f"Table:\n{table_text_raw}" # Базовое представление

            # Безопасно получаем объект метаданных таблицы
            table_metadata = getattr(table, 'metadata', None)
            page_number = None
            table_html = None

            if table_metadata:
                # ИСПОЛЬЗУЕМ GETATTR: Безопасный доступ к атрибутам ElementMetadata
                page_number = getattr(table_metadata, 'page_number', None)
                table_html = getattr(table_metadata, 'text_as_html', None)

            # Попробуем получить более красивое представление таблицы из HTML
            if table_html:
                try:
                    # Обернем в теги для pandas, если их нет
                    if not table_html.strip().lower().startswith('<table'):
                        table_html = f"<table>{table_html}</table>"
                    # Используем io.StringIO для чтения HTML строки в pandas
                    from io import StringIO
                    dfs = pd.read_html(StringIO(table_html))
                    if dfs:
                        # Берем первую найденную таблицу
                        table_text_processed = f"Table:\n{dfs[0].to_string(index=False, na_rep='-')}"
                except ImportError:
                    logger.warning("Библиотека 'lxml' не установлена. Установите ее ('pip install lxml') для лучшей обработки HTML таблиц pandas. Используется необработанный текст.")
                except Exception as e:
                    # Логируем ошибку обработки HTML, но продолжаем с необработанным текстом
                    logger.warning(f"Не удалось обработать HTML таблицы в {filename} с pandas: {e}. Используется необработанный текст.")

            # Проверяем, есть ли текст после всех обработок
            if not table_text_processed.strip() or table_text_processed.strip() == "Table:":
                 logger.warning(f"Пропущена пустая таблица или таблица без текста в {filename} (индекс до обработки: {global_chunk_index})")
                 continue

            data_to_embed.append(table_text_processed)
            data_objects.append({
                "text": table_text_processed,
                "filename": filename,
                "chunk_index": global_chunk_index,
                "page_number": page_number if page_number is not None else 0, # Ставим 0, если недоступно
                "element_types": json.dumps([type(table).__name__]), # Тип элемента - Table
            })
            global_chunk_index += 1

        # 5. Генерация эмбеддингов
        if not data_to_embed:
            logger.warning(f"Нет данных для векторизации и сохранения в файле {filename}")
            return True # Обработка завершена без ошибок, но данных для загрузки нет

        logger.info(f"Генерация {len(data_to_embed)} эмбеддингов для {filename}...")
        try:
            embeddings = embedder.encode(data_to_embed, show_progress_bar=False) # show_progress_bar можно включить для длинных файлов
            vectors = [embedding.tolist() for embedding in embeddings]
            if vectors: # Проверка, что векторы сгенерировались
              logger.debug(f"Эмбеддинги сгенерированы, размерность: {len(vectors[0])}")
            else:
              logger.error(f"Модель эмбеддингов не вернула векторов для {filename}")
              return False
        except Exception as e:
             logger.error(f"Ошибка при генерации эмбеддингов для {filename}: {e}")
             logger.error(traceback.format_exc())
             return False # Ошибка на этапе векторизации

        # 6. Загрузка данных в Weaviate (пакетная обработка v4)
        logger.info(f"Загрузка {len(data_objects)} объектов в коллекцию '{collection_name}' для файла {filename}...")
        try:
            collection = client.collections.get(collection_name) # Получаем объект коллекции
            with collection.batch.dynamic() as batch:
                for i, data_obj in enumerate(data_objects):
                    unique_identifier = f"{filename}_{data_obj['chunk_index']}"
                    object_uuid = generate_uuid5(unique_identifier)
                    if i < len(vectors):
                        batch.add_object(
                            properties=data_obj,
                            uuid=object_uuid,
                            vector=vectors[i]
                        )
                    else:
                        logger.error(f"Отсутствует вектор для объекта {i} (UUID {object_uuid}) в файле {filename}. Пропуск объекта.")
            if batch.number_errors > 0:
                logger.error(f"Возникло {batch.number_errors} ошибок при пакетной загрузке для {filename}.")
                return False
            else:
                logger.info(f"Успешно загружено {len(data_objects)} объектов для файла {filename}.")
                return True
        except Exception as e:
            logger.error(f"Ошибка при пакетной загрузке данных в Weaviate для {filename}: {e}")
            logger.error(traceback.format_exc())
            return False

    except Exception as e:
        # Общий обработчик ошибок для всего про
        # цесса файла
        logger.error(f"Критическая ошибка при обработке файла {filename}: {e}")
        logger.error(traceback.format_exc()) # Логируем traceback здесь
        return False # Файл не обработан из-за критической ошибки


def process_folder(folder_path, collection_name, client, embedder):
    """Обрабатывает все поддерживаемые файлы в указанной папке."""
    processed_count = 0
    error_count = 0
    skipped_count = 0

    if not os.path.exists(folder_path):
        logger.warning(f"Папка '{folder_path}' не существует. Создайте папку и поместите в нее файлы.")
        return

    supported_extensions = ('.pdf', '.docx', '.txt', '.md', '.html', '.json', '.eml')
    files_to_process = []
    try:
        for item in os.listdir(folder_path):
            item_path = os.path.join(folder_path, item)
            if os.path.isfile(item_path) and item.lower().endswith(supported_extensions):
                files_to_process.append(item)
    except Exception as e:
        logger.error(f"Не удалось прочитать содержимое папки '{folder_path}': {e}")
        return

    if not files_to_process:
        logger.warning(f"В папке '{folder_path}' нет поддерживаемых файлов ({', '.join(supported_extensions)}).")
        return

    logger.info(f"Начинаем обработку {len(files_to_process)} файлов из папки '{folder_path}'...")

    for filename in files_to_process:
        file_path = os.path.join(folder_path, filename)
        try:
            success = process_file(file_path, filename, collection_name, client, embedder)
            if success:
                processed_count += 1
            else:
                error_count += 1
        except Exception as e:
            logger.error(f"Неперехваченная ошибка при обработке файла {filename}: {e}")
            logger.error(traceback.format_exc())
            error_count += 1

    logger.info(f"Обработка папки '{folder_path}' завершена.")
    logger.info(f"Итог: Успешно обработано = {processed_count}, С ошибками = {error_count}, Пропущено = {skipped_count}")


def search_chunks(query, collection_name, client, embedder, n_results=5, alpha=0.5):
    """Выполняет гибридный поиск в Weaviate (v4 API)."""
    if not client or not client.is_ready():
         logger.error("Клиент Weaviate не подключен или не готов. Поиск невозможен.")
         return None

    logger.info(f"Выполнение гибридного поиска в коллекции '{collection_name}'...")
    logger.debug(f"Запрос: '{query}', Лимит: {n_results}, Alpha: {alpha}")
    try:
        # 1. Получаем вектор запроса
        query_embedding = embedder.encode(query).tolist()

        # 2. Получаем объект коллекции
        collection = client.collections.get(collection_name)

        # 3. Выполняем гибридный поиск
        response = collection.query.hybrid(
            query=query,            # Текстовый запрос для BM25 части
            vector=query_embedding, # Вектор запроса для векторной части
            alpha=alpha,            # Баланс между BM25 (0) и вектором (1)
            limit=n_results,
            # Указываем, какие метаданные вернуть (используем MetadataQuery)
            return_metadata=MetadataQuery(distance=True),
            # Указываем, какие свойства вернуть
            return_properties=[
                "text", "filename", "chunk_index", "page_number", "element_types"
            ]
        )

        # 4. Форматируем результаты
        formatted_results = {
            "documents": [],
            "metadatas": [],
            "distances": []
        }

        if response and response.objects:
            logger.info(f"Найдено {len(response.objects)} релевантных объектов.")
            for obj in response.objects:
                formatted_results["documents"].append(obj.properties.get("text", "N/A"))
                # Безопасно декодируем JSON строку с типами элементов
                element_types_str = obj.properties.get("element_types", "[]")
                try:
                    element_types_list = json.loads(element_types_str)
                except json.JSONDecodeError:
                    logger.warning(f"Не удалось декодировать JSON element_types: {element_types_str} для UUID {obj.uuid}")
                    element_types_list = ["parsing_error"]

                formatted_results["metadatas"].append({
                    "filename": obj.properties.get("filename", "N/A"),
                    "chunk_index": obj.properties.get("chunk_index", -1),
                    "page_number": obj.properties.get("page_number", 0),
                    "element_types": element_types_list,
                    "uuid": str(obj.uuid) # Добавим UUID для отладки
                })
                # Получаем расстояние из метаданных
                distance = obj.metadata.distance if obj.metadata else None
                formatted_results["distances"].append(distance)
        else:
            logger.info("Поиск не вернул результатов.")

        return formatted_results

    except Exception as e:
        logger.error(f"Ошибка во время гибридного поиска: {e}")
        logger.error(traceback.format_exc())
        return None

# --- Основной блок выполнения ---
if __name__ == "__main__":
    logger.info("Запуск основного процесса...")
    try:
        # Инициализация модели эмбеддингов
        logger.info(f"Загрузка модели эмбеддингов: {EMBEDDING_MODEL}...")
        embedder = SentenceTransformer(EMBEDDING_MODEL, use_auth_token=HF_TOKEN)
        logger.info("Модель эмбеддингов успешно загружена.")
        # Инициализация клиента Weaviate через контекстный менеджер
        with weaviate.connect_to_local(
            host=WEAVIATE_HOST,
            port=WEAVIATE_PORT,
            grpc_port=WEAVIATE_GRPC_PORT
        ) as client:
            # Коллекция уже создана/проверена в глобальной части скрипта
            # Просто обработка папки с файлами и загрузка в Weaviate
            process_folder(FOLDER_PATH, CLASS_NAME, client, embedder)
            # 2. Пример выполнения поиска после обработки
            logger.info("\n--- Пример Поиска ---")
            query = input("Введите ваш поисковый запрос: ")
            if query:
                search_results = search_chunks(
                    query=query,
                    collection_name=CLASS_NAME,
                    client=client,
                    embedder=embedder,
                    n_results=5,
                    alpha=0.5
                )
                if search_results and search_results['documents']:
                    print(f"\nРезультаты поиска для: '{query}':")
                    for doc, meta, dist in zip(search_results['documents'], search_results['metadatas'], search_results['distances']):
                        print("-" * 30)
                        print(f"Источник: {meta['filename']} (Чанк: {meta['chunk_index']}, Стр: {meta['page_number']}, UUID: {meta['uuid']})")
                        print(f"Типы элементов: {meta['element_types']}")
                        print(f"Расстояние: {dist:.4f}" if dist is not None else "Расстояние: N/A (возможно, только BM25 результат?)")
                        print("Текст:")
                        print(f"{doc[:1000]}...\n")
                else:
                    print(f"\nНе найдено релевантных документов для запроса: '{query}'")
            else:
                print("Поисковый запрос не был введен.")
    except Exception as e:
        logger.critical(f"Произошла критическая ошибка в основном блоке: {e}")
        logger.critical(traceback.format_exc())
    finally:
        logger.info("Процесс завершен.")
