# OpenSource RAG

![GitHub Workflow Status](https://img.shields.io/github/actions/workflow/status/yourusername/OpenSource-RAG/mcp-deploy.yml?style=flat-square)
![Python Version](https://img.shields.io/badge/python-3.10-blue?style=flat-square)
![Streamlit](https://img.shields.io/badge/streamlit-1.27+-red?style=flat-square)
![Weaviate](https://img.shields.io/badge/weaviate-v4.0-green?style=flat-square)

Современное приложение для семантического поиска и генерации ответов по документам с использованием открытых инструментов и моделей.

<div align="center">
  <img src="docs/images/screenshot.png" alt="OpenSource RAG Screenshot" width="80%">
</div>

## 🌟 Особенности

- **Персональные коллекции** - каждый пользователь имеет свою изолированную коллекцию документов
- **Обработка многих форматов** - индексация PDF, DOCX, TXT, HTML и других документов
- **Векторный и гибридный поиск** - комбинация BM25 и семантического поиска для оптимальных результатов
- **Поиск в интернете** - дополнение контекста результатами из Интернета 
- **Различные LLM** - поддержка всех моделей через OpenRouter
- **Современный интерфейс** - стильный чат с удобным UX

## 🚀 Технологии

- **Frontend**: Streamlit
- **База данных**: Weaviate (векторная БД)
- **Эмбеддинги**: Sentence Transformers
- **Языковые модели**: Claude, GPT и другие через OpenRouter
- **Поиск в интернете**: Google Programmable Search Engine

## 📋 Требования

- Python 3.10+
- Weaviate сервер (локальный или облачный)
- Ключи API для OpenRouter и Google PSE
- Опционально: HuggingFace токен для загрузки моделей

## 🔧 Установка и настройка

### 1. Клонирование репозитория

```bash
git clone https://github.com/yourusername/OpenSource-RAG.git
cd OpenSource-RAG
```

### 2. Создание виртуального окружения

```bash
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate
pip install -r requirements.txt
```

### 3. Настройка окружения

Создайте файл `.env` в корне проекта:

```
# Weaviate настройки
WEAVIATE_HOST=localhost
WEAVIATE_PORT=8080
WEAVIATE_GRPC_PORT=50051

# API ключи
OPENROUTER_API_KEY=your_openrouter_key
OPENROUTER_MODEL=anthropic/claude-3-haiku
OPENROUTER_API_URL=https://openrouter.ai/api/v1
HUGGINGFACE_TOKEN=your_huggingface_token
GOOGLE_PSE_API_KEY=your_google_api_key
GOOGLE_PSE_ID=your_google_pse_id
```

### 4. Запуск Weaviate (Docker)

```bash
docker run -d --name weaviate-rag \
  -p 8080:8080 -p 50051:50051 \
  -e AUTHENTICATION_ANONYMOUS_ACCESS_ENABLED=true \
  -e PERSISTENCE_DATA_PATH=/var/lib/weaviate \
  -e DEFAULT_VECTORIZER_MODULE=text2vec-transformers \
  -e ENABLE_MODULES=text2vec-transformers \
  -e TRANSFORMERS_INFERENCE_API=http://t2v-transformers:8080 \
  -v weaviate_data:/var/lib/weaviate \
  semitechnologies/weaviate:1.22.4
```

### 5. Запуск приложения

```bash
streamlit run app.py
```

## 🔄 Развертывание с MCP GitHub

Проект настроен для автоматического развертывания через MCP GitHub. Когда вы пушите код в ветку `main`, GitHub Actions автоматически запускает процесс деплоя.

### Настройка MCP GitHub:

1. Создайте секреты в настройках репозитория GitHub (Settings > Secrets):
   - `MCP_API_KEY` - ваш ключ API для MCP
   - Другие необходимые для деплоя переменные

2. Workflow файл находится в `.github/workflows/mcp-deploy.yml`

3. Проверьте статус деплоя во вкладке Actions на GitHub

## 🧩 Структура проекта

```
OpenSource-RAG/
├── app.py              # Основной Streamlit-интерфейс
├── main_fixed.py       # Модуль обработки документов
├── README.md           # Документация проекта
├── requirements.txt    # Зависимости проекта
├── .env                # Переменные окружения (не версионируется)
├── .env-example        # Пример файла .env
├── .github/            # GitHub Actions конфигурация
└── docs/               # Документация и ресурсы
```

## 🔒 Безопасность

В текущей версии используется упрощенная аутентификация в целях демонстрации. Для производственного использования рекомендуется:

- Реализовать полноценную аутентификацию (OAuth, JWT и т.д.)
- Настроить безопасное хранение ключей API
- Включить HTTPS и другие меры безопасности

## 📝 Цитирование

При использовании этого проекта в научных или коммерческих разработках, пожалуйста, цитируйте:

```
@misc{opensourcerag2023,
  author = {Your Name},
  title = {OpenSource RAG: Retrieval Augmented Generation с открытыми инструментами},
  year = {2023},
  publisher = {GitHub},
  url = {https://github.com/yourusername/OpenSource-RAG}
}
```

## 📜 Лицензия

Этот проект распространяется под лицензией MIT. Подробности в файле [LICENSE](LICENSE).

## 🤝 Вклад в проект

Вклады приветствуются! Пожалуйста, ознакомьтесь с [руководством по внесению вклада](CONTRIBUTING.md).