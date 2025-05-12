import os
from dotenv import load_dotenv
import logging
from typing import Optional
from dataclasses import dataclass
import re
from urllib.parse import urlparse

# Настройка логирования
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

@dataclass
class WeaviateConfig:
    host: str
    port: int
    grpc_port: int
    class_name: str
    data_path: Optional[str] = None

    def validate(self) -> bool:
        """Валидация конфигурации Weaviate"""
        if not self.host or not self.host.strip():
            logger.error("Хост Weaviate не может быть пустым")
            return False
        
        if not isinstance(self.port, int) or self.port <= 0 or self.port > 65535:
            logger.error(f"Некорректный порт Weaviate: {self.port}")
            return False
            
        if not isinstance(self.grpc_port, int) or self.grpc_port <= 0 or self.grpc_port > 65535:
            logger.error(f"Некорректный gRPC порт: {self.grpc_port}")
            return False
            
        if self.port == self.grpc_port:
            logger.error("HTTP и gRPC порты не могут быть одинаковыми")
            return False
            
        if not self.class_name or not re.match(r'^[A-Z][a-zA-Z0-9]*$', self.class_name):
            logger.error(f"Некорректное имя класса: {self.class_name}")
            return False
            
        if self.data_path and not os.path.exists(os.path.dirname(self.data_path)):
            logger.error(f"Путь к данным не существует: {self.data_path}")
            return False
            
        return True

@dataclass
class ModelConfig:
    embedding_model_name: str
    openrouter_api_key: Optional[str] = None
    openrouter_model: Optional[str] = None
    openrouter_api_url: Optional[str] = None

    def validate(self) -> bool:
        """Валидация конфигурации моделей"""
        if not self.embedding_model_name or not self.embedding_model_name.strip():
            logger.error("Имя модели эмбеддингов не может быть пустым")
            return False

        if self.openrouter_api_url:
            try:
                result = urlparse(self.openrouter_api_url)
                if not all([result.scheme, result.netloc]):
                    logger.error(f"Некорректный URL OpenRouter API: {self.openrouter_api_url}")
                    return False
            except Exception as e:
                logger.error(f"Ошибка при проверке URL OpenRouter API: {e}")
                return False

        if self.openrouter_model and not self.openrouter_api_key:
            logger.error("Указана модель OpenRouter, но отсутствует API ключ")
            return False

        return True

@dataclass
class ChunkingConfig:
    max_chars: int = 3000
    overlap: int = 300
    min_chars: int = 200

    def validate(self) -> bool:
        """Валидация конфигурации чанкинга"""
        if not isinstance(self.max_chars, int) or self.max_chars <= 0:
            logger.error(f"Некорректный максимальный размер чанка: {self.max_chars}")
            return False

        if not isinstance(self.overlap, int) or self.overlap < 0:
            logger.error(f"Некорректный размер перекрытия: {self.overlap}")
            return False

        if not isinstance(self.min_chars, int) or self.min_chars <= 0:
            logger.error(f"Некорректный минимальный размер чанка: {self.min_chars}")
            return False

        if self.min_chars >= self.max_chars:
            logger.error("Минимальный размер чанка должен быть меньше максимального")
            return False

        if self.overlap >= self.max_chars:
            logger.error("Размер перекрытия должен быть меньше максимального размера чанка")
            return False

        return True

class Config:
    def __init__(self):
        load_dotenv()
        
        # Weaviate конфигурация
        self.weaviate = WeaviateConfig(
            host=os.getenv("WEAVIATE_HOST", "localhost"),
            port=int(os.getenv("WEAVIATE_PORT", "8080")),
            grpc_port=int(os.getenv("WEAVIATE_GRPC_PORT", "50051")),
            class_name=os.getenv("WEAVIATE_CLASS_NAME", "DocumentChunkV4"),
            data_path=os.getenv("WEAVIATE_DATA_PATH")
        )
        
        # Модельная конфигурация
        self.model = ModelConfig(
            embedding_model_name=os.getenv("EMBEDDING_MODEL_NAME", "sentence-transformers/all-MiniLM-L6-v2"),
            openrouter_api_key=os.getenv("OPENROUTER_API_KEY"),
            openrouter_model=os.getenv("OPENROUTER_MODEL"),
            openrouter_api_url=os.getenv("OPENROUTER_API_URL")
        )
        
        # Конфигурация чанкинга
        self.chunking = ChunkingConfig(
            max_chars=int(os.getenv("CHUNK_MAX_CHARS", "3000")),
            overlap=int(os.getenv("CHUNK_OVERLAP", "300")),
            min_chars=int(os.getenv("CHUNK_MIN_CHARS", "200"))
        )

    def validate(self) -> bool:
        """Проверяет всю конфигурацию"""
        try:
            return all([
                self.weaviate.validate(),
                self.model.validate(),
                self.chunking.validate()
            ])
        except Exception as e:
            logger.error(f"Ошибка при валидации конфигурации: {e}")
            return False

    def get_weaviate_url(self) -> str:
        """Возвращает URL для подключения к Weaviate"""
        return f"http://{self.weaviate.host}:{self.weaviate.port}"

# Создаем глобальный экземпляр конфигурации
config = Config() 