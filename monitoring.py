from prometheus_client import Counter, Histogram, start_http_server, REGISTRY
import time
import logging
import functools
import asyncio
from typing import Optional, Callable
import socket

# Настройка логирования
logger = logging.getLogger(__name__)

class MetricsServer:
    _instance: Optional['MetricsServer'] = None
    _server_started = False
    
    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance
    
    @classmethod
    def start(cls, port: int = 9090, addr: str = ''):
        """Запускает сервер метрик Prometheus с проверкой занятости порта"""
        if cls._server_started:
            logger.warning("Сервер метрик уже запущен")
            return
            
        try:
            # Проверяем, не занят ли порт
            with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
                if s.connect_ex(('localhost', port)) == 0:
                    raise OSError(f"Порт {port} уже занят")
            
            start_http_server(port, addr)
            cls._server_started = True
            logger.info(f"Метрики Prometheus доступны на порту {port}")
        except Exception as e:
            logger.error(f"Не удалось запустить сервер метрик: {e}")
            raise
    
    @classmethod
    def clear(cls):
        """Очищает все метрики"""
        collectors = list(REGISTRY._collector_to_names.keys())
        for collector in collectors:
            REGISTRY.unregister(collector)
        logger.info("Все метрики очищены")

# Метрики Prometheus
REQUESTS_TOTAL = Counter(
    'rag_requests_total',
    'Total number of RAG requests',
    ['type']  # type может быть 'search', 'embedding', 'llm'
)

PROCESSING_TIME = Histogram(
    'rag_processing_seconds',
    'Time spent processing requests',
    ['operation'],
    buckets=(0.1, 0.5, 1.0, 2.0, 5.0, 10.0, float('inf'))
)

ERROR_COUNTER = Counter(
    'rag_errors_total',
    'Total number of errors',
    ['type']  # type может быть 'weaviate', 'embedding', 'llm', 'other'
)

def monitor_time(operation: str):
    """Декоратор для измерения времени выполнения операций"""
    def decorator(func: Callable):
        @functools.wraps(func)
        async def async_wrapper(*args, **kwargs):
            start_time = time.time()
            try:
                result = await func(*args, **kwargs)
                REQUESTS_TOTAL.labels(type=operation).inc()
                return result
            except Exception as e:
                ERROR_COUNTER.labels(type=operation).inc()
                raise
            finally:
                PROCESSING_TIME.labels(operation=operation).observe(
                    time.time() - start_time
                )
        
        @functools.wraps(func)
        def sync_wrapper(*args, **kwargs):
            start_time = time.time()
            try:
                result = func(*args, **kwargs)
                REQUESTS_TOTAL.labels(type=operation).inc()
                return result
            except Exception as e:
                ERROR_COUNTER.labels(type=operation).inc()
                raise
            finally:
                PROCESSING_TIME.labels(operation=operation).observe(
                    time.time() - start_time
                )
        
        return async_wrapper if asyncio.iscoroutinefunction(func) else sync_wrapper
    return decorator

class PerformanceMonitor:
    """Класс для мониторинга производительности"""
    
    @staticmethod
    def track_embedding_time(duration: float):
        """Отслеживает время создания эмбеддингов"""
        if duration < 0:
            logger.warning(f"Получено отрицательное время выполнения: {duration}")
            return
        PROCESSING_TIME.labels(operation='embedding').observe(duration)
    
    @staticmethod
    def track_search_time(duration: float):
        """Отслеживает время поиска"""
        if duration < 0:
            logger.warning(f"Получено отрицательное время выполнения: {duration}")
            return
        PROCESSING_TIME.labels(operation='search').observe(duration)
    
    @staticmethod
    def track_llm_time(duration: float):
        """Отслеживает время ответа LLM"""
        if duration < 0:
            logger.warning(f"Получено отрицательное время выполнения: {duration}")
            return
        PROCESSING_TIME.labels(operation='llm').observe(duration)
    
    @staticmethod
    def record_error(error_type: str):
        """Записывает ошибку определенного типа"""
        if error_type not in ['weaviate', 'embedding', 'llm', 'other']:
            logger.warning(f"Неизвестный тип ошибки: {error_type}")
            error_type = 'other'
        ERROR_COUNTER.labels(type=error_type).inc()

# Инициализация сервера метрик при импорте модуля
metrics_server = MetricsServer()

# Пример использования декоратора
@monitor_time('search')
async def search_documents(query: str):
    """Пример асинхронной функции поиска с мониторингом"""
    await asyncio.sleep(0.1)  # Имитация асинхронной операции
    return {"results": []}

@monitor_time('process')
def process_data(data: dict):
    """Пример синхронной функции с мониторингом"""
    time.sleep(0.1)  # Имитация обработки
    return data 