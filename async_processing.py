import asyncio
import aiohttp
import logging
from typing import List, Dict, Any, Optional
from tenacity import retry, stop_after_attempt, wait_exponential
from monitoring import monitor_time, PerformanceMonitor
import concurrent.futures
from functools import partial

logger = logging.getLogger(__name__)

class AsyncProcessor:
    def __init__(self, max_concurrent_tasks: int = 5, timeout: float = 30.0):
        self.semaphore = asyncio.Semaphore(max_concurrent_tasks)
        self.session = None
        self.timeout = timeout
        self._tasks: List[asyncio.Task] = []
        self._executor = concurrent.futures.ThreadPoolExecutor(max_workers=max_concurrent_tasks)
    
    async def __aenter__(self):
        self.session = aiohttp.ClientSession()
        return self
    
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        if self.session:
            await self.session.close()
        # Отменяем все активные задачи при выходе
        for task in self._tasks:
            if not task.done():
                task.cancel()
        # Ждем завершения всех задач
        if self._tasks:
            await asyncio.gather(*self._tasks, return_exceptions=True)
        self._executor.shutdown(wait=True)
    
    @retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=4, max=10))
    async def process_document(self, document: Dict[str, Any], timeout: Optional[float] = None) -> Dict[str, Any]:
        """Асинхронная обработка одного документа с таймаутом"""
        timeout = timeout or self.timeout
        async with self.semaphore:
            try:
                async with asyncio.timeout(timeout):
                    # Здесь будет логика обработки документа
                    # Например, создание эмбеддингов или другие операции
                    return document
            except asyncio.TimeoutError:
                logger.error(f"Таймаут при обработке документа после {timeout} секунд")
                PerformanceMonitor.record_error('document_processing')
                raise
            except Exception as e:
                logger.error(f"Ошибка при обработке документа: {e}")
                PerformanceMonitor.record_error('document_processing')
                raise
    
    async def process_batch(self, documents: List[Dict[str, Any]], timeout: Optional[float] = None) -> List[Dict[str, Any]]:
        """Асинхронная обработка батча документов с возможностью отмены"""
        tasks = []
        for doc in documents:
            task = asyncio.create_task(self.process_document(doc, timeout))
            self._tasks.append(task)
            tasks.append(task)
        
        try:
            results = await asyncio.gather(*tasks, return_exceptions=True)
            return [r for r in results if not isinstance(r, Exception)]
        finally:
            # Очищаем завершенные задачи
            self._tasks = [t for t in self._tasks if not t.done()]

    @monitor_time('batch_processing')
    async def process_all(self, documents: List[Dict[str, Any]], batch_size: int = 10) -> List[Dict[str, Any]]:
        """Обработка всех документов с батчингом"""
        results = []
        for i in range(0, len(documents), batch_size):
            batch = documents[i:i + batch_size]
            batch_results = await self.process_batch(batch)
            results.extend(batch_results)
        return results

    def cancel_all_tasks(self):
        """Отменяет все активные задачи"""
        for task in self._tasks:
            if not task.done():
                task.cancel()

class AsyncEmbedding:
    def __init__(self, model, batch_size: int = 32, timeout: float = 30.0):
        self.model = model
        self.batch_size = batch_size
        self.timeout = timeout
        self._executor = concurrent.futures.ThreadPoolExecutor(max_workers=1)
    
    async def _encode_in_thread(self, texts: List[str]) -> List[List[float]]:
        """Выполняет encode в отдельном потоке"""
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(
            self._executor,
            partial(self.model.encode, texts)
        )
    
    @monitor_time('embedding_generation')
    async def generate_embeddings(self, texts: List[str]) -> List[List[float]]:
        """Асинхронная генерация эмбеддингов с таймаутом"""
        embeddings = []
        for i in range(0, len(texts), self.batch_size):
            batch = texts[i:i + self.batch_size]
            try:
                async with asyncio.timeout(self.timeout):
                    batch_embeddings = await self._encode_in_thread(batch)
                    embeddings.extend(batch_embeddings)
            except asyncio.TimeoutError:
                logger.error(f"Таймаут при генерации эмбеддингов после {self.timeout} секунд")
                PerformanceMonitor.record_error('embedding_generation')
                raise
            except Exception as e:
                logger.error(f"Ошибка при генерации эмбеддингов для батча: {e}")
                PerformanceMonitor.record_error('embedding_generation')
                raise
        return embeddings
    
    def __del__(self):
        self._executor.shutdown(wait=False)

class AsyncWeaviateClient:
    def __init__(self, client, batch_size: int = 100, timeout: float = 30.0):
        self.client = client
        self.batch_size = batch_size
        self.timeout = timeout
    
    @monitor_time('weaviate_batch_import')
    async def batch_import(self, objects: List[Dict[str, Any]], class_name: str) -> None:
        """Асинхронный импорт объектов в Weaviate с таймаутом"""
        for i in range(0, len(objects), self.batch_size):
            batch = objects[i:i + self.batch_size]
            try:
                async with asyncio.timeout(self.timeout):
                    # Используем контекстный менеджер batch для автоматической отправки
                    with self.client.batch as batch_context:
                        for obj in batch:
                            batch_context.add_data_object(
                                data_object=obj,
                                class_name=class_name
                            )
            except asyncio.TimeoutError:
                logger.error(f"Таймаут при импорте батча после {self.timeout} секунд")
                PerformanceMonitor.record_error('weaviate_import')
                raise
            except Exception as e:
                logger.error(f"Ошибка при импорте батча в Weaviate: {e}")
                PerformanceMonitor.record_error('weaviate_import')
                raise

# Пример использования:
async def main():
    documents = [{"text": f"Document {i}"} for i in range(100)]
    
    async with AsyncProcessor() as processor:
        try:
            processed_docs = await processor.process_all(documents)
            logger.info(f"Обработано {len(processed_docs)} документов")
        except asyncio.CancelledError:
            logger.info("Обработка была отменена")
        except Exception as e:
            logger.error(f"Ошибка при обработке документов: {e}")

if __name__ == "__main__":
    asyncio.run(main()) 