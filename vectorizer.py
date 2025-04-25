"""
Модуль для векторизации чанков и параграфов и их загрузки в векторные базы данных.
Поддерживает работу с Weaviate и Qdrant с двухуровневой структурой данных.
Оптимизирован для гибридной RAG-архитектуры с поддержкой JSON-метаданных.
"""
import logging
import json
import traceback
import numpy as np
import time
from pathlib import Path
from typing import List, Dict, Any, Optional, Union, Tuple, Set
from datetime import datetime
from sentence_transformers import SentenceTransformer
from tenacity import retry, stop_after_attempt, wait_exponential, retry_if_exception_type

# Импорт клиентов векторных БД (с обработкой возможных ошибок импорта)
weaviate_available = True
qdrant_available = True

try:
    import weaviate
    from weaviate.exceptions import WeaviateException
except ImportError:
    weaviate_available = False

try:
    from qdrant_client import QdrantClient
    from qdrant_client.http import models
except ImportError:
    qdrant_available = False

# Импорт модулей пайплайна
from chunker import get_chunk_text, flatten_chunks, flatten_paragraphs

# Настройка логирования
logger = logging.getLogger(__name__)

# Константы
DEFAULT_MODEL_NAME = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
WEAVIATE_URL = "http://localhost:8080"
QDRANT_URL = "http://localhost:6333"
CHUNK_COLLECTION_NAME = "article_chunks"
PARAGRAPH_COLLECTION_NAME = "article_paragraphs"


class Vectorizer:
    """
    Класс для векторизации чанков и параграфов и их загрузки в векторные базы данных.
    Поддерживает работу с Weaviate и Qdrant для гибридной RAG-архитектуры.
    """
    
    def __init__(self, model_name: str = DEFAULT_MODEL_NAME):
        """
        Инициализация векторизатора.
        
        Args:
            model_name: Имя модели SentenceTransformers для генерации эмбеддингов
        """
        self.model_name = model_name
        self.model = None
        self.vector_size = 0
        self.weaviate_client = None
        self.qdrant_client = None
        
        # Инициализация модели эмбеддингов
        self._init_model()
        
        # Инициализация клиентов векторных БД (если доступны)
        self._init_weaviate()
        self._init_qdrant()
    
    def _init_model(self) -> None:
        """
        Инициализирует модель для генерации эмбеддингов.
        Логирует ошибки при неудаче.
        """
        try:
            logger.info(f"Загрузка модели эмбеддингов: {self.model_name}")
            self.model = SentenceTransformer(self.model_name)
            self.vector_size = self.model.get_sentence_embedding_dimension()
            logger.info(f"Модель загружена. Размерность векторов: {self.vector_size}")
        except Exception as e:
            logger.error(f"Ошибка при загрузке модели {self.model_name}: {str(e)}")
            logger.debug(f"Трассировка: {traceback.format_exc()}")
            raise
    
    def _init_weaviate(self) -> None:
        """
        Инициализирует клиент Weaviate (если доступен).
        Создает необходимые схемы данных для чанков и параграфов.
        """
        if not weaviate_available:
            logger.warning("Weaviate недоступен (библиотека не установлена)")
            return
        
        try:
            self.weaviate_client = weaviate.Client(WEAVIATE_URL)
            
            # Проверка доступности Weaviate
            if not self.weaviate_client.is_ready():
                logger.warning(f"Weaviate не готов на {WEAVIATE_URL}")
                self.weaviate_client = None
                return
            
            # Создание схемы, если не существует
            self._create_weaviate_schema()
            
            logger.info(f"Подключение к Weaviate установлено: {WEAVIATE_URL}")
            
        except Exception as e:
            logger.warning(f"Не удалось подключиться к Weaviate ({WEAVIATE_URL}): {str(e)}")
            logger.debug(f"Трассировка: {traceback.format_exc()}")
            self.weaviate_client = None
    
    def _init_qdrant(self) -> None:
        """
        Инициализирует клиент Qdrant (если доступен).
        Создает необходимые коллекции для чанков и параграфов.
        """
        if not qdrant_available:
            logger.warning("Qdrant недоступен (библиотека не установлена)")
            return
        
        try:
            self.qdrant_client = QdrantClient(url=QDRANT_URL)
            
            # Проверка доступности Qdrant
            self.qdrant_client.get_collections()
            
            # Создание коллекций, если не существуют
            self._create_qdrant_collections()
            
            logger.info(f"Подключение к Qdrant установлено: {QDRANT_URL}")
            
        except Exception as e:
            logger.warning(f"Не удалось подключиться к Qdrant ({QDRANT_URL}): {str(e)}")
            logger.debug(f"Трассировка: {traceback.format_exc()}")
            self.qdrant_client = None
    
    def _create_weaviate_schema(self) -> None:
        """
        Создает схему в Weaviate для хранения чанков и параграфов статей.
        Поддерживает хранение метаданных для гибридной RAG-архитектуры.
        """
        if not self.weaviate_client:
            return
        
        try:
            # 1. Создание класса для чанков
            if not self.weaviate_client.schema.exists(CHUNK_COLLECTION_NAME):
                chunk_class = {
                    "class": CHUNK_COLLECTION_NAME,
                    "description": "Чанки статей TechnoNICOL с метаданными для гибридной RAG-системы",
                    "vectorizer": "none",  # Мы будем предоставлять собственные векторы
                    "properties": [
                        {
                            "name": "article_id",
                            "dataType": ["int"],
                            "description": "ID статьи в базе данных"
                        },
                        {
                            "name": "chunk_id",
                            "dataType": ["string"],
                            "description": "ID чанка в статье"
                        },
                        {
                            "name": "title",
                            "dataType": ["string"],
                            "description": "Заголовок чанка"
                        },
                        {
                            "name": "content",
                            "dataType": ["text"],
                            "description": "Текстовое содержимое чанка"
                        },
                        {
                            "name": "labels",
                            "dataType": ["string[]"],
                            "description": "Метки чанка"
                        },
                        {
                            "name": "article_title",
                            "dataType": ["string"],
                            "description": "Заголовок статьи"
                        },
                        {
                            "name": "article_url",
                            "dataType": ["string"],
                            "description": "URL статьи"
                        },
                        {
                            "name": "breadcrumbs",
                            "dataType": ["string[]"],
                            "description": "Хлебные крошки статьи"
                        },
                        {
                            "name": "outgoing_links",
                            "dataType": ["string[]"],
                            "description": "Исходящие ссылки из чанка"
                        },
                        {
                            "name": "metadata_json",
                            "dataType": ["text"],
                            "description": "JSON-метаданные для гибридной RAG-системы"
                        },
                        {
                            "name": "has_tech_specs",
                            "dataType": ["boolean"],
                            "description": "Флаг наличия технических характеристик"
                        },
                        {
                            "name": "has_numbers",
                            "dataType": ["boolean"],
                            "description": "Флаг наличия числовых данных"
                        },
                        {
                            "name": "indexed_at",
                            "dataType": ["date"],
                            "description": "Дата и время индексации"
                        }
                    ]
                }
                
                self.weaviate_client.schema.create_class(chunk_class)
                logger.info(f"Создан класс {CHUNK_COLLECTION_NAME} в Weaviate")
            
            # 2. Создание класса для параграфов
            if not self.weaviate_client.schema.exists(PARAGRAPH_COLLECTION_NAME):
                paragraph_class = {
                    "class": PARAGRAPH_COLLECTION_NAME,
                    "description": "Параграфы статей TechnoNICOL с метаданными для гибридной RAG-системы",
                    "vectorizer": "none",  # Мы будем предоставлять собственные векторы
                    "properties": [
                        {
                            "name": "article_id",
                            "dataType": ["int"],
                            "description": "ID статьи в базе данных"
                        },
                        {
                            "name": "chunk_id",
                            "dataType": ["string"],
                            "description": "ID чанка, содержащего параграф"
                        },
                        {
                            "name": "paragraph_id",
                            "dataType": ["string"],
                            "description": "ID параграфа"
                        },
                        {
                            "name": "text",
                            "dataType": ["text"],
                            "description": "Текст параграфа"
                        },
                        {
                            "name": "labels",
                            "dataType": ["string[]"],
                            "description": "Метки параграфа"
                        },
                        {
                            "name": "chunk_title",
                            "dataType": ["string"],
                            "description": "Заголовок родительского чанка"
                        },
                        {
                            "name": "article_title",
                            "dataType": ["string"],
                            "description": "Заголовок статьи"
                        },
                        {
                            "name": "article_url",
                            "dataType": ["string"],
                            "description": "URL статьи"
                        },
                        {
                            "name": "metadata_json",
                            "dataType": ["text"],
                            "description": "JSON-метаданные для гибридной RAG-системы"
                        },
                        {
                            "name": "has_numeric_data",
                            "dataType": ["boolean"],
                            "description": "Флаг наличия числовых данных в параграфе"
                        },
                        {
                            "name": "indexed_at",
                            "dataType": ["date"],
                            "description": "Дата и время индексации"
                        }
                    ]
                }
                
                self.weaviate_client.schema.create_class(paragraph_class)
                logger.info(f"Создан класс {PARAGRAPH_COLLECTION_NAME} в Weaviate")
            
        except Exception as e:
            logger.error(f"Ошибка при создании схемы в Weaviate: {str(e)}")
            logger.debug(f"Трассировка: {traceback.format_exc()}")
    
    def _create_qdrant_collections(self) -> None:
        """
        Создает коллекции в Qdrant для хранения чанков и параграфов статей.
        Настраивает параметры индексации для оптимальной производительности.
        """
        if not self.qdrant_client:
            return
        
        try:
            # Получение списка коллекций
            collections = self.qdrant_client.get_collections().collections
            collection_names = [c.name for c in collections]
            
            # 1. Создание коллекции для чанков
            if CHUNK_COLLECTION_NAME not in collection_names:
                self.qdrant_client.create_collection(
                    collection_name=CHUNK_COLLECTION_NAME,
                    vectors_config=models.VectorParams(
                        size=self.vector_size,
                        distance=models.Distance.COSINE
                    ),
                    optimizers_config=models.OptimizersConfigDiff(
                        indexing_threshold=20000,  # Оптимизация для большого количества данных
                        payload_indexing=models.PayloadIndexParams(
                            indexed_fields=["labels", "has_tech_specs", "has_numbers"]
                        )
                    )
                )
                logger.info(f"Создана коллекция {CHUNK_COLLECTION_NAME} в Qdrant")
            
            # 2. Создание коллекции для параграфов
            if PARAGRAPH_COLLECTION_NAME not in collection_names:
                self.qdrant_client.create_collection(
                    collection_name=PARAGRAPH_COLLECTION_NAME,
                    vectors_config=models.VectorParams(
                        size=self.vector_size,
                        distance=models.Distance.COSINE
                    ),
                    optimizers_config=models.OptimizersConfigDiff(
                        indexing_threshold=50000,  # Оптимизация для очень большого количества данных
                        payload_indexing=models.PayloadIndexParams(
                            indexed_fields=["labels", "has_numeric_data"]
                        )
                    )
                )
                logger.info(f"Создана коллекция {PARAGRAPH_COLLECTION_NAME} в Qdrant")
            
        except Exception as e:
            logger.error(f"Ошибка при создании коллекций в Qdrant: {str(e)}")
            logger.debug(f"Трассировка: {traceback.format_exc()}")
    
    def generate_embedding(self, text: str) -> np.ndarray:
        """
        Генерирует эмбеддинг для текста.
        
        Args:
            text: Входной текст
            
        Returns:
            np.ndarray: Вектор эмбеддинга
        """
        if not self.model:
            error_msg = "Модель для эмбеддингов не инициализирована"
            logger.error(error_msg)
            raise ValueError(error_msg)
        
        try:
            # Генерация эмбеддинга
            embedding = self.model.encode(text, show_progress_bar=False)
            return embedding
            
        except Exception as e:
            logger.error(f"Ошибка при генерации эмбеддинга: {str(e)}")
            logger.debug(f"Трассировка: {traceback.format_exc()}")
            # В случае ошибки возвращаем вектор из нулей правильного размера
            return np.zeros(self.vector_size)
    
    @retry(stop=stop_after_attempt(3), 
        wait=wait_exponential(multiplier=1, min=1, max=10), 
        retry=retry_if_exception_type(Exception))
    def index_chunk_in_weaviate(self, chunk_data: Dict[str, Any], article_id: int, 
                            article_title: str, article_url: str, 
                            embedding: np.ndarray, breadcrumbs: List[str] = None) -> bool:
        """
        Индексирует чанк в Weaviate.
        
        Args:
            chunk_data: Данные чанка
            article_id: ID статьи
            article_title: Заголовок статьи
            article_url: URL статьи
            embedding: Вектор эмбеддинга
            breadcrumbs: Хлебные крошки статьи
            
        Returns:
            bool: True, если индексация успешна
        """
        if not self.weaviate_client:
            return False
        
        try:
            # Извлечение метаданных для индексации
            metadata = chunk_data.get("metadata", {})
            metadata_json = json.dumps(metadata, ensure_ascii=False) if metadata else None
            
            # Извлечение специальных флагов для быстрого поиска
            has_tech_specs = bool(metadata.get("tech_specs", []))
            has_numbers = bool(chunk_data.get("numbers", []))
            
            # Подготовка данных для Weaviate
            data_object = {
                "article_id": article_id,
                "chunk_id": chunk_data.get("id", ""),
                "title": chunk_data.get("title", "") or "",
                "content": get_chunk_text(chunk_data),
                "labels": chunk_data.get("labels", []),
                "article_title": article_title,
                "article_url": article_url,
                "breadcrumbs": breadcrumbs or [],
                "outgoing_links": chunk_data.get("outgoing_links", []),
                "metadata_json": metadata_json,
                "has_tech_specs": has_tech_specs,
                "has_numbers": has_numbers,
                "indexed_at": datetime.now().isoformat()
            }
            
            # Уникальный идентификатор для объекта
            uuid = weaviate.util.generate_uuid5(chunk_data.get("id", ""))
            
            # Добавление объекта с предоставленным эмбеддингом
            self.weaviate_client.data_object.create(
                data_object=data_object,
                class_name=CHUNK_COLLECTION_NAME,
                uuid=uuid,
                vector=embedding.tolist()
            )
            
            logger.debug(f"Чанк {chunk_data.get('id', '')} проиндексирован в Weaviate")
            return True
            
        except Exception as e:
            logger.error(f"Ошибка при индексации чанка в Weaviate: {str(e)}")
            logger.debug(f"Трассировка: {traceback.format_exc()}")
            raise  # Для повторной попытки через retry
    
    @retry(stop=stop_after_attempt(3), 
        wait=wait_exponential(multiplier=1, min=1, max=10), 
        retry=retry_if_exception_type(Exception))
    def index_paragraph_in_weaviate(self, paragraph: Dict[str, Any], chunk_data: Dict[str, Any],
                                article_id: int, article_title: str, article_url: str, 
                                embedding: np.ndarray) -> bool:
        """
        Индексирует параграф в Weaviate.
        
        Args:
            paragraph: Данные параграфа
            chunk_data: Данные родительского чанка
            article_id: ID статьи
            article_title: Заголовок статьи
            article_url: URL статьи
            embedding: Вектор эмбеддинга
            
        Returns:
            bool: True, если индексация успешна
        """
        if not self.weaviate_client:
            return False
        
        try:
            # Получение меток параграфа
            paragraph_id = paragraph.get("id", "")
            labels = []
            if "paragraph_labels" in chunk_data and paragraph_id in chunk_data["paragraph_labels"]:
                labels = chunk_data["paragraph_labels"][paragraph_id]
            elif "labels" in paragraph:
                labels = paragraph.get("labels", [])
            
            # Извлечение метаданных
            metadata = paragraph.get("metadata", {})
            metadata_json = json.dumps(metadata, ensure_ascii=False) if metadata else None
            
            # Определение наличия числовых данных
            has_numeric_data = False
            if metadata and "has_numeric_data" in metadata:
                has_numeric_data = metadata["has_numeric_data"]
            
            # Подготовка данных для Weaviate
            data_object = {
                "article_id": article_id,
                "chunk_id": chunk_data.get("id", ""),
                "paragraph_id": paragraph_id,
                "text": paragraph.get("text", ""),
                "labels": labels,
                "chunk_title": chunk_data.get("title", "") or "",
                "article_title": article_title,
                "article_url": article_url,
                "metadata_json": metadata_json,
                "has_numeric_data": has_numeric_data,
                "indexed_at": datetime.now().isoformat()
            }
            
            # Уникальный идентификатор для объекта
            uuid = weaviate.util.generate_uuid5(paragraph_id)
            
            # Добавление объекта с предоставленным эмбеддингом
            self.weaviate_client.data_object.create(
                data_object=data_object,
                class_name=PARAGRAPH_COLLECTION_NAME,
                uuid=uuid,
                vector=embedding.tolist()
            )
            
            logger.debug(f"Параграф {paragraph_id} проиндексирован в Weaviate")
            return True
            
        except Exception as e:
            logger.error(f"Ошибка при индексации параграфа в Weaviate: {str(e)}")
            logger.debug(f"Трассировка: {traceback.format_exc()}")
            raise  # Для повторной попытки через retry
    
    @retry(stop=stop_after_attempt(3), 
        wait=wait_exponential(multiplier=1, min=1, max=10), 
        retry=retry_if_exception_type(Exception))
    def index_chunk_in_qdrant(self, chunk_data: Dict[str, Any], article_id: int, 
                            article_title: str, article_url: str, 
                            embedding: np.ndarray, breadcrumbs: List[str] = None) -> bool:
        """
        Индексирует чанк в Qdrant.
        
        Args:
            chunk_data: Данные чанка
            article_id: ID статьи
            article_title: Заголовок статьи
            article_url: URL статьи
            embedding: Вектор эмбеддинга
            breadcrumbs: Хлебные крошки статьи
            
        Returns:
            bool: True, если индексация успешна
        """
        if not self.qdrant_client:
            return False
        
        try:
            # Извлечение метаданных
            metadata = chunk_data.get("metadata", {})
            
            # Извлечение специальных флагов для быстрого поиска
            has_tech_specs = bool(metadata.get("tech_specs", []))
            has_numbers = bool(chunk_data.get("numbers", []))
            
            # Подготовка данных для Qdrant
            # В Qdrant можно хранить вложенные структуры в payload, поэтому сохраняем метаданные напрямую
            payload = {
                "article_id": article_id,
                "chunk_id": chunk_data.get("id", ""),
                "title": chunk_data.get("title", "") or "",
                "content": get_chunk_text(chunk_data),
                "labels": chunk_data.get("labels", []),
                "article_title": article_title,
                "article_url": article_url,
                "breadcrumbs": breadcrumbs or [],
                "outgoing_links": chunk_data.get("outgoing_links", []),
                "metadata": metadata,  # Полные метаданные в JSON-совместимом формате
                "has_tech_specs": has_tech_specs,
                "has_numbers": has_numbers,
                "indexed_at": datetime.now().isoformat()
            }
            
            # Добавление точки в коллекцию
            self.qdrant_client.upsert(
                collection_name=CHUNK_COLLECTION_NAME,
                points=[
                    models.PointStruct(
                        id=chunk_data.get("id", ""),
                        vector=embedding.tolist(),
                        payload=payload
                    )
                ]
            )
            
            logger.debug(f"Чанк {chunk_data.get('id', '')} проиндексирован в Qdrant")
            return True
            
        except Exception as e:
            logger.error(f"Ошибка при индексации чанка в Qdrant: {str(e)}")
            logger.debug(f"Трассировка: {traceback.format_exc()}")
            raise  # Для повторной попытки через retry
    
    @retry(stop=stop_after_attempt(3), 
        wait=wait_exponential(multiplier=1, min=1, max=10), 
        retry=retry_if_exception_type(Exception))
    def index_paragraph_in_qdrant(self, paragraph: Dict[str, Any], chunk_data: Dict[str, Any],
                            article_id: int, article_title: str, article_url: str, 
                            embedding: np.ndarray) -> bool:
        """
        Индексирует параграф в Qdrant.
        
        Args:
            paragraph: Данные параграфа
            chunk_data: Данные родительского чанка
            article_id: ID статьи
            article_title: Заголовок статьи
            article_url: URL статьи
            embedding: Вектор эмбеддинга
            
        Returns:
            bool: True, если индексация успешна
        """
        if not self.qdrant_client:
            return False
        
        try:
            # Получение меток параграфа
            paragraph_id = paragraph.get("id", "")
            labels = []
            if "paragraph_labels" in chunk_data and paragraph_id in chunk_data["paragraph_labels"]:
                labels = chunk_data["paragraph_labels"][paragraph_id]
            elif "labels" in paragraph:
                labels = paragraph.get("labels", [])
            
            # Извлечение метаданных
            metadata = paragraph.get("metadata", {})
            
            # Определение наличия числовых данных
            has_numeric_data = False
            if metadata and "has_numeric_data" in metadata:
                has_numeric_data = metadata["has_numeric_data"]
            
            # Подготовка данных для Qdrant
            payload = {
                "article_id": article_id,
                "chunk_id": chunk_data.get("id", ""),
                "paragraph_id": paragraph_id,
                "text": paragraph.get("text", ""),
                "labels": labels,
                "chunk_title": chunk_data.get("title", "") or "",
                "article_title": article_title,
                "article_url": article_url,
                "metadata": metadata,  # Полные метаданные
                "has_numeric_data": has_numeric_data,
                "indexed_at": datetime.now().isoformat()
            }
            
            # Добавление точки в коллекцию
            self.qdrant_client.upsert(
                collection_name=PARAGRAPH_COLLECTION_NAME,
                points=[
                    models.PointStruct(
                        id=paragraph_id,
                        vector=embedding.tolist(),
                        payload=payload
                    )
                ]
            )
            
            logger.debug(f"Параграф {paragraph_id} проиндексирован в Qdrant")
            return True
            
        except Exception as e:
            logger.error(f"Ошибка при индексации параграфа в Qdrant: {str(e)}")
            logger.debug(f"Трассировка: {traceback.format_exc()}")
            raise  # Для повторной попытки через retry
    
    def index_chunk(self, chunk_data: Dict[str, Any], article_id: int, 
                article_title: str, article_url: str, 
                breadcrumbs: List[str] = None) -> bool:
        """
        Индексирует чанк в векторных базах данных.
        
        Args:
            chunk_data: Данные чанка
            article_id: ID статьи
            article_title: Заголовок статьи
            article_url: URL статьи
            breadcrumbs: Хлебные крошки статьи
            
        Returns:
            bool: True, если индексация успешна хотя бы в одной БД
        """
        chunk_id = chunk_data.get("id", "unknown")
        chunk_text = get_chunk_text(chunk_data)
        
        if not chunk_text.strip():
            logger.warning(f"Пустой текст в чанке {chunk_id}, пропуск индексации")
            return False
        
        try:
            # Генерация эмбеддинга
            embedding = self.generate_embedding(chunk_text)
            
            # Индексация в доступных векторных БД
            weaviate_success = False
            qdrant_success = False
            
            if self.weaviate_client:
                try:
                    weaviate_success = self.index_chunk_in_weaviate(
                        chunk_data, article_id, article_title, article_url, embedding, breadcrumbs
                    )
                except Exception as e:
                    logger.error(f"Не удалось проиндексировать в Weaviate чанк {chunk_id}: {str(e)}")
                    logger.debug(f"Трассировка: {traceback.format_exc()}")
            
            if self.qdrant_client:
                try:
                    qdrant_success = self.index_chunk_in_qdrant(
                        chunk_data, article_id, article_title, article_url, embedding, breadcrumbs
                    )
                except Exception as e:
                    logger.error(f"Не удалось проиндексировать в Qdrant чанк {chunk_id}: {str(e)}")
                    logger.debug(f"Трассировка: {traceback.format_exc()}")
            
            # Индексация каждого параграфа в чанке
            if (weaviate_success or qdrant_success) and "paragraphs" in chunk_data and chunk_data["paragraphs"]:
                for paragraph in chunk_data["paragraphs"]:
                    self.index_paragraph(
                        paragraph, chunk_data, article_id, article_title, article_url
                    )
            
            return weaviate_success or qdrant_success
            
        except Exception as e:
            logger.error(f"Ошибка при индексации чанка {chunk_id}: {str(e)}")
            logger.debug(f"Трассировка: {traceback.format_exc()}")
            return False
    
    def index_paragraph(self, paragraph: Dict[str, Any], chunk_data: Dict[str, Any],
                    article_id: int, article_title: str, article_url: str) -> bool:
        """
        Индексирует параграф в векторных базах данных.
        
        Args:
            paragraph: Данные параграфа
            chunk_data: Данные родительского чанка
            article_id: ID статьи
            article_title: Заголовок статьи
            article_url: URL статьи
            
        Returns:
            bool: True, если индексация успешна хотя бы в одной БД
        """
        paragraph_id = paragraph.get("id", "unknown")
        paragraph_text = paragraph.get("text", "")
        
        if not paragraph_text.strip():
            logger.warning(f"Пустой текст в параграфе {paragraph_id}, пропуск индексации")
            return False
        
        try:
            # Добавление контекста из заголовка чанка для улучшения семантики
            context_prefix = ""
            if chunk_data.get("title"):
                context_prefix = f"{chunk_data['title']}: "
            
            # Генерация эмбеддинга с контекстом
            full_text = context_prefix + paragraph_text
            embedding = self.generate_embedding(full_text)
            
            # Индексация в доступных векторных БД
            weaviate_success = False
            qdrant_success = False
            
            if self.weaviate_client:
                try:
                    weaviate_success = self.index_paragraph_in_weaviate(
                        paragraph, chunk_data, article_id, article_title, article_url, embedding
                    )
                except Exception as e:
                    logger.error(f"Не удалось проиндексировать в Weaviate параграф {paragraph_id}: {str(e)}")
                    logger.debug(f"Трассировка: {traceback.format_exc()}")
            
            if self.qdrant_client:
                try:
                    qdrant_success = self.index_paragraph_in_qdrant(
                        paragraph, chunk_data, article_id, article_title, article_url, embedding
                    )
                except Exception as e:
                    logger.error(f"Не удалось проиндексировать в Qdrant параграф {paragraph_id}: {str(e)}")
                    logger.debug(f"Трассировка: {traceback.format_exc()}")
            
            return weaviate_success or qdrant_success
            
        except Exception as e:
            logger.error(f"Ошибка при индексации параграфа {paragraph_id}: {str(e)}")
            logger.debug(f"Трассировка: {traceback.format_exc()}")
            return False
    
    def search_similar(self, query_text: str, collection: str = "both", 
                    top_k: int = 5, min_score: float = 0.65,
                    filter_labels: List[str] = None,
                    filter_params: Dict[str, Any] = None) -> List[Dict[str, Any]]:
        """
        Выполняет поиск похожих чанков или параграфов.
        
        Args:
            query_text: Текст запроса
            collection: Коллекция для поиска ("chunks", "paragraphs" или "both")
            top_k: Количество возвращаемых результатов
            min_score: Минимальный порог сходства
            filter_labels: Список меток для фильтрации (если None, фильтрация не применяется)
            filter_params: Дополнительные параметры фильтрации для гибридной RAG-системы
            
        Returns:
            List[Dict[str, Any]]: Список похожих элементов
        """
        try:
            # Генерация эмбеддинга для запроса
            query_embedding = self.generate_embedding(query_text)
            
            results = []
            
            # Проверка наличия фильтра по техническим параметрам для гибридной RAG-системы
            has_tech_filter = filter_params and any(k in filter_params for k in 
                                                    ["has_tech_specs", "has_numbers", "has_numeric_data"])
            
            # Поиск в коллекции чанков
            if collection in ["chunks", "both"]:
                # Предпочитаем Qdrant для гибридных запросов, так как он лучше работает с JSON-метаданными
                if self.qdrant_client and (has_tech_filter or not self.weaviate_client):
                    try:
                        chunk_results = self._search_in_qdrant(
                            query_embedding, CHUNK_COLLECTION_NAME, top_k, min_score, 
                            filter_labels, filter_params
                        )
                        for result in chunk_results:
                            result["entity_type"] = "chunk"
                        results.extend(chunk_results)
                    except Exception as e:
                        logger.error(f"Ошибка при поиске чанков в Qdrant: {str(e)}")
                        logger.debug(f"Трассировка: {traceback.format_exc()}")
                
                elif self.weaviate_client:
                    try:
                        chunk_results = self._search_in_weaviate(
                            query_embedding, CHUNK_COLLECTION_NAME, top_k, min_score, 
                            filter_labels, filter_params
                        )
                        for result in chunk_results:
                            result["entity_type"] = "chunk"
                        results.extend(chunk_results)
                    except Exception as e:
                        logger.error(f"Ошибка при поиске чанков в Weaviate: {str(e)}")
                        logger.debug(f"Трассировка: {traceback.format_exc()}")
            
            # Поиск в коллекции параграфов
            if collection in ["paragraphs", "both"]:
                if self.qdrant_client and (has_tech_filter or not self.weaviate_client):
                    try:
                        paragraph_results = self._search_in_qdrant(
                            query_embedding, PARAGRAPH_COLLECTION_NAME, top_k, min_score, 
                            filter_labels, filter_params
                        )
                        for result in paragraph_results:
                            result["entity_type"] = "paragraph"
                        results.extend(paragraph_results)
                    except Exception as e:
                        logger.error(f"Ошибка при поиске параграфов в Qdrant: {str(e)}")
                        logger.debug(f"Трассировка: {traceback.format_exc()}")
                
                elif self.weaviate_client:
                    try:
                        paragraph_results = self._search_in_weaviate(
                            query_embedding, PARAGRAPH_COLLECTION_NAME, top_k, min_score, 
                            filter_labels, filter_params
                        )
                        for result in paragraph_results:
                            result["entity_type"] = "paragraph"
                        results.extend(paragraph_results)
                    except Exception as e:
                        logger.error(f"Ошибка при поиске параграфов в Weaviate: {str(e)}")
                        logger.debug(f"Трассировка: {traceback.format_exc()}")
            
            # Сортировка результатов по релевантности (score)
            results.sort(key=lambda x: x.get("score", 0), reverse=True)
            
            # Ограничение общего числа результатов
            if len(results) > top_k:
                results = results[:top_k]
            
            return results
            
        except Exception as e:
            logger.error(f"Ошибка при поиске похожих элементов: {str(e)}")
            logger.debug(f"Трассировка: {traceback.format_exc()}")
            return []
    
    def _search_in_qdrant(self, query_embedding: np.ndarray, collection_name: str, 
                        top_k: int, min_score: float, 
                        filter_labels: List[str] = None,
                        filter_params: Dict[str, Any] = None) -> List[Dict[str, Any]]:
        """
        Выполняет поиск в Qdrant.
        
        Args:
            query_embedding: Эмбеддинг запроса
            collection_name: Имя коллекции
            top_k: Количество результатов
            min_score: Минимальный порог сходства
            filter_labels: Список меток для фильтрации
            filter_params: Дополнительные параметры фильтрации
            
        Returns:
            List[Dict[str, Any]]: Результаты поиска
        """
        # Создание фильтра
        filter_query = None
        must_conditions = []
        
        # Добавление фильтра по меткам, если они указаны
        if filter_labels and len(filter_labels) > 0:
            must_conditions.append(
                models.FieldCondition(
                    key="labels",
                    match=models.MatchAny(any=filter_labels)
                )
            )
        
        # Добавление фильтров для гибридной RAG-системы
        if filter_params:
            # Фильтр по наличию технических характеристик
            if "has_tech_specs" in filter_params:
                must_conditions.append(
                    models.FieldCondition(
                        key="has_tech_specs",
                        match=models.MatchValue(value=True)
                    )
                )
            
            # Фильтр по наличию числовых данных
            if "has_numbers" in filter_params and collection_name == CHUNK_COLLECTION_NAME:
                must_conditions.append(
                    models.FieldCondition(
                        key="has_numbers",
                        match=models.MatchValue(value=True)
                    )
                )
            
            if "has_numeric_data" in filter_params and collection_name == PARAGRAPH_COLLECTION_NAME:
                must_conditions.append(
                    models.FieldCondition(
                        key="has_numeric_data",
                        match=models.MatchValue(value=True)
                    )
                )
        
        # Применяем фильтры, если они есть
        if must_conditions:
            filter_query = models.Filter(must=must_conditions)
        
        try:
            search_result = self.qdrant_client.search(
                collection_name=collection_name,
                query_vector=query_embedding.tolist(),
                limit=top_k,
                score_threshold=min_score,
                query_filter=filter_query
            )
            
            results = []
            for hit in search_result:
                # Копируем payload и добавляем score
                result = hit.payload.copy()
                result["score"] = hit.score
                
                # Разворачиваем метаданные в корень результата для совместимости
                if "metadata" in result and isinstance(result["metadata"], dict):
                    for meta_key, meta_value in result["metadata"].items():
                        if meta_key not in result:
                            result[meta_key] = meta_value
                
                results.append(result)
            
            return results
            
        except Exception as e:
            logger.error(f"Ошибка при поиске в Qdrant: {str(e)}")
            logger.debug(f"Трассировка: {traceback.format_exc()}")
            return []
    
    def _search_in_weaviate(self, query_embedding: np.ndarray, class_name: str, 
                        top_k: int, min_score: float, 
                        filter_labels: List[str] = None,
                        filter_params: Dict[str, Any] = None) -> List[Dict[str, Any]]:
        """
        Выполняет поиск в Weaviate.
        
        Args:
            query_embedding: Эмбеддинг запроса
            class_name: Имя класса
            top_k: Количество результатов
            min_score: Минимальный порог сходства
            filter_labels: Список меток для фильтрации
            filter_params: Дополнительные параметры фильтрации
            
        Returns:
            List[Dict[str, Any]]: Результаты поиска
        """
        # Определение полей для запроса в зависимости от класса
        if class_name == CHUNK_COLLECTION_NAME:
            properties = [
                "article_id", "chunk_id", "title", "content",
                "labels", "article_title", "article_url", "breadcrumbs", 
                "outgoing_links", "metadata_json", "has_tech_specs", "has_numbers"
            ]
        else:
            properties = [
                "article_id", "chunk_id", "paragraph_id", "text",
                "labels", "chunk_title", "article_title", "article_url", 
                "metadata_json", "has_numeric_data"
            ]
        
        # Создание фильтра
        where_filter = None
        filter_conditions = []
        
        # Добавление фильтра по меткам
        if filter_labels and len(filter_labels) > 0:
            label_conditions = []
            for label in filter_labels:
                label_conditions.append({
                    "path": ["labels"],
                    "operator": "Equal",
                    "valueString": label
                })
            
            if len(label_conditions) > 1:
                filter_conditions.append({
                    "operator": "Or",
                    "operands": label_conditions
                })
            else:
                filter_conditions.append(label_conditions[0])
        
        # Добавление фильтров для гибридной RAG-системы
        if filter_params:
            # Фильтр по наличию технических характеристик
            if "has_tech_specs" in filter_params and class_name == CHUNK_COLLECTION_NAME:
                filter_conditions.append({
                    "path": ["has_tech_specs"],
                    "operator": "Equal",
                    "valueBoolean": True
                })
            
            # Фильтр по наличию числовых данных
            if "has_numbers" in filter_params and class_name == CHUNK_COLLECTION_NAME:
                filter_conditions.append({
                    "path": ["has_numbers"],
                    "operator": "Equal",
                    "valueBoolean": True
                })
            
            if "has_numeric_data" in filter_params and class_name == PARAGRAPH_COLLECTION_NAME:
                filter_conditions.append({
                    "path": ["has_numeric_data"],
                    "operator": "Equal",
                    "valueBoolean": True
                })
        
        # Объединение всех условий фильтрации
        if len(filter_conditions) > 1:
            where_filter = {
                "operator": "And",
                "operands": filter_conditions
            }
        elif len(filter_conditions) == 1:
            where_filter = filter_conditions[0]
        
        try:
            # Создание запроса
            query = (self.weaviate_client.query
                    .get(class_name, properties)
                    .with_near_vector({
                        "vector": query_embedding.tolist(),
                        "certainty": min_score
                    })
                    .with_limit(top_k))
            
            # Добавление фильтра, если он есть
            if where_filter:
                query = query.with_where(where_filter)
            
            # Выполнение запроса
            query_result = query.do()
            
            results = []
            if "data" in query_result and "Get" in query_result["data"]:
                for hit in query_result["data"]["Get"][class_name]:
                    # Добавление оценки сходства
                    hit["score"] = hit.get("_additional", {}).get("certainty", 0)
                    
                    # Удаление служебных полей
                    if "_additional" in hit:
                        del hit["_additional"]
                    
                    # Десериализация метаданных, если они есть
                    if "metadata_json" in hit and hit["metadata_json"]:
                        try:
                            metadata = json.loads(hit["metadata_json"])
                            # Разворачиваем метаданные в корень результата для совместимости
                            for meta_key, meta_value in metadata.items():
                                if meta_key not in hit:
                                    hit[meta_key] = meta_value
                        except:
                            pass
                    
                    results.append(hit)
            
            return results
            
        except Exception as e:
            logger.error(f"Ошибка при поиске в Weaviate: {str(e)}")
            logger.debug(f"Трассировка: {traceback.format_exc()}")
            return []


# Создание глобального экземпляра векторизатора
_vectorizer_instance = None

def get_vectorizer(model_name: str = DEFAULT_MODEL_NAME) -> Vectorizer:
    """
    Получает или создает экземпляр векторизатора.
    
    Args:
        model_name: Имя модели для эмбеддингов
        
    Returns:
        Vectorizer: Экземпляр векторизатора
    """
    global _vectorizer_instance
    
    try:
        if _vectorizer_instance is None:
            _vectorizer_instance = Vectorizer(model_name)
            logger.info(f"Создан новый экземпляр векторизатора с моделью {model_name}")
        
        return _vectorizer_instance
        
    except Exception as e:
        logger.error(f"Ошибка при получении экземпляра векторизатора: {str(e)}")
        logger.debug(f"Трассировка: {traceback.format_exc()}")
        raise


def vectorize_article(chunks: List[Dict[str, Any]], article_id: int, 
                    article_title: str, article_url: str, 
                    breadcrumbs: List[str] = None) -> Tuple[int, int]:
    """
    Векторизирует и индексирует чанки и параграфы статьи.
    
    Args:
        chunks: Список чанков статьи
        article_id: ID статьи
        article_title: Заголовок статьи
        article_url: URL статьи
        breadcrumbs: Хлебные крошки статьи
        
    Returns:
        Tuple[int, int]: Количество успешно проиндексированных чанков и параграфов
    """
    try:
        # Получение экземпляра векторизатора
        vectorizer = get_vectorizer()
        
        # Получение плоского списка чанков
        flat_chunks = flatten_chunks(chunks)
        
        logger.info(f"Векторизация чанков и параграфов для статьи '{article_title}' (id={article_id})")
        
        # Подсчет параграфов для статистики
        total_paragraphs = sum(len(chunk.get("paragraphs", [])) for chunk in flat_chunks)
        
        logger.info(f"Найдено {len(flat_chunks)} чанков и {total_paragraphs} параграфов для индексации")
        
        # Индексация каждого чанка
        chunks_indexed = 0
        paragraphs_indexed = 0
        
        for i, chunk in enumerate(flat_chunks):
            try:
                # Логирование прогресса
                if i % 10 == 0 or i == len(flat_chunks) - 1:
                    logger.info(f"Индексация чанка {i+1}/{len(flat_chunks)}")
                else:
                    logger.debug(f"Индексация чанка {i+1}/{len(flat_chunks)}")
                
                # Индексация чанка
                success = vectorizer.index_chunk(
                    chunk, article_id, article_title, article_url, breadcrumbs
                )
                
                if success:
                    chunks_indexed += 1
                    
                    # Подсчет проиндексированных параграфов
                    if "paragraphs" in chunk:
                        paragraphs_indexed += len(chunk["paragraphs"])
                    
            except Exception as e:
                logger.error(f"Ошибка при индексации чанка {chunk.get('id', 'unknown')}: {str(e)}")
                logger.debug(f"Трассировка: {traceback.format_exc()}")
        
        # Проверка на ошибки индексации
        if chunks_indexed < len(flat_chunks):
            logger.warning(
                f"Не все чанки были успешно проиндексированы: {chunks_indexed}/{len(flat_chunks)}"
            )
        
        if paragraphs_indexed < total_paragraphs:
            logger.warning(
                f"Не все параграфы были успешно проиндексированы: {paragraphs_indexed}/{total_paragraphs}"
            )
        
        logger.info(
            f"Завершена векторизация статьи '{article_title}': "
            f"проиндексировано {chunks_indexed} чанков и {paragraphs_indexed} параграфов"
        )
        
        return chunks_indexed, paragraphs_indexed
        
    except Exception as e:
        logger.error(f"Ошибка при векторизации статьи {article_title} (id={article_id}): {str(e)}")
        logger.debug(f"Трассировка: {traceback.format_exc()}")
        return 0, 0


def search_vector_db(query: str, collection: str = "both", top_k: int = 5, 
                min_score: float = 0.65, filter_labels: List[str] = None,
                ilter_params: Dict[str, Any] = None) -> List[Dict[str, Any]]:
    """
    Выполняет векторный поиск по базе данных.
    
    Args:
        query: Текст запроса
        collection: Коллекция для поиска ("chunks", "paragraphs" или "both")
        top_k: Количество возвращаемых результатов
        min_score: Минимальный порог сходства
        filter_labels: Список меток для фильтрации
        filter_params: Дополнительные параметры фильтрации для гибридной RAG-системы
        
    Returns:
        List[Dict[str, Any]]: Результаты поиска
    """
    try:
        vectorizer = get_vectorizer()
        return vectorizer.search_similar(
            query, collection, top_k, min_score, filter_labels, filter_params
        )
    except Exception as e:
        logger.error(f"Ошибка при выполнении векторного поиска: {str(e)}")
        logger.debug(f"Трассировка: {traceback.format_exc()}")
        return []
