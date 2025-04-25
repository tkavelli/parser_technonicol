"""
Модуль для работы с базой данных SQLite.
Создает и заполняет таблицы с данными статей, чанков и параграфов.
Поддерживает двухуровневую структуру данных и гибридную RAG-архитектуру.
"""
import os
import sqlite3
import logging
import json
import traceback
from pathlib import Path
from typing import Dict, Any, List, Optional, Union, Tuple, Set
from datetime import datetime

# Настройка логирования
logger = logging.getLogger(__name__)


def init_db(db_path: Union[str, Path]) -> None:
    """
    Инициализирует базу данных, создавая необходимые таблицы.
    
    Args:
        db_path: Путь к файлу базы данных
    """
    db_path = Path(db_path)
    
    # Создание директории для базы данных
    os.makedirs(db_path.parent, exist_ok=True)
    
    logger.info(f"Инициализация базы данных: {db_path}")
    
    conn = None
    try:
        conn = sqlite3.connect(db_path)
        cursor = conn.cursor()
        
        # Создание основных таблиц
        _create_main_tables(cursor)
        
        # Создание вспомогательных таблиц
        _create_auxiliary_tables(cursor)
        
        # Создание индексов для оптимизации запросов
        _create_indexes(cursor)
        
        conn.commit()
        logger.info("База данных успешно инициализирована")
        
    except sqlite3.Error as e:
        logger.error(f"Ошибка при инициализации базы данных: {str(e)}")
        logger.debug(f"Трассировка: {traceback.format_exc()}")
        if conn:
            conn.rollback()
        raise
    
    finally:
        if conn:
            conn.close()


def _create_main_tables(cursor: sqlite3.Cursor) -> None:
    """
    Создает основные таблицы базы данных.
    
    Args:
        cursor: Курсор SQLite
    """
    # Создание таблицы articles
    cursor.execute("""
    CREATE TABLE IF NOT EXISTS articles (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        title TEXT NOT NULL,
        url TEXT NOT NULL UNIQUE,
        slug TEXT NOT NULL UNIQUE,
        breadcrumbs TEXT,
        parsed_at TEXT NOT NULL,
        metadata TEXT
    )
    """)
    
    # Создание таблицы chunks
    cursor.execute("""
    CREATE TABLE IF NOT EXISTS chunks (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        article_id INTEGER NOT NULL,
        chunk_id TEXT NOT NULL,
        title TEXT,
        level INTEGER NOT NULL,
        content TEXT,
        labels TEXT,
        metadata TEXT,
        FOREIGN KEY (article_id) REFERENCES articles(id),
        UNIQUE (article_id, chunk_id)
    )
    """)
    
    # Создание таблицы paragraphs
    cursor.execute("""
    CREATE TABLE IF NOT EXISTS paragraphs (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        chunk_id INTEGER NOT NULL,
        paragraph_id TEXT NOT NULL,
        text TEXT NOT NULL,
        labels TEXT,
        metadata TEXT,
        FOREIGN KEY (chunk_id) REFERENCES chunks(id),
        UNIQUE (chunk_id, paragraph_id)
    )
    """)


def _create_auxiliary_tables(cursor: sqlite3.Cursor) -> None:
    """
    Создает вспомогательные таблицы базы данных.
    
    Args:
        cursor: Курсор SQLite
    """
    # Создание таблицы numbers
    cursor.execute("""
    CREATE TABLE IF NOT EXISTS numbers (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        chunk_id INTEGER NOT NULL,
        value REAL NOT NULL,
        unit TEXT NOT NULL,
        raw_text TEXT NOT NULL,
        is_range_start INTEGER DEFAULT 0,
        is_range_end INTEGER DEFAULT 0,
        FOREIGN KEY (chunk_id) REFERENCES chunks(id)
    )
    """)
    
    # Создание таблицы links
    cursor.execute("""
    CREATE TABLE IF NOT EXISTS links (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        chunk_id INTEGER NOT NULL,
        target_article_url TEXT NOT NULL,
        FOREIGN KEY (chunk_id) REFERENCES chunks(id)
    )
    """)
    
    # Создание таблицы images
    cursor.execute("""
    CREATE TABLE IF NOT EXISTS images (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        chunk_id INTEGER NOT NULL,
        filename TEXT NOT NULL,
        original_src TEXT NOT NULL,
        alt TEXT,
        caption TEXT,
        metadata TEXT,
        FOREIGN KEY (chunk_id) REFERENCES chunks(id)
    )
    """)
    
    # Создание таблицы categorized_entities
    cursor.execute("""
    CREATE TABLE IF NOT EXISTS categorized_entities (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        article_id INTEGER,
        chunk_id INTEGER,
        paragraph_id INTEGER,
        category_name TEXT NOT NULL,
        category_value TEXT NOT NULL,
        source TEXT NOT NULL, /* breadcrumb, keyword, ml, gpt */
        confidence REAL DEFAULT 1.0,
        FOREIGN KEY (article_id) REFERENCES articles(id),
        FOREIGN KEY (chunk_id) REFERENCES chunks(id),
        FOREIGN KEY (paragraph_id) REFERENCES paragraphs(id)
    )
    """)
    
    # Таблица для технических характеристик
    cursor.execute("""
    CREATE TABLE IF NOT EXISTS tech_specs (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        chunk_id INTEGER NOT NULL,
        name TEXT NOT NULL,
        value TEXT NOT NULL,
        has_numeric_value INTEGER DEFAULT 0,
        numeric_value REAL,
        unit TEXT,
        FOREIGN KEY (chunk_id) REFERENCES chunks(id)
    )
    """)


def _create_indexes(cursor: sqlite3.Cursor) -> None:
    """
    Создает индексы для оптимизации запросов.
    
    Args:
        cursor: Курсор SQLite
    """
    # Индексы для основных таблиц
    cursor.execute("CREATE INDEX IF NOT EXISTS idx_articles_slug ON articles(slug)")
    cursor.execute("CREATE INDEX IF NOT EXISTS idx_articles_url ON articles(url)")
    
    cursor.execute("CREATE INDEX IF NOT EXISTS idx_chunks_article_id ON chunks(article_id)")
    cursor.execute("CREATE INDEX IF NOT EXISTS idx_chunks_chunk_id ON chunks(chunk_id)")
    
    cursor.execute("CREATE INDEX IF NOT EXISTS idx_paragraphs_chunk_id ON paragraphs(chunk_id)")
    cursor.execute("CREATE INDEX IF NOT EXISTS idx_paragraphs_paragraph_id ON paragraphs(paragraph_id)")
    
    # Индексы для вспомогательных таблиц
    cursor.execute("CREATE INDEX IF NOT EXISTS idx_numbers_chunk_id ON numbers(chunk_id)")
    cursor.execute("CREATE INDEX IF NOT EXISTS idx_numbers_unit ON numbers(unit)")
    cursor.execute("CREATE INDEX IF NOT EXISTS idx_numbers_value ON numbers(value)")
    
    cursor.execute("CREATE INDEX IF NOT EXISTS idx_links_chunk_id ON links(chunk_id)")
    cursor.execute("CREATE INDEX IF NOT EXISTS idx_links_target_url ON links(target_article_url)")
    
    cursor.execute("CREATE INDEX IF NOT EXISTS idx_images_chunk_id ON images(chunk_id)")
    
    cursor.execute("CREATE INDEX IF NOT EXISTS idx_entities_article_id ON categorized_entities(article_id)")
    cursor.execute("CREATE INDEX IF NOT EXISTS idx_entities_chunk_id ON categorized_entities(chunk_id)")
    cursor.execute("CREATE INDEX IF NOT EXISTS idx_entities_paragraph_id ON categorized_entities(paragraph_id)")
    cursor.execute("CREATE INDEX IF NOT EXISTS idx_entities_category ON categorized_entities(category_name, category_value)")
    
    cursor.execute("CREATE INDEX IF NOT EXISTS idx_tech_specs_chunk_id ON tech_specs(chunk_id)")
    cursor.execute("CREATE INDEX IF NOT EXISTS idx_tech_specs_name ON tech_specs(name)")
    cursor.execute("CREATE INDEX IF NOT EXISTS idx_tech_specs_numeric ON tech_specs(has_numeric_value, numeric_value)")


def save_to_db(article_data: Dict[str, Any], db_path: Union[str, Path]) -> int:
    """
    Сохраняет данные статьи в базу данных.
    
    Args:
        article_data: Структурированные данные статьи
        db_path: Путь к файлу базы данных
        
    Returns:
        int: ID статьи в базе данных
    """
    db_path = Path(db_path)
    
    logger.info(f"Сохранение статьи '{article_data.get('title', 'без заголовка')}' в базу данных")
    
    conn = None
    try:
        conn = sqlite3.connect(db_path)
        conn.row_factory = sqlite3.Row
        cursor = conn.cursor()
        
        # Начало транзакции
        conn.execute("BEGIN TRANSACTION")
        
        # 1. Сохранение статьи
        article_id = _save_article(cursor, article_data)
        
        # 2. Сохранение чанков и связанных данных
        _save_chunks(cursor, article_data.get("chunks", []), article_id)
        
        # Подтверждение транзакции
        conn.commit()
        
        # Подсчет вставленных записей для логирования
        stats = _get_save_statistics(cursor, article_id)
        
        logger.info(
            f"Данные сохранены в БД: статья id={article_id}, "
            f"{stats['chunks_count']} чанков, {stats['paragraphs_count']} параграфов, "
            f"{stats['numbers_count']} числовых данных, {stats['links_count']} ссылок, "
            f"{stats['images_count']} изображений, {stats['categories_count']} категорий"
        )
        
        return article_id
        
    except sqlite3.Error as e:
        if conn:
            conn.rollback()
        logger.error(f"Ошибка при сохранении в базу данных: {str(e)}")
        logger.debug(f"Трассировка: {traceback.format_exc()}")
        raise
    
    finally:
        if conn:
            conn.close()


def _get_save_statistics(cursor: sqlite3.Cursor, article_id: int) -> Dict[str, int]:
    """
    Получает статистику о количестве сохраненных записей.
    
    Args:
        cursor: Курсор SQLite
        article_id: ID статьи
        
    Returns:
        Dict[str, int]: Статистика сохраненных записей
    """
    stats = {}
    
    # Количество чанков
    cursor.execute("SELECT COUNT(*) FROM chunks WHERE article_id = ?", (article_id,))
    stats["chunks_count"] = cursor.fetchone()[0]
    
    # Количество параграфов
    cursor.execute(
        "SELECT COUNT(*) FROM paragraphs WHERE chunk_id IN (SELECT id FROM chunks WHERE article_id = ?)", 
        (article_id,)
    )
    stats["paragraphs_count"] = cursor.fetchone()[0]
    
    # Количество числовых данных
    cursor.execute(
        "SELECT COUNT(*) FROM numbers WHERE chunk_id IN (SELECT id FROM chunks WHERE article_id = ?)",
        (article_id,)
    )
    stats["numbers_count"] = cursor.fetchone()[0]
    
    # Количество ссылок
    cursor.execute(
        "SELECT COUNT(*) FROM links WHERE chunk_id IN (SELECT id FROM chunks WHERE article_id = ?)",
        (article_id,)
    )
    stats["links_count"] = cursor.fetchone()[0]
    
    # Количество изображений
    cursor.execute(
        "SELECT COUNT(*) FROM images WHERE chunk_id IN (SELECT id FROM chunks WHERE article_id = ?)",
        (article_id,)
    )
    stats["images_count"] = cursor.fetchone()[0]
    
    # Количество категорий
    cursor.execute(
        """
        SELECT COUNT(*) FROM categorized_entities 
        WHERE article_id = ? OR chunk_id IN (SELECT id FROM chunks WHERE article_id = ?)
        OR paragraph_id IN (SELECT id FROM paragraphs WHERE chunk_id IN 
                            (SELECT id FROM chunks WHERE article_id = ?))
        """, 
        (article_id, article_id, article_id)
    )
    stats["categories_count"] = cursor.fetchone()[0]
    
    # Количество технических спецификаций
    cursor.execute(
        """
        SELECT COUNT(*) FROM tech_specs
        WHERE chunk_id IN (SELECT id FROM chunks WHERE article_id = ?)
        """,
        (article_id,)
    )
    stats["tech_specs_count"] = cursor.fetchone()[0]
    
    return stats


def _save_article(cursor: sqlite3.Cursor, article_data: Dict[str, Any]) -> int:
    """
    Сохраняет информацию о статье в базу данных.
    
    Args:
        cursor: Курсор SQLite
        article_data: Данные статьи
        
    Returns:
        int: ID статьи в базе данных
    """
    # Сериализация хлебных крошек в JSON (если есть)
    breadcrumbs_json = None
    if "breadcrumbs" in article_data and article_data["breadcrumbs"]:
        breadcrumbs_json = json.dumps(article_data["breadcrumbs"], ensure_ascii=False)
    
    # Сериализация метаданных (если есть)
    metadata_json = None
    if "metadata" in article_data and article_data["metadata"]:
        metadata_json = json.dumps(article_data["metadata"], ensure_ascii=False)
    
    # Проверка, существует ли статья с таким URL
    cursor.execute("SELECT id FROM articles WHERE url = ?", (article_data.get("url", ""),))
    existing = cursor.fetchone()
    
    if existing:
        # Статья уже существует, обновляем
        article_id = existing[0]
        cursor.execute("""
        UPDATE articles 
        SET title = ?, slug = ?, breadcrumbs = ?, parsed_at = ?, metadata = ? 
        WHERE id = ?
        """, (
            article_data.get("title", ""),
            article_data.get("slug", ""),
            breadcrumbs_json,
            article_data.get("parsed_at", datetime.now().isoformat()),
            metadata_json,
            article_id
        ))
        logger.debug(f"Обновлена существующая статья с id={article_id}")
    else:
        # Вставка новой статьи
        cursor.execute("""
        INSERT INTO articles (title, url, slug, breadcrumbs, parsed_at, metadata) 
        VALUES (?, ?, ?, ?, ?, ?)
        """, (
            article_data.get("title", ""),
            article_data.get("url", ""),
            article_data.get("slug", ""),
            breadcrumbs_json,
            article_data.get("parsed_at", datetime.now().isoformat()),
            metadata_json
        ))
        article_id = cursor.lastrowid
        logger.debug(f"Создана новая статья с id={article_id}")
    
    return article_id


def _save_chunks(cursor: sqlite3.Cursor, chunks: List[Dict[str, Any]], article_id: int, 
                parent_db_id: Optional[int] = None) -> None:
    """
    Рекурсивно сохраняет чанки и связанные с ними данные.
    
    Args:
        cursor: Курсор SQLite
        chunks: Список чанков
        article_id: ID статьи
        parent_db_id: ID родительского чанка в БД (для рекурсии)
    """
    for chunk in chunks:
        # 1. Сохранение чанка
        chunk_db_id = _save_chunk(cursor, chunk, article_id)
        
        # 2. Сохранение параграфов
        if "paragraphs" in chunk and chunk["paragraphs"]:
            _save_paragraphs(cursor, chunk["paragraphs"], chunk_db_id, chunk.get("paragraph_labels", {}))
        
        # 3. Сохранение числовых данных
        _save_numbers(cursor, chunk.get("numbers", []), chunk_db_id)
        
        # 4. Сохранение исходящих ссылок
        _save_links(cursor, chunk.get("outgoing_links", []), chunk_db_id)
        
        # 5. Сохранение изображений
        _save_images(cursor, chunk.get("images", []), chunk_db_id)
        
        # 6. Сохранение категорий/меток
        _save_categories(cursor, article_id, chunk_db_id, None, chunk.get("labels", []), "keyword")
        
        # 7. Сохранение технических характеристик (для гибридной RAG-системы)
        _save_tech_specs(cursor, chunk, chunk_db_id)
        
        # 8. Рекурсивная обработка дочерних чанков
        if chunk.get("children"):
            _save_chunks(cursor, chunk["children"], article_id, chunk_db_id)


def _save_chunk(cursor: sqlite3.Cursor, chunk: Dict[str, Any], article_id: int) -> int:
    """
    Сохраняет чанк в базу данных.
    
    Args:
        cursor: Курсор SQLite
        chunk: Данные чанка
        article_id: ID статьи
        
    Returns:
        int: ID чанка в базе данных
    """
    # Получение текстового содержимого чанка из элементов
    content = get_chunk_text(chunk)
    
    # Сериализация списка меток в строку JSON
    labels_json = json.dumps(chunk.get("labels", []), ensure_ascii=False) if chunk.get("labels") else "[]"
    
    # Сериализация метаданных
    metadata_json = None
    if "metadata" in chunk and chunk["metadata"]:
        metadata_json = json.dumps(chunk["metadata"], ensure_ascii=False)
    
    # Проверка, существует ли чанк с таким ID для статьи
    cursor.execute(
        "SELECT id FROM chunks WHERE article_id = ? AND chunk_id = ?",
        (article_id, chunk.get("id", ""))
    )
    existing = cursor.fetchone()
    
    if existing:
        # Чанк существует, обновляем
        chunk_db_id = existing[0]
        cursor.execute("""
        UPDATE chunks 
        SET title = ?, level = ?, content = ?, labels = ?, metadata = ?
        WHERE id = ?
        """, (
            chunk.get("title", ""),
            chunk.get("level", 0),
            content,
            labels_json,
            metadata_json,
            chunk_db_id
        ))
        logger.debug(f"Обновлен существующий чанк с id={chunk_db_id}")
    else:
        # Вставка нового чанка
        cursor.execute("""
        INSERT INTO chunks (article_id, chunk_id, title, level, content, labels, metadata) 
        VALUES (?, ?, ?, ?, ?, ?, ?)
        """, (
            article_id,
            chunk.get("id", ""),
            chunk.get("title", ""),
            chunk.get("level", 0),
            content,
            labels_json,
            metadata_json
        ))
        chunk_db_id = cursor.lastrowid
        logger.debug(f"Создан новый чанк с id={chunk_db_id}")
    
    return chunk_db_id


def _save_paragraphs(cursor: sqlite3.Cursor, paragraphs: List[Dict[str, Any]], 
                    chunk_db_id: int, paragraph_labels: Dict[str, List[str]]) -> None:
    """
    Сохраняет параграфы чанка.
    
    Args:
        cursor: Курсор SQLite
        paragraphs: Список параграфов
        chunk_db_id: ID чанка в базе данных
        paragraph_labels: Словарь с метками параграфов {paragraph_id: [labels]}
    """
    # Удаление старых записей для этого чанка
    cursor.execute("DELETE FROM paragraphs WHERE chunk_id = ?", (chunk_db_id,))
    
    # Вставка новых записей
    for paragraph in paragraphs:
        paragraph_id = paragraph.get("id", "")
        
        # Получение меток для этого параграфа
        labels = paragraph_labels.get(paragraph_id, [])
        labels_json = json.dumps(labels, ensure_ascii=False) if labels else "[]"
        
        # Сериализация метаданных
        metadata_json = None
        if "metadata" in paragraph and paragraph["metadata"]:
            metadata_json = json.dumps(paragraph["metadata"], ensure_ascii=False)
        
        cursor.execute("""
        INSERT INTO paragraphs (chunk_id, paragraph_id, text, labels, metadata) 
        VALUES (?, ?, ?, ?, ?)
        """, (
            chunk_db_id,
            paragraph_id,
            paragraph.get("text", ""),
            labels_json,
            metadata_json
        ))
        paragraph_db_id = cursor.lastrowid
        
        # Сохранение категорий/меток параграфа
        if labels:
            _save_categories(cursor, None, chunk_db_id, paragraph_db_id, labels, "keyword")


def _save_numbers(cursor: sqlite3.Cursor, numbers: List[Dict[str, Any]], chunk_db_id: int) -> None:
    """
    Сохраняет числовые данные чанка.
    
    Args:
        cursor: Курсор SQLite
        numbers: Список числовых данных
        chunk_db_id: ID чанка в базе данных
    """
    # Удаление старых записей для этого чанка
    cursor.execute("DELETE FROM numbers WHERE chunk_id = ?", (chunk_db_id,))
    
    # Вставка новых записей
    for num_data in numbers:
        try:
            cursor.execute("""
            INSERT INTO numbers (
                chunk_id, value, unit, raw_text, 
                is_range_start, is_range_end
            ) 
            VALUES (?, ?, ?, ?, ?, ?)
            """, (
                chunk_db_id,
                float(num_data.get("value", 0)),
                num_data.get("unit", ""),
                num_data.get("raw_text", ""),
                1 if num_data.get("is_range_start") else 0,
                1 if num_data.get("is_range_end") else 0
            ))
        except Exception as e:
            logger.error(f"Ошибка при сохранении числового значения: {str(e)}")
            logger.debug(f"Данные: {num_data}")
            logger.debug(f"Трассировка: {traceback.format_exc()}")


def _save_links(cursor: sqlite3.Cursor, links: List[str], chunk_db_id: int) -> None:
    """
    Сохраняет исходящие ссылки чанка.
    
    Args:
        cursor: Курсор SQLite
        links: Список ссылок
        chunk_db_id: ID чанка в базе данных
    """
    # Удаление старых записей для этого чанка
    cursor.execute("DELETE FROM links WHERE chunk_id = ?", (chunk_db_id,))
    
    # Вставка новых записей
    for link in links:
        try:
            cursor.execute("""
            INSERT INTO links (chunk_id, target_article_url) 
            VALUES (?, ?)
            """, (chunk_db_id, link))
        except Exception as e:
            logger.error(f"Ошибка при сохранении ссылки: {str(e)}")
            logger.debug(f"Ссылка: {link}")
            logger.debug(f"Трассировка: {traceback.format_exc()}")


def _save_images(cursor: sqlite3.Cursor, images: List[Dict[str, Any]], chunk_db_id: int) -> None:
    """
    Сохраняет информацию об изображениях чанка.
    
    Args:
        cursor: Курсор SQLite
        images: Список данных изображений
        chunk_db_id: ID чанка в базе данных
    """
    # Удаление старых записей для этого чанка
    cursor.execute("DELETE FROM images WHERE chunk_id = ?", (chunk_db_id,))
    
    # Вставка новых записей
    for img in images:
        # Сериализация метаданных
        metadata_json = None
        if "metadata" in img and img["metadata"]:
            metadata_json = json.dumps(img["metadata"], ensure_ascii=False)
            
        try:
            cursor.execute("""
            INSERT INTO images (chunk_id, filename, original_src, alt, caption, metadata) 
            VALUES (?, ?, ?, ?, ?, ?)
            """, (
                chunk_db_id,
                img.get("filename", ""),
                img.get("original_src", ""),
                img.get("alt", ""),
                img.get("caption", ""),
                metadata_json
            ))
        except Exception as e:
            logger.error(f"Ошибка при сохранении изображения: {str(e)}")
            logger.debug(f"Данные изображения: {img}")
            logger.debug(f"Трассировка: {traceback.format_exc()}")


def _save_categories(cursor: sqlite3.Cursor, article_id: Optional[int], 
                    chunk_id: Optional[int], paragraph_id: Optional[int],
                    categories: List[str], source: str, confidence: float = 1.0) -> None:
    """
    Сохраняет категории/метки для различных сущностей.
    
    Args:
        cursor: Курсор SQLite
        article_id: ID статьи (или None)
        chunk_id: ID чанка (или None)
        paragraph_id: ID параграфа (или None)
        categories: Список категорий/меток
        source: Источник метки (breadcrumb, keyword, ml, gpt)
        confidence: Уверенность в метке (0-1)
    """
    for category in categories:
        # Разделение на тип категории и значение
        if ":" in category:
            category_name, category_value = category.split(":", 1)
        else:
            category_name = "label"
            category_value = category
        
        try:
            cursor.execute("""
            INSERT INTO categorized_entities (
                article_id, chunk_id, paragraph_id, 
                category_name, category_value, source, confidence
            ) 
            VALUES (?, ?, ?, ?, ?, ?, ?)
            """, (
                article_id,
                chunk_id,
                paragraph_id,
                category_name,
                category_value,
                source,
                confidence
            ))
        except Exception as e:
            logger.error(f"Ошибка при сохранении категории: {str(e)}")
            logger.debug(f"Категория: {category}")
            logger.debug(f"Трассировка: {traceback.format_exc()}")


def _save_tech_specs(cursor: sqlite3.Cursor, chunk: Dict[str, Any], chunk_db_id: int) -> None:
    """
    Сохраняет технические характеристики из метаданных чанка.
    
    Args:
        cursor: Курсор SQLite
        chunk: Данные чанка
        chunk_db_id: ID чанка в базе данных
    """
    # Проверка наличия технических характеристик в метаданных
    if "metadata" not in chunk or "tech_specs" not in chunk["metadata"]:
        return
    
    tech_specs = chunk["metadata"]["tech_specs"]
    
    # Удаление старых записей для этого чанка
    cursor.execute("DELETE FROM tech_specs WHERE chunk_id = ?", (chunk_db_id,))
    
    # Вставка новых записей
    for spec in tech_specs:
        # Проверка на наличие числовых значений
        has_numeric = 0
        numeric_value = None
        unit = None
        
        if "numeric_values" in spec and spec["numeric_values"]:
            has_numeric = 1
            # Берем первое числовое значение как основное
            first_numeric = spec["numeric_values"][0]
            numeric_value = first_numeric.get("numeric_value", first_numeric.get("value"))
            unit = first_numeric.get("unit", "")
        
        try:
            cursor.execute("""
            INSERT INTO tech_specs (
                chunk_id, name, value, has_numeric_value, numeric_value, unit
            ) 
            VALUES (?, ?, ?, ?, ?, ?)
            """, (
                chunk_db_id,
                spec.get("name", ""),
                spec.get("value", ""),
                has_numeric,
                numeric_value,
                unit
            ))
        except Exception as e:
            logger.error(f"Ошибка при сохранении технической характеристики: {str(e)}")
            logger.debug(f"Характеристика: {spec}")
            logger.debug(f"Трассировка: {traceback.format_exc()}")


def get_chunk_text(chunk: Dict[str, Any]) -> str:
    """
    Вспомогательная функция для получения текстового содержимого чанка.
    
    Args:
        chunk: Данные чанка
        
    Returns:
        str: Текстовое содержимое
    """
    text_parts = []
    
    try:
        # Добавляем заголовок
        if chunk.get("title"):
            text_parts.append(chunk["title"])
        
        # Добавляем текст из всех элементов
        for element in chunk.get("elements", []):
            element_type = element.get("type", "")
            
            if element_type == "paragraph":
                text_parts.append(element.get("text", ""))
                
            elif element_type == "list":
                for item in element.get("items", []):
                    text_parts.append(f"• {item}")
                    
            elif element_type == "table":
                # Добавляем заголовки таблицы
                if element.get("headers"):
                    text_parts.append(" | ".join(element["headers"]))
                # Добавляем строки таблицы
                for row in element.get("rows", []):
                    text_parts.append(" | ".join(row))
                    
            elif element_type in ["figure", "image"] and "caption" in element:
                if element.get("caption"):
                    text_parts.append(element["caption"])
    
    except Exception as e:
        logger.error(f"Ошибка при получении текста чанка: {str(e)}")
        logger.debug(f"Трассировка: {traceback.format_exc()}")
    
    return "\n".join(text_parts)


def get_article_by_url(db_path: Union[str, Path], url: str) -> Optional[Dict[str, Any]]:
    """
    Получает информацию о статье по URL.
    
    Args:
        db_path: Путь к файлу базы данных
        url: URL статьи
        
    Returns:
        Optional[Dict[str, Any]]: Данные статьи или None, если не найдена
    """
    db_path = Path(db_path)
    
    conn = None
    try:
        conn = sqlite3.connect(db_path)
        conn.row_factory = sqlite3.Row
        cursor = conn.cursor()
        
        cursor.execute("SELECT * FROM articles WHERE url = ?", (url,))
        article_row = cursor.fetchone()
        
        if not article_row:
            return None
        
        # Преобразование row в словарь
        article = dict(article_row)
        
        # Десериализация хлебных крошек
        if article["breadcrumbs"]:
            article["breadcrumbs"] = json.loads(article["breadcrumbs"])
        else:
            article["breadcrumbs"] = []
        
        # Десериализация метаданных
        if article["metadata"]:
            article["metadata"] = json.loads(article["metadata"])
        
        return article
        
    except sqlite3.Error as e:
        logger.error(f"Ошибка при получении статьи по URL {url}: {str(e)}")
        logger.debug(f"Трассировка: {traceback.format_exc()}")
        return None
    
    finally:
        if conn:
            conn.close()


def get_chunks_by_article_id(db_path: Union[str, Path], article_id: int) -> List[Dict[str, Any]]:
    """
    Получает все чанки статьи.
    
    Args:
        db_path: Путь к файлу базы данных
        article_id: ID статьи
        
    Returns:
        List[Dict[str, Any]]: Список чанков статьи
    """
    db_path = Path(db_path)
    
    conn = None
    try:
        conn = sqlite3.connect(db_path)
        conn.row_factory = sqlite3.Row
        cursor = conn.cursor()
        
        # Получение всех чанков статьи
        cursor.execute("""
        SELECT * FROM chunks 
        WHERE article_id = ? 
        ORDER BY id
        """, (article_id,))
        
        chunks = []
        for row in cursor.fetchall():
            chunk = dict(row)
            
            # Десериализация меток
            if chunk["labels"]:
                chunk["labels"] = json.loads(chunk["labels"])
            else:
                chunk["labels"] = []
            
            # Десериализация метаданных
            if chunk["metadata"]:
                chunk["metadata"] = json.loads(chunk["metadata"])
            
            # Получение параграфов
            cursor.execute("""
            SELECT * FROM paragraphs 
            WHERE chunk_id = ?
            ORDER BY id
            """, (chunk["id"],))
            chunk["paragraphs"] = []
            chunk["paragraph_labels"] = {}
            
            for p_row in cursor.fetchall():
                paragraph = dict(p_row)
                
                # Десериализация меток параграфа
                if paragraph["labels"]:
                    paragraph["labels"] = json.loads(paragraph["labels"])
                    # Сохранение в словаре для удобства доступа
                    chunk["paragraph_labels"][paragraph["paragraph_id"]] = paragraph["labels"]
                else:
                    paragraph["labels"] = []
                    chunk["paragraph_labels"][paragraph["paragraph_id"]] = []
                
                # Десериализация метаданных
                if paragraph["metadata"]:
                    paragraph["metadata"] = json.loads(paragraph["metadata"])
                
                chunk["paragraphs"].append(paragraph)
            
            # Получение числовых данных
            cursor.execute("""
            SELECT * FROM numbers 
            WHERE chunk_id = ?
            """, (chunk["id"],))
            chunk["numbers"] = [dict(row) for row in cursor.fetchall()]
            
            # Получение ссылок
            cursor.execute("""
            SELECT target_article_url FROM links 
            WHERE chunk_id = ?
            """, (chunk["id"],))
            chunk["outgoing_links"] = [row[0] for row in cursor.fetchall()]
            
            # Получение изображений
            cursor.execute("""
            SELECT * FROM images 
            WHERE chunk_id = ?
            """, (chunk["id"],))
            chunk["images"] = []
            for img_row in cursor.fetchall():
                img = dict(img_row)
                if img["metadata"]:
                    img["metadata"] = json.loads(img["metadata"])
                chunk["images"].append(img)
            
            # Получение технических характеристик
            cursor.execute("""
            SELECT * FROM tech_specs
            WHERE chunk_id = ?
            """, (chunk["id"],))
            chunk["tech_specs"] = [dict(row) for row in cursor.fetchall()]
            
            chunks.append(chunk)
        
        return chunks
        
    except sqlite3.Error as e:
        logger.error(f"Ошибка при получении чанков статьи {article_id}: {str(e)}")
        logger.debug(f"Трассировка: {traceback.format_exc()}")
        return []
    
    finally:
        if conn:
            conn.close()


def search_by_number(db_path: Union[str, Path], value: float, unit: str, 
                    tolerance: float = 0.01) -> List[Dict[str, Any]]:
    """
    Ищет чанки, содержащие числовое значение с заданной единицей измерения.
    
    Args:
        db_path: Путь к файлу базы данных
        value: Числовое значение для поиска
        unit: Единица измерения
        tolerance: Допустимое отклонение от значения (относительное)
        
    Returns:
        List[Dict[str, Any]]: Список найденных чанков
    """
    db_path = Path(db_path)
    
    conn = None
    try:
        conn = sqlite3.connect(db_path)
        conn.row_factory = sqlite3.Row
        cursor = conn.cursor()
        
        # Вычисление допустимых границ
        min_value = value * (1 - tolerance)
        max_value = value * (1 + tolerance)
        
        # Поиск чанков с заданным числовым значением
        cursor.execute("""
        SELECT c.*, a.title as article_title, a.url as article_url
        FROM chunks c
        JOIN articles a ON c.article_id = a.id
        JOIN numbers n ON n.chunk_id = c.id
        WHERE n.unit = ? AND n.value BETWEEN ? AND ?
        GROUP BY c.id
        """, (unit, min_value, max_value))
        
        results = []
        for row in cursor.fetchall():
            chunk = dict(row)
            
            # Десериализация меток
            if chunk["labels"]:
                chunk["labels"] = json.loads(chunk["labels"])
            else:
                chunk["labels"] = []
            
            # Десериализация метаданных
            if chunk["metadata"]:
                chunk["metadata"] = json.loads(chunk["metadata"])
            
            results.append(chunk)
        
        # Также ищем в технических характеристиках
        cursor.execute("""
        SELECT c.*, a.title as article_title, a.url as article_url
        FROM chunks c
        JOIN articles a ON c.article_id = a.id
        JOIN tech_specs ts ON ts.chunk_id = c.id
        WHERE ts.unit = ? AND ts.has_numeric_value = 1 
              AND ts.numeric_value BETWEEN ? AND ?
        GROUP BY c.id
        """, (unit, min_value, max_value))
        
        for row in cursor.fetchall():
            chunk = dict(row)
            chunk_id = chunk["id"]
            
            # Проверяем, что такой чанк еще не добавлен
            if not any(c["id"] == chunk_id for c in results):
                # Десериализация меток
                if chunk["labels"]:
                    chunk["labels"] = json.loads(chunk["labels"])
                else:
                    chunk["labels"] = []
                
                # Десериализация метаданных
                if chunk["metadata"]:
                    chunk["metadata"] = json.loads(chunk["metadata"])
                
                results.append(chunk)
        
        return results
        
    except sqlite3.Error as e:
        logger.error(f"Ошибка при поиске по числовому значению {value} {unit}: {str(e)}")
        logger.debug(f"Трассировка: {traceback.format_exc()}")
        return []
    
    finally:
        if conn:
            conn.close()


def search_by_label(db_path: Union[str, Path], label: str, 
                  entity_type: str = "chunk") -> List[Dict[str, Any]]:
    """
    Ищет сущности (чанки или параграфы) с заданной меткой.
    
    Args:
        db_path: Путь к файлу базы данных
        label: Метка для поиска
        entity_type: Тип сущности для поиска ("chunk" или "paragraph")
        
    Returns:
        List[Dict[str, Any]]: Список найденных сущностей
    """
    db_path = Path(db_path)
    
    conn = None
    try:
        conn = sqlite3.connect(db_path)
        conn.row_factory = sqlite3.Row
        cursor = conn.cursor()
        
        results = []
        
        if entity_type == "chunk":
            # Поиск чанков с заданной меткой
            cursor.execute("""
            SELECT c.*, a.title as article_title, a.url as article_url
            FROM chunks c
            JOIN articles a ON c.article_id = a.id
            WHERE c.labels LIKE ?
            """, (f'%"{label}"%',))
            
            for row in cursor.fetchall():
                chunk = dict(row)
                
                # Десериализация меток
                if chunk["labels"]:
                    chunk["labels"] = json.loads(chunk["labels"])
                else:
                    chunk["labels"] = []
                
                # Десериализация метаданных
                if chunk["metadata"]:
                    chunk["metadata"] = json.loads(chunk["metadata"])
                
                # Проверяем, что метка действительно присутствует (для исключения ложных срабатываний LIKE)
                if label in chunk["labels"]:
                    results.append(chunk)
        
        elif entity_type == "paragraph":
            # Поиск параграфов с заданной меткой
            cursor.execute("""
            SELECT p.*, c.title as chunk_title, a.title as article_title, a.url as article_url
            FROM paragraphs p
            JOIN chunks c ON p.chunk_id = c.id
            JOIN articles a ON c.article_id = a.id
            WHERE p.labels LIKE ?
            """, (f'%"{label}"%',))
            
            for row in cursor.fetchall():
                paragraph = dict(row)
                
                # Десериализация меток
                if paragraph["labels"]:
                    paragraph["labels"] = json.loads(paragraph["labels"])
                else:
                    paragraph["labels"] = []
                
                # Десериализация метаданных
                if paragraph["metadata"]:
                    paragraph["metadata"] = json.loads(paragraph["metadata"])
                
                # Проверяем, что метка действительно присутствует
                if label in paragraph["labels"]:
                    results.append(paragraph)
        
        # Дополнительный поиск по categorized_entities
        entity_id_field = f"{entity_type}_id"
        cursor.execute(f"""
        SELECT e.*, a.title as article_title, a.url as article_url
        FROM categorized_entities e
        JOIN articles a ON e.article_id = a.id
        WHERE e.category_value = ? AND e.{entity_id_field} IS NOT NULL
        """, (label,))
        
        for row in cursor.fetchall():
            entity_id = row[entity_id_field]
            
            # Получение полных данных о сущности
            if entity_type == "chunk":
                cursor.execute("""
                SELECT c.*, a.title as article_title, a.url as article_url
                FROM chunks c
                JOIN articles a ON c.article_id = a.id
                WHERE c.id = ?
                """, (entity_id,))
                entity_row = cursor.fetchone()
                
                if entity_row:
                    entity = dict(entity_row)
                    
                    # Десериализация меток
                    if entity["labels"]:
                        entity["labels"] = json.loads(entity["labels"])
                    else:
                        entity["labels"] = []
                    
                    # Десериализация метаданных
                    if entity["metadata"]:
                        entity["metadata"] = json.loads(entity["metadata"])
                    
                    # Избегаем дублирования в результатах
                    if not any(r["id"] == entity["id"] for r in results):
                        results.append(entity)
            
            elif entity_type == "paragraph":
                cursor.execute("""
                SELECT p.*, c.title as chunk_title, a.title as article_title, a.url as article_url
                FROM paragraphs p
                JOIN chunks c ON p.chunk_id = c.id
                JOIN articles a ON c.article_id = a.id
                WHERE p.id = ?
                """, (entity_id,))
                entity_row = cursor.fetchone()
                
                if entity_row:
                    entity = dict(entity_row)
                    
                    # Десериализация меток
                    if entity["labels"]:
                        entity["labels"] = json.loads(entity["labels"])
                    else:
                        entity["labels"] = []
                    
                    # Десериализация метаданных
                    if entity["metadata"]:
                        entity["metadata"] = json.loads(entity["metadata"])
                    
                    # Избегаем дублирования в результатах
                    if not any(r["id"] == entity["id"] for r in results):
                        results.append(entity)
        
        return results
        
    except sqlite3.Error as e:
        logger.error(f"Ошибка при поиске по метке {label}: {str(e)}")
        logger.debug(f"Трассировка: {traceback.format_exc()}")
        return []
    
    finally:
        if conn:
            conn.close()


def search_text(db_path: Union[str, Path], query: str, 
               entity_type: str = "both") -> List[Dict[str, Any]]:
    """
    Выполняет полнотекстовый поиск по чанкам и/или параграфам.
    
    Args:
        db_path: Путь к файлу базы данных
        query: Поисковый запрос
        entity_type: Тип сущности для поиска ("chunk", "paragraph" или "both")
        
    Returns:
        List[Dict[str, Any]]: Список найденных сущностей
    """
    db_path = Path(db_path)
    
    conn = None
    try:
        conn = sqlite3.connect(db_path)
        conn.row_factory = sqlite3.Row
        cursor = conn.cursor()
        
        # Подготовка запроса для FTS
        search_term = f'%{query}%'
        results = []
        
        if entity_type in ["chunk", "both"]:
            # Поиск в чанках
            cursor.execute("""
            SELECT c.*, a.title as article_title, a.url as article_url
            FROM chunks c
            JOIN articles a ON c.article_id = a.id
            WHERE c.content LIKE ? OR c.title LIKE ?
            ORDER BY 
                CASE 
                    WHEN c.title LIKE ? THEN 1
                    ELSE 2
                END,
                c.id
            """, (search_term, search_term, search_term))
            
            for row in cursor.fetchall():
                chunk = dict(row)
                
                # Десериализация меток
                if chunk["labels"]:
                    chunk["labels"] = json.loads(chunk["labels"])
                else:
                    chunk["labels"] = []
                
                # Десериализация метаданных
                if chunk["metadata"]:
                    chunk["metadata"] = json.loads(chunk["metadata"])
                
                chunk["entity_type"] = "chunk"
                results.append(chunk)
        
        if entity_type in ["paragraph", "both"]:
            # Поиск в параграфах
            cursor.execute("""
            SELECT p.*, c.title as chunk_title, a.title as article_title, a.url as article_url
            FROM paragraphs p
            JOIN chunks c ON p.chunk_id = c.id
            JOIN articles a ON c.article_id = a.id
            WHERE p.text LIKE ?
            ORDER BY p.id
            """, (search_term,))
            
            for row in cursor.fetchall():
                paragraph = dict(row)
                
                # Десериализация меток
                if paragraph["labels"]:
                    paragraph["labels"] = json.loads(paragraph["labels"])
                else:
                    paragraph["labels"] = []
                
                # Десериализация метаданных
                if paragraph["metadata"]:
                    paragraph["metadata"] = json.loads(paragraph["metadata"])
                
                paragraph["entity_type"] = "paragraph"
                results.append(paragraph)
        
        # Добавляем поиск в технических характеристиках
        if entity_type in ["chunk", "both"]:
            cursor.execute("""
            SELECT c.*, a.title as article_title, a.url as article_url
            FROM chunks c
            JOIN articles a ON c.article_id = a.id
            JOIN tech_specs ts ON ts.chunk_id = c.id
            WHERE ts.name LIKE ? OR ts.value LIKE ?
            GROUP BY c.id
            """, (search_term, search_term))
            
            for row in cursor.fetchall():
                chunk = dict(row)
                chunk_id = chunk["id"]
                
                # Проверка на дубликаты
                if any(r.get("id") == chunk_id and r.get("entity_type") == "chunk" for r in results):
                    continue
                
                # Десериализация меток
                if chunk["labels"]:
                    chunk["labels"] = json.loads(chunk["labels"])
                else:
                    chunk["labels"] = []
                
                # Десериализация метаданных
                if chunk["metadata"]:
                    chunk["metadata"] = json.loads(chunk["metadata"])
                
                chunk["entity_type"] = "chunk"
                results.append(chunk)
        
        return results
        
    except sqlite3.Error as e:
        logger.error(f"Ошибка при полнотекстовом поиске '{query}': {str(e)}")
        logger.debug(f"Трассировка: {traceback.format_exc()}")
        return []
    
    finally:
        if conn:
            conn.close()


def search_technical_parameters(db_path: Union[str, Path], params: Dict[str, Any]) -> List[Dict[str, Any]]:
    """
    Поиск чанков по техническим параметрам для гибридной RAG-системы.
    
    Args:
        db_path: Путь к файлу базы данных
        params: Словарь параметров для поиска вида {"имя_параметра": значение}
        
    Returns:
        List[Dict[str, Any]]: Список найденных чанков
    """
    db_path = Path(db_path)
    
    conn = None
    try:
        conn = sqlite3.connect(db_path)
        conn.row_factory = sqlite3.Row
        cursor = conn.cursor()
        
        # Создание условий поиска
        conditions = []
        query_params = []
        
        for param_name, param_value in params.items():
            # Если значение - число, ищем в числовых полях
            if isinstance(param_value, (int, float)):
                conditions.append("(ts.name LIKE ? AND ts.has_numeric_value = 1 AND ts.numeric_value = ?)")
                query_params.extend([f"%{param_name}%", param_value])
            # Если значение - строка, ищем в текстовых полях
            elif isinstance(param_value, str):
                conditions.append("(ts.name LIKE ? AND ts.value LIKE ?)")
                query_params.extend([f"%{param_name}%", f"%{param_value}%"])
        
        if not conditions:
            return []
        
        # Объединение условий через OR
        where_clause = " OR ".join(conditions)
        
        # Выполнение запроса
        cursor.execute(f"""
        SELECT c.*, a.title as article_title, a.url as article_url
        FROM chunks c
        JOIN articles a ON c.article_id = a.id
        JOIN tech_specs ts ON ts.chunk_id = c.id
        WHERE {where_clause}
        GROUP BY c.id
        """, query_params)
        
        results = []
        for row in cursor.fetchall():
            chunk = dict(row)
            
            # Десериализация меток
            if chunk["labels"]:
                chunk["labels"] = json.loads(chunk["labels"])
            else:
                chunk["labels"] = []
            
            # Десериализация метаданных
            if chunk["metadata"]:
                chunk["metadata"] = json.loads(chunk["metadata"])
            
            # Получаем связанные технические характеристики
            cursor.execute("""
            SELECT * FROM tech_specs
            WHERE chunk_id = ?
            """, (chunk["id"],))
            chunk["tech_specs"] = [dict(row) for row in cursor.fetchall()]
            
            results.append(chunk)
        
        return results
        
    except sqlite3.Error as e:
        logger.error(f"Ошибка при поиске по техническим параметрам: {str(e)}")
        logger.debug(f"Трассировка: {traceback.format_exc()}")
        return []
    
    finally:
        if conn:
            conn.close()