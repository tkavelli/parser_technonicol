"""
Модуль для извлечения структурированных данных из чанков.
Обрабатывает числа, единицы измерения, ссылки и добавляет контекстные метки.
Поддерживает двухуровневую классификацию контента (чанки и параграфы).
"""
import re
import os
import logging
import traceback
import time
from pathlib import Path
from typing import List, Dict, Any, Set, Tuple, Optional, Union, Pattern
from urllib.parse import urljoin, urlparse
from bs4 import BeautifulSoup

# Импорт модулей пайплайна
from fetcher import fetch_image
from text_utils import (
    get_chunk_text, flatten_chunks, flatten_paragraphs,
    extract_numbers, extract_numerical_metadata, log_error
)
from classifier import get_classifier, classify_article, ContentClassifier

# Настройка логирования
logger = logging.getLogger(__name__)


def extract_data(chunks: List[Dict[str, Any]], article_url: str, 
                images_dir: Union[str, Path], use_ml_classification: bool = False,
                use_gpt: bool = False) -> List[Dict[str, Any]]:
    """
    Извлекает структурированные данные из чанков статьи.
    
    Args:
        chunks: Список чанков статьи
        article_url: URL статьи (для обработки относительных ссылок)
        images_dir: Директория для сохранения изображений
        use_ml_classification: Использовать ли ML-модели для классификации
        use_gpt: Использовать ли GPT для классификации
        
    Returns:
        List[Dict[str, Any]]: Обогащенные чанки со структурированными данными
    """
    logger.info(f"Начало извлечения данных для {article_url}")
    start_time = time.time()
    
    try:
        # Обеспечение наличия директории для изображений
        images_dir = Path(images_dir)
        os.makedirs(images_dir, exist_ok=True)
        
        # Классификация чанков и параграфов
        if use_ml_classification or use_gpt:
            # Создание классификатора с расширенными возможностями
            classifier = get_classifier(
                use_bert=use_ml_classification,
                use_gpt=use_gpt
            )
            # Классификация всех чанков и параграфов
            chunks, paragraph_labels = classify_article(chunks, classifier)
            
            logger.info(f"Выполнена расширенная классификация (ML: {use_ml_classification}, GPT: {use_gpt})")
        else:
            # Простая классификация на основе ключевых слов
            classifier = get_classifier()
            chunks, paragraph_labels = classify_article(chunks, classifier)
            
            logger.info("Выполнена базовая классификация (ключевые слова)")
        
        # Обработка каждого чанка
        flat_chunks = flatten_chunks(chunks)
        
        processed_chunks = 0
        extracted_numbers = 0
        extracted_links = 0
        processed_images = 0
        
        for chunk in flat_chunks:
            chunk_id = chunk.get("id", "unknown")
            try:
                logger.debug(f"Обработка чанка {chunk_id}")
                
                # 1. Извлечение числовых данных
                chunk_text = get_chunk_text(chunk)
                chunk["numbers"] = extract_numbers(chunk_text)
                extracted_numbers += len(chunk["numbers"])
                
                # 2. Обработка внутренних ссылок
                chunk["outgoing_links"] = extract_links(chunk, article_url)
                extracted_links += len(chunk["outgoing_links"])
                
                # 3. Загрузка изображений
                image_count = process_images(chunk, article_url, images_dir)
                processed_images += image_count
                
                # 4. Дополнительное обогащение для гибридной RAG-системы
                enrich_chunk_for_rag(chunk)
                
                processed_chunks += 1
                
            except Exception as e:
                context = {"chunk_id": chunk_id}
                log_error(logger, f"Ошибка при извлечении данных из чанка", e, context)
        
        elapsed_time = time.time() - start_time
        logger.info(f"Завершено извлечение данных для {article_url} за {elapsed_time:.2f} сек: "
                   f"обработано {processed_chunks} чанков, извлечено {extracted_numbers} числовых данных, "
                   f"{extracted_links} ссылок, обработано {processed_images} изображений")
        
        return chunks
        
    except Exception as e:
        elapsed_time = time.time() - start_time
        context = {
            "url": article_url,
            "processing_time": f"{elapsed_time:.2f} сек",
            "chunks_count": len(chunks) if chunks else 0
        }
        log_error(logger, "Ошибка при извлечении данных", e, context)
        return chunks


def extract_links(chunk: Dict[str, Any], base_url: str) -> List[str]:
    """
    Извлекает исходящие ссылки из элементов чанка.
    
    Args:
        chunk: Чанк для обработки
        base_url: Базовый URL статьи
        
    Returns:
        List[str]: Список исходящих ссылок
    """
    links = []
    
    try:
        # Обработка элементов чанка
        for element in chunk.get("elements", []):
            # Извлечение ссылок из HTML-тегов внутри текста для абзацев
            if element.get("type") == "paragraph" and "html" in element:
                # Парсинг HTML-фрагмента
                soup = BeautifulSoup(element["html"], "html.parser")
                
                # Извлечение всех ссылок
                for a_tag in soup.find_all('a', href=True):
                    href = a_tag['href']
                    
                    # Преобразование относительных ссылок в абсолютные
                    if not href.startswith(('http://', 'https://')):
                        href = urljoin(base_url, href)
                    
                    # Фильтрация внутренних ссылок (на тот же домен)
                    if is_internal_link(href, base_url):
                        # Нормализация URL для избежания дубликатов
                        href = normalize_url(href)
                        if href and href not in links:
                            links.append(href)
        
        # Также проверяем параграфы чанка, если они содержат HTML
        for paragraph in chunk.get("paragraphs", []):
            if "html" in paragraph:
                soup = BeautifulSoup(paragraph["html"], "html.parser")
                
                for a_tag in soup.find_all('a', href=True):
                    href = a_tag['href']
                    
                    if not href.startswith(('http://', 'https://')):
                        href = urljoin(base_url, href)
                    
                    if is_internal_link(href, base_url) and href not in links:
                        href = normalize_url(href)
                        if href and href not in links:
                            links.append(href)
        
    except Exception as e:
        context = {"chunk_id": chunk.get('id', 'unknown')}
        log_error(logger, "Ошибка при извлечении ссылок из чанка", e, context)
    
    return links


def is_internal_link(url: str, base_url: str) -> bool:
    """
    Проверяет, является ли ссылка внутренней (на тот же домен).
    
    Args:
        url: URL для проверки
        base_url: Базовый URL сайта
        
    Returns:
        bool: True, если ссылка внутренняя
    """
    try:
        parsed_url = urlparse(url)
        parsed_base = urlparse(base_url)
        
        return parsed_url.netloc == parsed_base.netloc
    except Exception:
        return False


def normalize_url(url: str) -> str:
    """
    Нормализует URL для предотвращения дубликатов.
    
    Args:
        url: URL для нормализации
        
    Returns:
        str: Нормализованный URL
    """
    try:
        # Удаление якорей и параметров запроса
        parsed = urlparse(url)
        normalized = f"{parsed.scheme}://{parsed.netloc}{parsed.path}"
        
        # Удаление завершающего слеша, если он есть
        if normalized.endswith('/'):
            normalized = normalized[:-1]
        
        return normalized
    except Exception:
        return url


def process_images(chunk: Dict[str, Any], article_url: str, images_dir: Union[str, Path]) -> int:
    """
    Обрабатывает и загружает изображения, связанные с чанком.
    
    Args:
        chunk: Чанк для обработки
        article_url: URL статьи
        images_dir: Директория для сохранения изображений
        
    Returns:
        int: Количество обработанных изображений
    """
    images_dir = Path(images_dir)
    image_refs = []
    processed_count = 0
    
    try:
        for i, element in enumerate(chunk.get("elements", [])):
            if element.get("type") in ["image", "figure"]:
                src = element.get("src", "")
                if src:
                    # Генерация имени файла для сохранения
                    image_filename = generate_image_filename(src, i, chunk.get("id", "unknown"))
                    image_path = images_dir / image_filename
                    
                    # Загрузка изображения
                    if fetch_image(src, image_path, article_url):
                        processed_count += 1
                        
                        # Добавление информации об изображении
                        image_info = {
                            "filename": image_filename,
                            "original_src": src,
                            "alt": element.get("alt", ""),
                            "caption": element.get("caption", ""),
                            "path": str(image_path),
                            "local_path": str(image_path)
                        }
                        
                        # Добавляем метаданные для RAG
                        if "metadata" in element:
                            image_info["metadata"] = element["metadata"]
                        
                        image_refs.append(image_info)
                        
                        # Обновление элемента с локальным путем
                        element["local_path"] = str(image_path)
                        
                        # Если в элементе есть метаданные, добавляем информацию о загрузке
                        if "metadata" in element:
                            element["metadata"]["image_loaded"] = True
                            element["metadata"]["local_path"] = str(image_path)
        
        # Сохранение ссылок на изображения в чанке
        chunk["images"] = image_refs
        
        # Добавление количества изображений в метаданные
        if "metadata" in chunk:
            chunk["metadata"]["image_count"] = len(image_refs)
        
    except Exception as e:
        context = {"chunk_id": chunk.get('id', 'unknown')}
        log_error(logger, "Ошибка при обработке изображений для чанка", e, context)
    
    return processed_count


def generate_image_filename(src: str, index: int, chunk_id: str) -> str:
    """
    Генерирует корректное имя файла для изображения.
    
    Args:
        src: URL источника изображения
        index: Индекс изображения в чанке
        chunk_id: ID чанка
        
    Returns:
        str: Сгенерированное имя файла
    """
    try:
        # Извлечение имени файла из URL
        parsed_url = urlparse(src)
        filename = os.path.basename(parsed_url.path)
        
        # Проверка валидности имени
        if not filename or '.' not in filename:
            # Если имя невалидное, генерируем новое с расширением по умолчанию
            filename = f"img_{index}.jpg"
        
        # Ограничение длины имени файла
        if len(filename) > 200:
            name, ext = os.path.splitext(filename)
            filename = f"{name[:190]}{ext}"
        
        # Добавление ID чанка для уникальности
        sanitized_chunk_id = chunk_id.replace("/", "_")
        prefixed_filename = f"{sanitized_chunk_id}_{index}_{filename}"
        
        # Удаление недопустимых символов
        sanitized_filename = re.sub(r'[<>:"/\\|?*]', '_', prefixed_filename)
        
        return sanitized_filename
        
    except Exception as e:
        logger.error(f"Ошибка при генерации имени файла для изображения: {str(e)}")
        # В случае ошибки возвращаем безопасное имя файла
        return f"image_{chunk_id}_{index}.jpg"


def enrich_chunk_for_rag(chunk: Dict[str, Any]) -> None:
    """
    Обогащает чанк дополнительными метаданными для гибридной RAG-системы.
    
    Args:
        chunk: Чанк для обогащения
    """
    try:
        # Инициализация метаданных, если их нет
        if "metadata" not in chunk:
            chunk["metadata"] = {}
        
        # Добавление информации о числовых данных
        numerical_metadata = extract_numerical_metadata(chunk.get("numbers", []))
        chunk["metadata"]["numerical_data"] = numerical_metadata
        
        # Добавление информации о ссылках
        chunk["metadata"]["outgoing_links_count"] = len(chunk.get("outgoing_links", []))
        
        # Извлечение технических характеристик из таблиц
        tech_specs = extract_tech_specs_from_tables(chunk.get("elements", []))
        if tech_specs:
            chunk["metadata"]["tech_specs"] = tech_specs
        
        # Обработка параграфов
        for paragraph in chunk.get("paragraphs", []):
            enrich_paragraph_for_rag(paragraph)
            
    except Exception as e:
        context = {"chunk_id": chunk.get('id', 'unknown')}
        log_error(logger, "Ошибка при обогащении чанка для RAG", e, context)


def enrich_paragraph_for_rag(paragraph: Dict[str, Any]) -> None:
    """
    Обогащает параграф дополнительными метаданными для гибридной RAG-системы.
    
    Args:
        paragraph: Параграф для обогащения
    """
    try:
        # Инициализация метаданных, если их нет
        if "metadata" not in paragraph:
            paragraph["metadata"] = {}
        
        # Извлечение числовых данных из текста параграфа
        text = paragraph.get("text", "")
        numbers = extract_numbers(text)
        
        # Если есть числовые данные, добавляем их в метаданные
        if numbers:
            paragraph["metadata"]["has_numeric_data"] = True
            paragraph["metadata"]["numeric_values"] = [
                {
                    "value": n["numeric_value"],
                    "unit": n["normalized_unit"],
                    "raw": n["raw_text"]
                }
                for n in numbers
            ]
        else:
            paragraph["metadata"]["has_numeric_data"] = False
            
    except Exception as e:
        context = {"paragraph_id": paragraph.get('id', 'unknown')}
        log_error(logger, "Ошибка при обогащении параграфа для RAG", e, context)


def extract_tech_specs_from_tables(elements: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """
    Извлекает технические характеристики из таблиц.
    
    Args:
        elements: Список элементов чанка
        
    Returns:
        List[Dict[str, Any]]: Список извлеченных технических характеристик
    """
    specs = []
    
    try:
        for element in elements:
            if element.get("type") == "table":
                headers = element.get("headers", [])
                rows = element.get("rows", [])
                
                # Проверка на наличие заголовков и строк
                if not headers or not rows:
                    continue
                
                # Определяем, является ли таблица таблицей характеристик
                is_specs_table = is_technical_specs_table(headers)
                
                if is_specs_table:
                    # Извлекаем характеристики из таблицы
                    for row in rows:
                        if len(row) >= 2:  # Должны быть хотя бы название и значение
                            name = row[0]
                            value = row[1]
                            
                            # Проверка на числовые значения с единицами измерения
                            numeric_values = extract_numbers(value)
                            
                            spec_entry = {
                                "name": name,
                                "value": value
                            }
                            
                            # Если есть числовые значения, добавляем их структурированно
                            if numeric_values:
                                spec_entry["numeric_values"] = numeric_values
                            
                            specs.append(spec_entry)
        
        return specs
        
    except Exception as e:
        log_error(logger, "Ошибка при извлечении технических характеристик из таблиц", e)
        return []


def is_technical_specs_table(headers: List[str]) -> bool:
    """
    Определяет, является ли таблица таблицей технических характеристик.
    
    Args:
        headers: Заголовки таблицы
        
    Returns:
        bool: True, если таблица содержит технические характеристики
    """
    try:
        # Ключевые слова, указывающие на таблицу характеристик
        specs_keywords = [
            "характеристика", "параметр", "свойство", "показатель", 
            "значение", "величина", "specification", "parameter"
        ]
        
        # Проверка заголовков на наличие ключевых слов
        headers_text = " ".join(headers).lower()
        for keyword in specs_keywords:
            if keyword in headers_text:
                return True
        
        # Проверка типичной структуры таблицы характеристик
        if len(headers) >= 2:
            first_col = headers[0].lower()
            second_col = headers[1].lower()
            
            name_keywords = ["наименование", "параметр", "характеристика", "показатель"]
            value_keywords = ["значение", "величина", "показатель", "value"]
            
            has_name_col = any(keyword in first_col for keyword in name_keywords)
            has_value_col = any(keyword in second_col for keyword in value_keywords)
            
            if has_name_col and has_value_col:
                return True
        
        return False
        
    except Exception as e:
        log_error(logger, "Ошибка при определении таблицы технических характеристик", e)
        return False