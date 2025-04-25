"""
Модуль для разбиения контента на иерархические смысловые чанки.
Группирует элементы контента в логические разделы, основываясь на заголовках.
Поддерживает двухуровневую структуру: чанки (по заголовкам) и параграфы.
"""
import logging
import traceback
import uuid
import time
from typing import List, Dict, Any, Optional, Tuple, Set

from text_utils import get_chunk_text, log_error

# Настройка логирования
logger = logging.getLogger(__name__)


def create_chunks(content_elements: List[Dict[str, Any]], article_slug: str,
                 breadcrumbs: Optional[List[str]] = None) -> List[Dict[str, Any]]:
    """
    Создает иерархическую структуру чанков из последовательного списка элементов.
    
    Args:
        content_elements: Список элементов контента, полученных из парсера
        article_slug: Слаг статьи (используется для генерации ID)
        breadcrumbs: "Хлебные крошки" статьи (категории из навигации)
        
    Returns:
        List[Dict[str, Any]]: Список чанков верхнего уровня с вложенными подчанками
    """
    logger.info(f"Начало разбиения на чанки для {article_slug}")
    start_time = time.time()
    
    try:
        # Проверка на пустые входные данные
        if not content_elements:
            logger.warning(f"Получен пустой список элементов контента для {article_slug}")
            return []
        
        # Инициализация идентификатора чанка
        chunk_counter = 0
        
        # Инициализация результата
        chunks = []
        
        # Стек текущих чанков по уровням заголовков (0 - корневой уровень без заголовка)
        current_chunks = {0: None}
        current_level = 0
        
        # Временный чанк для контента перед первым заголовком (введение)
        intro_content = []
        
        # Обработка всех элементов
        for element in content_elements:
            # Если элемент - заголовок, создаем новый чанк
            if element.get("type") == "heading":
                heading_level = element.get("level", 1)
                heading_text = element.get("text", "")
                
                # Некорректные уровни заголовков корректируем
                if heading_level < 1 or heading_level > 6:
                    logger.warning(f"Некорректный уровень заголовка {heading_level} для '{heading_text}'")
                    heading_level = max(1, min(6, heading_level))
                
                # Если это первый заголовок и у нас есть введение, создаем чанк введения
                if intro_content and not chunks:
                    chunk_counter += 1
                    intro_chunk = _create_chunk(
                        article_slug,
                        chunk_counter,
                        "Введение",
                        0,
                        intro_content,
                        breadcrumbs
                    )
                    chunks.append(intro_chunk)
                    intro_content = []
                
                # Создаем новый чанк для текущего заголовка
                chunk_counter += 1
                new_chunk = _create_chunk(
                    article_slug,
                    chunk_counter,
                    heading_text,
                    heading_level,
                    [],
                    breadcrumbs
                )
                
                # Находим родительский чанк для вложения
                parent_level = _find_parent_level(heading_level, current_chunks)
                
                # Добавляем новый чанк к родительскому или в корневой список
                if parent_level > 0 and current_chunks[parent_level]:
                    current_chunks[parent_level]["children"].append(new_chunk)
                else:
                    chunks.append(new_chunk)
                
                # Обновляем текущий чанк для этого уровня и удаляем все уровни ниже
                current_chunks[heading_level] = new_chunk
                current_chunks = _clean_lower_levels(current_chunks, heading_level)
                
                current_level = heading_level
                
            else:
                # Не заголовок - добавляем элемент к текущему чанку или во введение
                if current_level > 0 and current_level in current_chunks:
                    current_chunks[current_level]["elements"].append(element)
                    
                    # Если элемент - параграф, добавляем его в список параграфов чанка
                    if element.get("type") == "paragraph":
                        _add_paragraph_to_chunk(current_chunks[current_level], element)
                else:
                    # Если мы еще не встретили заголовок, добавляем к введению
                    intro_content.append(element)
        
        # Обработка оставшегося контента
        chunks = _process_remaining_content(
            chunks, intro_content, article_slug, chunk_counter, breadcrumbs
        )
        
        elapsed_time = time.time() - start_time
        logger.info(f"Создано {chunk_counter} чанков для {article_slug} за {elapsed_time:.2f} сек")
        
        # Проверка на связность структуры
        if not _validate_chunk_structure(chunks):
            logger.warning(f"Структура чанков не прошла валидацию для {article_slug}")
        
        # Дополнительная постобработка чанков для гибридной RAG-системы
        chunks = _enrich_chunks_for_rag(chunks)
        
        # Подсчет общего количества параграфов
        from text_utils import flatten_chunks
        flat_chunks = flatten_chunks(chunks)
        total_paragraphs = sum(len(chunk.get("paragraphs", [])) for chunk in flat_chunks)
        logger.info(f"Всего создано {len(flat_chunks)} чанков и {total_paragraphs} параграфов")
        
        return chunks
        
    except Exception as e:
        elapsed_time = time.time() - start_time
        context = {
            "article_slug": article_slug,
            "content_length": len(content_elements) if content_elements else 0,
            "process_time": f"{elapsed_time:.2f} сек"
        }
        log_error(logger, f"Ошибка при создании чанков", e, context)
        return []


def _create_chunk(article_slug: str, 
                 chunk_counter: int, 
                 title: str, 
                 level: int, 
                 elements: List[Dict[str, Any]],
                 breadcrumbs: Optional[List[str]] = None) -> Dict[str, Any]:
    """
    Создает новый чанк с заданными параметрами.
    
    Args:
        article_slug: Слаг статьи
        chunk_counter: Счетчик чанков
        title: Заголовок чанка
        level: Уровень вложенности (0-6)
        elements: Список элементов контента
        breadcrumbs: Хлебные крошки статьи
        
    Returns:
        Dict[str, Any]: Структура чанка
    """
    chunk_id = f"{article_slug}_chunk_{chunk_counter}"
    
    return {
        "id": chunk_id,
        "title": title,
        "level": level,
        "elements": elements.copy(),
        "paragraphs": _extract_paragraphs(elements, article_slug, chunk_counter, title),
        "children": [],
        "labels": [],
        "paragraph_labels": {},  # Словарь {paragraph_id: [labels]}
        "numbers": [],
        "outgoing_links": [],
        "metadata": {
            "breadcrumbs": breadcrumbs or [],
            "chunk_number": chunk_counter,
            "content_types": _get_content_types(elements)
        }
    }


def _add_paragraph_to_chunk(chunk: Dict[str, Any], paragraph_element: Dict[str, Any]) -> None:
    """
    Добавляет параграф в список параграфов чанка.
    
    Args:
        chunk: Чанк, к которому добавляется параграф
        paragraph_element: Элемент с типом paragraph
    """
    paragraph_id = f"{chunk['id']}_p_{len(chunk['paragraphs']) + 1}"
    
    paragraph = {
        "id": paragraph_id,
        "text": paragraph_element["text"],
        "type": "paragraph",
        "parent_chunk_id": chunk["id"],
        "parent_chunk_title": chunk["title"],
        "metadata": paragraph_element.get("metadata", {})
    }
    
    # Если в элементе есть HTML, сохраняем его
    if "html" in paragraph_element:
        paragraph["html"] = paragraph_element["html"]
        
    chunk["paragraphs"].append(paragraph)


def _find_parent_level(heading_level: int, current_chunks: Dict[int, Any]) -> int:
    """
    Находит родительский уровень для заголовка указанного уровня.
    
    Args:
        heading_level: Уровень текущего заголовка
        current_chunks: Словарь текущих чанков по уровням
        
    Returns:
        int: Уровень родительского чанка
    """
    parent_level = heading_level - 1
    while parent_level > 0 and parent_level not in current_chunks:
        parent_level -= 1
    
    return parent_level


def _clean_lower_levels(current_chunks: Dict[int, Any], current_level: int) -> Dict[int, Any]:
    """
    Удаляет уровни, которые ниже текущего.
    
    Args:
        current_chunks: Словарь текущих чанков по уровням
        current_level: Текущий уровень заголовка
        
    Returns:
        Dict[int, Any]: Обновленный словарь чанков по уровням
    """
    result = {k: v for k, v in current_chunks.items() if k <= current_level}
    return result


def _extract_paragraphs(elements: List[Dict[str, Any]], article_slug: str, 
                       chunk_counter: int, chunk_title: str) -> List[Dict[str, Any]]:
    """
    Извлекает параграфы из списка элементов.
    
    Args:
        elements: Список элементов
        article_slug: Слаг статьи
        chunk_counter: Счетчик чанков
        chunk_title: Заголовок чанка
        
    Returns:
        List[Dict[str, Any]]: Список параграфов
    """
    paragraphs = []
    chunk_id = f"{article_slug}_chunk_{chunk_counter}"
    
    for i, element in enumerate(elements):
        if element.get("type") == "paragraph":
            paragraph_id = f"{chunk_id}_p_{i+1}"
            paragraph = {
                "id": paragraph_id,
                "text": element["text"],
                "type": "paragraph",
                "parent_chunk_id": chunk_id,
                "parent_chunk_title": chunk_title,
                "metadata": element.get("metadata", {})
            }
            
            # Если в элементе есть HTML, сохраняем его
            if "html" in element:
                paragraph["html"] = element["html"]
                
            paragraphs.append(paragraph)
    
    return paragraphs


def _process_remaining_content(chunks: List[Dict[str, Any]], 
                              intro_content: List[Dict[str, Any]], 
                              article_slug: str, 
                              chunk_counter: int,
                              breadcrumbs: Optional[List[str]] = None) -> List[Dict[str, Any]]:
    """
    Обрабатывает оставшийся контент, не привязанный к заголовкам.
    
    Args:
        chunks: Текущий список чанков
        intro_content: Контент введения
        article_slug: Слаг статьи
        chunk_counter: Текущий счетчик чанков
        breadcrumbs: Хлебные крошки статьи
        
    Returns:
        List[Dict[str, Any]]: Обновленный список чанков
    """
    # Если у нас есть контент перед первым заголовком и мы не создали для него чанк
    if intro_content and not chunks:
        chunk_counter += 1
        intro_chunk = _create_chunk(
            article_slug, 
            chunk_counter, 
            "Введение", 
            0, 
            intro_content,
            breadcrumbs
        )
        chunks.append(intro_chunk)
    
    # Если после обработки всех элементов чанков не создано, создаем один общий чанк
    if not chunks and intro_content:
        chunk_counter += 1
        full_chunk = _create_chunk(
            article_slug, 
            chunk_counter, 
            "Полный контент", 
            0, 
            intro_content,
            breadcrumbs
        )
        chunks.append(full_chunk)
    
    return chunks


def _get_content_types(elements: List[Dict[str, Any]]) -> List[str]:
    """
    Возвращает список типов контента, содержащихся в элементах.
    
    Args:
        elements: Список элементов контента
        
    Returns:
        List[str]: Список уникальных типов контента
    """
    content_types: Set[str] = set()
    
    for element in elements:
        element_type = element.get("type")
        if element_type:
            content_types.add(element_type)
    
    return list(content_types)


def get_chunk_headings(chunk: Dict[str, Any], include_subchunks: bool = True) -> List[str]:
    """
    Извлекает заголовки из чанка и его подчанков.
    
    Args:
        chunk: Чанк для извлечения заголовков
        include_subchunks: Включать ли заголовки подчанков
        
    Returns:
        List[str]: Список заголовков
    """
    headings = []
    
    try:
        # Добавляем заголовок чанка
        if chunk.get("title"):
            headings.append(chunk["title"])
        
        # Добавляем заголовки из элементов контента
        for element in chunk.get("elements", []):
            if element.get("type") == "heading":
                headings.append(element.get("text", ""))
        
        # Рекурсивно добавляем заголовки из подчанков
        if include_subchunks:
            for child in chunk.get("children", []):
                child_headings = get_chunk_headings(child, include_subchunks)
                headings.extend(child_headings)
                
    except Exception as e:
        context = {"chunk_id": chunk.get('id', 'unknown')}
        log_error(logger, f"Ошибка при извлечении заголовков из чанка", e, context)
    
    return headings


def _validate_chunk_structure(chunks: List[Dict[str, Any]]) -> bool:
    """
    Проверяет корректность структуры чанков.
    
    Args:
        chunks: Список чанков для проверки
        
    Returns:
        bool: True, если структура корректна, иначе False
    """
    # Проверка на пустой список
    if not chunks:
        logger.warning("Пустой список чанков")
        return False
    
    # Проверка каждого чанка
    chunk_ids = set()
    
    def _check_chunk(chunk):
        # Проверка наличия обязательных полей
        required_fields = ["id", "title", "elements", "children", "paragraphs"]
        for field in required_fields:
            if field not in chunk:
                logger.warning(f"В чанке отсутствует обязательное поле: {field}")
                return False
        
        # Проверка уникальности ID
        if chunk["id"] in chunk_ids:
            logger.warning(f"Дублирующийся ID чанка: {chunk['id']}")
            return False
        
        chunk_ids.add(chunk["id"])
        
        # Рекурсивная проверка детей
        for child in chunk["children"]:
            if not _check_chunk(child):
                return False
        
        return True
    
    # Проверка всех корневых чанков
    try:
        for chunk in chunks:
            if not _check_chunk(chunk):
                return False
        
        return True
        
    except Exception as e:
        log_error(logger, f"Ошибка при валидации структуры чанков", e)
        return False


def _enrich_chunks_for_rag(chunks: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """
    Обогащает чанки дополнительными метаданными для гибридной RAG-системы.
    
    Args:
        chunks: Список чанков
        
    Returns:
        List[Dict[str, Any]]: Обогащенные чанки
    """
    try:
        # Получаем плоский список чанков для обработки
        from text_utils import flatten_chunks
        flat_chunks = flatten_chunks(chunks)
        
        for chunk in flat_chunks:
            # Инициализация метаданных, если их нет
            if "metadata" not in chunk:
                chunk["metadata"] = {}
            
            # Добавляем количество элементов, параграфов и дочерних чанков
            chunk["metadata"]["element_count"] = len(chunk.get("elements", []))
            chunk["metadata"]["paragraph_count"] = len(chunk.get("paragraphs", []))
            chunk["metadata"]["child_count"] = len(chunk.get("children", []))
            
            # Извлечение заголовков для улучшения поиска
            chunk["metadata"]["headings"] = get_chunk_headings(chunk, include_subchunks=False)
            
            # Анализ содержимого для определения характеристик
            chunk_text = get_chunk_text(chunk)
            chunk["metadata"]["text_length"] = len(chunk_text)
            
            # Извлечение потенциальных числовых параметров для SQL/JSON поиска
            chunk["metadata"]["has_numbers"] = bool(chunk.get("numbers"))
            
            # Дополнительные метаданные для связи с другими чанками
            chunk["metadata"]["outgoing_links_count"] = len(chunk.get("outgoing_links", []))
            
        # Обновляем ссылки на метаданные родитель-потомок
        _update_parent_child_references(chunks)
        
    except Exception as e:
        log_error(logger, f"Ошибка при обогащении чанков метаданными", e)
    
    return chunks


def _update_parent_child_references(chunks: List[Dict[str, Any]], parent_id: Optional[str] = None) -> None:
    """
    Обновляет ссылки родитель-потомок в метаданных чанков.
    
    Args:
        chunks: Список чанков
        parent_id: ID родительского чанка
    """
    for chunk in chunks:
        if parent_id:
            if "metadata" not in chunk:
                chunk["metadata"] = {}
            chunk["metadata"]["parent_chunk_id"] = parent_id
        
        # Рекурсивно обновляем ссылки для дочерних чанков
        if chunk.get("children"):
            _update_parent_child_references(chunk["children"], chunk["id"])