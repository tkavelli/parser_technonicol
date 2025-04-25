"""
Общие утилиты для работы с текстом и извлечения данных.
Централизует функции обработки текста, используемые в разных модулях пайплайна.
"""
import re
import json
import logging
import unicodedata
import traceback
from typing import List, Dict, Any, Optional, Union, Pattern, Set, Tuple
from bs4 import BeautifulSoup

# Настройка логирования
logger = logging.getLogger(__name__)

# Константы для регулярных выражений
# Паттерн для поиска чисел с единицами измерения
NUMERIC_PATTERN = r'(\d+(?:[,.]\d+)?(?:\s*[-–—]\s*\d+(?:[,.]\d+)?)?)\s*([а-яА-Яa-zA-Z°%]+)'
# Паттерн для поиска диапазонов
RANGE_PATTERN = r'(\d+(?:[,.]\d+)?)\s*[-–—]\s*(\d+(?:[,.]\d+)?)'
# Компилированные регулярные выражения для повышения производительности
NUMERIC_REGEX = re.compile(NUMERIC_PATTERN)
RANGE_REGEX = re.compile(RANGE_PATTERN)


def clean_text(text: str) -> str:
    """
    Очищает текст от лишних пробелов, переносов строк и т.д.
    
    Args:
        text: Исходный текст
        
    Returns:
        str: Очищенный текст
    """
    # Замена нескольких переносов строк на один
    text = re.sub(r'\n+', '\n', text)
    
    # Замена множественных пробелов на один
    text = re.sub(r' +', ' ', text)
    
    # Удаление пробелов в начале и конце строк
    lines = text.split('\n')
    lines = [line.strip() for line in lines]
    text = '\n'.join(lines)
    
    return text.strip()


def extract_html_text(html: str) -> str:
    """
    Извлекает текст из HTML без использования BeautifulSoup.
    
    Args:
        html: HTML-строка
        
    Returns:
        str: Извлеченный текст
    """
    # Удаление HTML-тегов
    text = re.sub(r'<[^>]+>', ' ', html)
    
    # Обработка HTML-сущностей
    text = re.sub(r'&nbsp;', ' ', text)
    text = re.sub(r'&lt;', '<', text)
    text = re.sub(r'&gt;', '>', text)
    text = re.sub(r'&amp;', '&', text)
    text = re.sub(r'&quot;', '"', text)
    text = re.sub(r'&apos;', "'", text)
    
    # Удаление лишних пробелов
    text = re.sub(r'\s+', ' ', text)
    
    return text.strip()


def get_chunk_text(chunk: Dict[str, Any]) -> str:
    """
    Извлекает полный текст из чанка, объединяя его элементы.
    
    Args:
        chunk: Данные чанка
        
    Returns:
        str: Текстовое содержимое чанка
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
                    
            elif element_type == "blockquote":
                text_parts.append(element.get("text", ""))
                
            elif element_type in ["figure", "image"] and "caption" in element:
                if element.get("caption"):
                    text_parts.append(element["caption"])
    
    except Exception as e:
        logger.error(f"Ошибка при получении текста чанка: {str(e)}")
        logger.debug(f"Трассировка: {traceback.format_exc()}")
    
    return "\n".join(text_parts)


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
        logger.error(f"Ошибка при извлечении заголовков из чанка {chunk.get('id', 'unknown')}: {str(e)}")
        logger.debug(f"Трассировка: {traceback.format_exc()}")
    
    return headings


def flatten_chunks(chunks: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """
    Преобразует иерархическую структуру чанков в плоский список.
    
    Args:
        chunks: Список иерархических чанков
        
    Returns:
        List[Dict[str, Any]]: Плоский список всех чанков
    """
    flat_list = []
    
    def _flatten(chunk_list):
        for chunk in chunk_list:
            # Создаем копию чанка без детей для плоского списка
            flat_chunk = chunk.copy()
            children = flat_chunk.pop("children", [])
            flat_list.append(flat_chunk)
            
            # Рекурсивно обрабатываем детей
            if children:
                _flatten(children)
    
    try:
        _flatten(chunks)
    except Exception as e:
        logger.error(f"Ошибка при преобразовании чанков в плоский список: {str(e)}")
        logger.debug(f"Трассировка: {traceback.format_exc()}")
    
    return flat_list


def flatten_paragraphs(chunks: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """
    Извлекает все параграфы из иерархии чанков в плоский список.
    
    Args:
        chunks: Список иерархических чанков
        
    Returns:
        List[Dict[str, Any]]: Плоский список всех параграфов
    """
    all_paragraphs = []
    
    try:
        # Получаем плоский список чанков
        flat_chunks = flatten_chunks(chunks)
        
        # Извлекаем параграфы из каждого чанка
        for chunk in flat_chunks:
            if "paragraphs" in chunk:
                all_paragraphs.extend(chunk["paragraphs"])
                
    except Exception as e:
        logger.error(f"Ошибка при извлечении параграфов: {str(e)}")
        logger.debug(f"Трассировка: {traceback.format_exc()}")
    
    return all_paragraphs


def extract_numbers(text: str) -> List[Dict[str, Any]]:
    """
    Извлекает числовые значения с единицами измерения из текста.
    
    Args:
        text: Текст для обработки
        
    Returns:
        List[Dict[str, Any]]: Список найденных числовых данных с единицами измерения
    """
    numbers = []
    
    try:
        # Поиск всех совпадений в тексте с помощью скомпилированного регулярного выражения
        matches = NUMERIC_REGEX.finditer(text)
        
        for match in matches:
            value_str = match.group(1)
            unit = match.group(2).lower()
            raw_text = match.group(0)
            
            # Проверка и нормализация единиц измерения
            unit = normalize_unit(unit)
            
            # Обработка диапазонов чисел
            range_match = RANGE_REGEX.search(value_str)
            if range_match:
                # Разбиваем на два значения
                start_value = range_match.group(1).replace(',', '.')
                end_value = range_match.group(2).replace(',', '.')
                
                # Проверка валидности
                if is_valid_number(start_value) and is_valid_number(end_value):
                    # Добавляем начало диапазона
                    numbers.append({
                        "value": start_value,
                        "unit": unit,
                        "raw_text": raw_text,
                        "is_range_start": True,
                        "is_range_end": False,
                        "numeric_value": float(start_value),
                        "normalized_unit": unit
                    })
                    
                    # Добавляем конец диапазона
                    numbers.append({
                        "value": end_value,
                        "unit": unit,
                        "raw_text": raw_text,
                        "is_range_start": False,
                        "is_range_end": True,
                        "numeric_value": float(end_value),
                        "normalized_unit": unit
                    })
            else:
                # Обычное числовое значение
                value = value_str.strip().replace(',', '.')
                
                # Проверка валидности
                if is_valid_number(value):
                    numbers.append({
                        "value": value,
                        "unit": unit,
                        "raw_text": raw_text,
                        "is_range_start": False,
                        "is_range_end": False,
                        "numeric_value": float(value),
                        "normalized_unit": unit
                    })
        
    except Exception as e:
        logger.error(f"Ошибка при извлечении числовых значений: {str(e)}")
        logger.debug(f"Трассировка: {traceback.format_exc()}")
    
    return numbers


def is_valid_number(value: str) -> bool:
    """
    Проверяет, является ли строка корректным числом.
    
    Args:
        value: Строка для проверки
        
    Returns:
        bool: True, если строка является корректным числом
    """
    try:
        float(value)
        return True
    except ValueError:
        return False


def normalize_unit(unit: str) -> str:
    """
    Нормализует единицы измерения для унификации поиска.
    
    Args:
        unit: Исходная единица измерения
        
    Returns:
        str: Нормализованная единица измерения
    """
    # Словарь для нормализации единиц (можно расширить)
    unit_mapping = {
        "мм": "мм",
        "см": "см",
        "м": "м",
        "м2": "м²",
        "м²": "м²",
        "м3": "м³",
        "м³": "м³",
        "мм2": "мм²",
        "мм²": "мм²",
        "кг": "кг",
        "г": "г",
        "т": "т",
        "л": "л",
        "мл": "мл",
        "с": "с",
        "ч": "ч",
        "мин": "мин",
        "°с": "°C",
        "°c": "°C",
        "%": "%"
    }
    
    normalized = unit.lower().replace(" ", "")
    return unit_mapping.get(normalized, unit)


def sanitize_filename(name: str) -> str:
    """
    Очищает строку для использования в имени файла.
    
    Args:
        name: Исходная строка
        
    Returns:
        str: Очищенная строка
    """
    # Нормализация Unicode и преобразование к ASCII (для совместимости)
    name = unicodedata.normalize('NFKD', name)
    name = name.encode('ascii', 'ignore').decode('ascii')
    
    # Замена специальных символов на дефисы
    name = re.sub(r'[^\w\s-]', '', name.lower())
    
    # Замена пробелов на дефисы
    name = re.sub(r'[\s]+', '-', name.strip())
    
    # Удаление повторяющихся дефисов
    name = re.sub(r'[-]+', '-', name)
    
    return name


def extract_numerical_metadata(numbers: List[Dict[str, Any]]) -> Dict[str, Any]:
    """
    Извлекает метаданные о числовых значениях для гибридной RAG-системы.
    
    Args:
        numbers: Список числовых данных
        
    Returns:
        Dict[str, Any]: Метаданные о числовых значениях
    """
    result = {
        "has_values": bool(numbers),
        "count": len(numbers),
        "units": set(),
        "min_values": {},
        "max_values": {},
        "value_ranges": {}
    }
    
    try:
        # Если нет числовых данных, возвращаем базовую структуру
        if not numbers:
            result["units"] = []
            return result
        
        # Сбор уникальных единиц измерения
        units = set()
        min_values = {}
        max_values = {}
        
        for num in numbers:
            unit = num["unit"]
            units.add(unit)
            value = float(num["numeric_value"])
            
            # Обновление минимальных/максимальных значений по единицам измерения
            if unit not in min_values or value < min_values[unit]:
                min_values[unit] = value
            
            if unit not in max_values or value > max_values[unit]:
                max_values[unit] = value
        
        # Формирование диапазонов значений
        value_ranges = {}
        for unit in units:
            if unit in min_values and unit in max_values:
                value_ranges[unit] = {
                    "min": min_values[unit],
                    "max": max_values[unit]
                }
        
        # Обновление результата
        result["units"] = list(units)
        result["min_values"] = min_values
        result["max_values"] = max_values
        result["value_ranges"] = value_ranges
        
        return result
        
    except Exception as e:
        logger.error(f"Ошибка при извлечении метаданных о числовых значениях: {str(e)}")
        logger.debug(f"Трассировка: {traceback.format_exc()}")
        # В случае ошибки возвращаем базовую структуру
        return {"has_values": bool(numbers), "count": len(numbers), "units": []}


def split_text_to_paragraphs(text: str) -> List[str]:
    """
    Разбивает текст на абзацы.
    
    Args:
        text: Исходный текст
        
    Returns:
        List[str]: Список абзацев
    """
    # Разбивка по двойным переносам строк (пустые строки между абзацами)
    paragraphs = re.split(r'\n\s*\n', text)
    
    # Удаление пустых абзацев и очистка
    paragraphs = [p.strip() for p in paragraphs if p.strip()]
    
    return paragraphs


def log_error(logger, message: str, error, context: Dict[str, Any] = None):
    """
    Логирует ошибку с контекстом для улучшения отладки.
    
    Args:
        logger: Объект логгера
        message: Сообщение об ошибке
        error: Объект исключения
        context: Словарь с контекстной информацией
    """
    logger.error(f"{message}: {str(error)}")
    
    if context:
        try:
            context_str = json.dumps(context, ensure_ascii=False, default=str)
            logger.error(f"Контекст: {context_str}")
        except Exception:
            logger.error(f"Контекст: {context} (не удалось сериализовать)")
    
    logger.debug(f"Трассировка: {traceback.format_exc()}")


def truncate_text(text: str, max_length: int = 100, suffix: str = '...') -> str:
    """
    Обрезает текст до указанной длины, добавляя суффикс если текст был обрезан.
    
    Args:
        text: Исходный текст
        max_length: Максимальная длина результата (включая суффикс)
        suffix: Суффикс для обрезанного текста
        
    Returns:
        str: Обрезанный текст
    """
    if len(text) <= max_length:
        return text
    
    # Обрезка с учетом длины суффикса
    return text[:max_length - len(suffix)] + suffix


def extract_breadcrumbs_from_url(url: str) -> List[str]:
    """
    Извлекает "хлебные крошки" из URL, основываясь на структуре URL tn.ru.
    
    Args:
        url: URL статьи
        
    Returns:
        List[str]: Список хлебных крошек
    """
    try:
        from urllib.parse import urlparse, unquote
        
        # Разбор URL
        parsed = urlparse(url)
        path = unquote(parsed.path)
        
        # Проверка, что URL соответствует структуре базы знаний TechnoNICOL
        if '/knowledge-base/' in path or '/materialy/' in path:
            # Разбиваем путь на компоненты
            path_parts = [p for p in path.split('/') if p and p not in ['knowledge-base', 'materialy']]
            
            # Преобразуем slug в читаемый текст
            breadcrumbs = []
            for part in path_parts:
                # Замена дефисов на пробелы и капитализация
                readable = part.replace('-', ' ').title()
                breadcrumbs.append(readable)
                
            return breadcrumbs
            
        return []
        
    except Exception as e:
        logger.error(f"Ошибка при извлечении хлебных крошек из URL {url}: {str(e)}")
        logger.debug(f"Трассировка: {traceback.format_exc()}")
        return []
