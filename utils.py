"""
Вспомогательные функции и утилиты для пайплайна обработки статей.
"""
import os
import re
import unicodedata
import logging
from datetime import datetime
from pathlib import Path
from typing import Dict, Any, Optional, Union, List
from urllib.parse import urlparse


def setup_logging(log_level=logging.INFO, log_file=None) -> logging.Logger:
    """
    Настраивает логирование для всего приложения.
    
    Args:
        log_level: Уровень логирования
        log_file: Путь к файлу лога (если None, вывод только в консоль)
        
    Returns:
        logging.Logger: Настроенный логгер
    """
    # Создание корневого логгера
    logger = logging.getLogger()
    logger.setLevel(log_level)
    
    # Очистка существующих обработчиков
    for handler in logger.handlers[:]:
        logger.removeHandler(handler)
    
    # Создание обработчика консоли
    console_handler = logging.StreamHandler()
    console_handler.setLevel(log_level)
    
    # Формат лога
    formatter = logging.Formatter(
        '%(asctime)s [%(levelname)s] %(name)s: %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    console_handler.setFormatter(formatter)
    
    # Добавление обработчика консоли
    logger.addHandler(console_handler)
    
    # Создание файлового обработчика, если указан файл лога
    if log_file:
        # Создание директории для лога, если не существует
        log_dir = os.path.dirname(log_file)
        if log_dir:
            os.makedirs(log_dir, exist_ok=True)
        
        # Создание обработчика файла
        file_handler = logging.FileHandler(log_file, encoding='utf-8')
        file_handler.setLevel(log_level)
        file_handler.setFormatter(formatter)
        
        # Добавление обработчика файла
        logger.addHandler(file_handler)
    
    return logger


def get_article_slug(url: str) -> str:
    """
    Генерирует слаг из URL статьи.
    
    Args:
        url: URL статьи
        
    Returns:
        str: Слаг для статьи
    """
    # Извлечение последнего компонента пути URL
    parsed_url = urlparse(url)
    path_parts = parsed_url.path.strip('/').split('/')
    
    # Если путь пустой, используем домен
    if not path_parts or not path_parts[-1]:
        return sanitize_filename(parsed_url.netloc)
    
    # Удаление расширения файла, если есть (.html, .php и т.д.)
    last_part = path_parts[-1]
    last_part = re.sub(r'\.(html|php|aspx|jsp)$', '', last_part)
    
    # Если последняя часть пути слишком длинная, обрезаем
    if len(last_part) > 50:
        last_part = last_part[:50]
    
    # Очистка от специальных символов и нормализация
    return sanitize_filename(last_part)


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


def ensure_dir(path: Union[str, Path]) -> Path:
    """
    Убеждается, что директория существует, создавая её при необходимости.
    
    Args:
        path: Путь к директории
        
    Returns:
        Path: Объект Path для директории
    """
    path = Path(path)
    path.mkdir(parents=True, exist_ok=True)
    return path


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


def extract_domain(url: str) -> str:
    """
    Извлекает домен из URL.
    
    Args:
        url: URL для обработки
        
    Returns:
        str: Домен из URL
    """
    parsed_url = urlparse(url)
    domain = parsed_url.netloc
    
    # Удаление www. если есть
    if domain.startswith('www.'):
        domain = domain[4:]
    
    return domain


def format_file_size(size_bytes: int) -> str:
    """
    Форматирует размер файла в человекочитаемую строку.
    
    Args:
        size_bytes: Размер в байтах
        
    Returns:
        str: Форматированная строка размера
    """
    if size_bytes < 1024:
        return f"{size_bytes} Б"
    elif size_bytes < 1024 * 1024:
        return f"{size_bytes / 1024:.1f} КБ"
    elif size_bytes < 1024 * 1024 * 1024:
        return f"{size_bytes / (1024 * 1024):.1f} МБ"
    else:
        return f"{size_bytes / (1024 * 1024 * 1024):.1f} ГБ"


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


def merge_dictionaries(dict1: Dict[str, Any], dict2: Dict[str, Any],
                       overwrite: bool = True) -> Dict[str, Any]:
    """
    Объединяет два словаря.
    
    Args:
        dict1: Первый словарь
        dict2: Второй словарь
        overwrite: Перезаписывать ли существующие ключи
        
    Returns:
        Dict[str, Any]: Объединенный словарь
    """
    result = dict1.copy()
    
    for key, value in dict2.items():
        if key not in result or overwrite:
            result[key] = value
    
    return result


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


def is_valid_url(url: str) -> bool:
    """
    Проверяет, является ли строка корректным URL.
    
    Args:
        url: Строка для проверки
        
    Returns:
        bool: True, если URL корректен
    """
    try:
        result = urlparse(url)
        return all([result.scheme, result.netloc])
    except Exception:
        return False


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


def get_current_timestamp() -> str:
    """
    Возвращает текущую дату и время в ISO формате.
    
    Returns:
        str: Текущая дата и время в формате ISO
    """
    return datetime.now().strftime("%Y-%m-%dT%H:%M:%S")