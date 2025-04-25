"""
Вспомогательные функции и утилиты для пайплайна обработки статей.
"""
import os
import logging
import time
from datetime import datetime
from pathlib import Path
from typing import Dict, Any, Optional, Union, List
from urllib.parse import urlparse

from text_utils import sanitize_filename


def setup_logging(log_level=logging.INFO, log_dir=None) -> logging.Logger:
    """
    Настраивает логирование для всего приложения.
    
    Args:
        log_level: Уровень логирования
        log_dir: Директория для сохранения логов (если None, логи только в консоль)
        
    Returns:
        logging.Logger: Настроенный логгер
    """
    # Создание корневого логгера
    logger = logging.getLogger()
    logger.setLevel(log_level)
    
    # Очистка существующих обработчиков
    for handler in logger.handlers[:]:
        logger.removeHandler(handler)
    
    # Формат лога для обоих обработчиков
    formatter = logging.Formatter(
        '%(asctime)s [%(levelname)s] %(name)s: %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    
    # Создание обработчика консоли
    console_handler = logging.StreamHandler()
    console_handler.setLevel(log_level)
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)
    
    # Создание файлового обработчика, если указана директория логов
    if log_dir:
        # Создание директории для логов, если не существует
        log_dir = Path(log_dir)
        log_dir.mkdir(parents=True, exist_ok=True)
        
        # Имя файла лога с датой и временем
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        log_file = log_dir / f"pipeline_{timestamp}.log"
        
        # Создание обработчика файла
        file_handler = logging.FileHandler(log_file, encoding='utf-8')
        file_handler.setLevel(log_level)
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)
        
        logger.info(f"Логирование в файл: {log_file}")
    
    logger.info(f"Логирование инициализировано (уровень: {logging.getLevelName(log_level)})")
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
    
    return sanitize_filename(last_part)


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


def get_current_timestamp() -> str:
    """
    Возвращает текущую дату и время в ISO формате.
    
    Returns:
        str: Текущая дата и время в формате ISO
    """
    return datetime.now().strftime("%Y-%m-%dT%H:%M:%S")


def create_timer():
    """
    Создает простой таймер для измерения времени выполнения операций.
    
    Returns:
        tuple: (start_timer, get_elapsed) - функции для старта таймера и получения прошедшего времени
    """
    timer_data = {"start": None}
    
    def start_timer():
        timer_data["start"] = time.time()
    
    def get_elapsed() -> float:
        if timer_data["start"] is None:
            return 0
        return time.time() - timer_data["start"]
    
    return start_timer, get_elapsed


# Импорт для обратной совместимости
import re  # Для регулярных выражений в get_article_slug
