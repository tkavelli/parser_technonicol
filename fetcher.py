"""
Модуль для загрузки HTML-контента статей и изображений.
Поддерживает повторные попытки, кеширование и проверку статус-кодов HTTP.
"""
import os
import time
import random
import logging
import traceback
from urllib.parse import urlparse, urljoin
from pathlib import Path
from typing import Optional, Union, List, Dict, Any, Tuple

import requests
from requests.exceptions import RequestException, ConnectionError, Timeout, HTTPError

from text_utils import log_error, sanitize_filename

# Настройка логирования
logger = logging.getLogger(__name__)

# Константы
MAX_RETRIES = 3
RETRY_DELAY = 2
USER_AGENTS = [
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36",
    "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/605.1.15 (KHTML, like Gecko) Version/14.1.1 Safari/605.1.15",
    "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/92.0.4515.107 Safari/537.36",
]


def fetch_article(url: str, save_path: Union[str, Path], force_download: bool = False) -> str:
    """
    Загружает HTML-контент статьи и сохраняет его локально.
    
    Args:
        url: URL статьи для загрузки
        save_path: Путь для сохранения HTML-файла
        force_download: Принудительная загрузка, даже если файл уже существует
        
    Returns:
        str: Содержимое HTML
        
    Raises:
        requests.RequestException: Если не удалось загрузить статью после всех попыток
    """
    save_path = Path(save_path)
    
    # Проверка наличия уже загруженного файла
    if not force_download and save_path.exists():
        try:
            file_size = save_path.stat().st_size
            logger.info(f"Используем локальную копию: {save_path} (размер: {file_size} байт)")
            with open(save_path, 'r', encoding='utf-8') as f:
                html_content = f.read()
                
            # Проверка минимальной валидности
            if _validate_html_content(html_content):
                return html_content
            else:
                logger.warning(f"Локальный файл {save_path} не проходит валидацию. Загружаем заново.")
        except Exception as e:
            context = {"path": str(save_path)}
            log_error(logger, f"Ошибка при чтении локального файла", e, context)
            # Если произошла ошибка при чтении файла, попробуем загрузить заново
    
    # Обеспечение наличия директории для сохранения
    try:
        os.makedirs(save_path.parent, exist_ok=True)
    except Exception as e:
        context = {"directory": str(save_path.parent)}
        log_error(logger, f"Не удалось создать директорию", e, context)
        raise
    
    # Попытки загрузки с повторами при ошибках
    for attempt in range(1, MAX_RETRIES + 1):
        try:
            # Выбор случайного User-Agent для имитации браузера
            headers = _get_request_headers()
            
            logger.info(f"Загрузка URL: {url} (попытка {attempt}/{MAX_RETRIES})")
            
            # Выполнение запроса с таймаутом
            response = requests.get(url, headers=headers, timeout=30)
            
            # Проверка успешного статус-кода
            response.raise_for_status()
            
            # Получение контента и его кодировки
            html_content = response.text
            
            # Проверка содержимого на минимальную валидность
            if not _validate_html_content(html_content):
                logger.warning(f"Полученный HTML не проходит базовую валидацию для {url}")
                if attempt < MAX_RETRIES:
                    _add_request_delay(RETRY_DELAY * attempt, RETRY_DELAY * attempt * 2)
                    continue
            
            # Сохранение HTML в файл
            with open(save_path, 'w', encoding='utf-8') as f:
                f.write(html_content)
            
            logger.info(f"Успешно загружено и сохранено: {save_path} ({len(html_content)} байт)")
            
            # Добавление небольшой задержки между запросами 
            _add_request_delay(0.5, 1.5)
            
            return html_content
            
        except HTTPError as e:
            status_code = e.response.status_code if hasattr(e, 'response') else 'неизвестен'
            logger.error(f"HTTP ошибка при загрузке {url}: статус {status_code}")
            _handle_retry_logic(e, attempt, url)
            
        except ConnectionError as e:
            logger.error(f"Ошибка соединения при загрузке {url}: {str(e)}")
            _handle_retry_logic(e, attempt, url)
            
        except Timeout as e:
            logger.error(f"Таймаут при загрузке {url}: {str(e)}")
            _handle_retry_logic(e, attempt, url)
            
        except RequestException as e:
            logger.error(f"Ошибка запроса при загрузке {url}: {str(e)}")
            _handle_retry_logic(e, attempt, url)
            
        except Exception as e:
            context = {"url": url, "attempt": attempt}
            log_error(logger, "Непредвиденная ошибка при загрузке", e, context)
            _handle_retry_logic(e, attempt, url)
    
    # Если все попытки исчерпаны, выбрасываем исключение
    error_msg = f"Не удалось загрузить {url} после {MAX_RETRIES} попыток"
    logger.error(error_msg)
    raise RequestException(error_msg)


def fetch_image(image_url: str, save_path: Union[str, Path], base_url: Optional[str] = None) -> bool:
    """
    Загружает изображение и сохраняет его локально.
    
    Args:
        image_url: URL изображения (может быть относительным)
        save_path: Путь для сохранения изображения
        base_url: Базовый URL для преобразования относительных ссылок
        
    Returns:
        bool: True если загрузка успешна, иначе False
    """
    save_path = Path(save_path)
    
    # Преобразование относительного URL в абсолютный, если необходимо
    try:
        absolute_url = _ensure_absolute_url(image_url, base_url)
    except Exception as e:
        context = {"image_url": image_url, "base_url": base_url}
        log_error(logger, "Ошибка при обработке URL изображения", e, context)
        return False
    
    # Проверка наличия файла
    if save_path.exists():
        file_size = save_path.stat().st_size
        if file_size > 0:
            logger.debug(f"Изображение уже существует: {save_path} ({file_size} байт)")
            return True
        else:
            logger.warning(f"Найден пустой файл изображения: {save_path}, будет загружен заново")
    
    # Обеспечение наличия директории
    try:
        os.makedirs(save_path.parent, exist_ok=True)
    except Exception as e:
        context = {"directory": str(save_path.parent)}
        log_error(logger, "Не удалось создать директорию для изображения", e, context)
        return False
    
    # Попытки загрузки с повторами при ошибках
    for attempt in range(1, MAX_RETRIES + 1):
        try:
            # Выбор случайного User-Agent
            headers = _get_request_headers()
            
            # Загрузка изображения
            logger.debug(f"Загрузка изображения: {absolute_url} (попытка {attempt}/{MAX_RETRIES})")
            response = requests.get(absolute_url, headers=headers, timeout=20, stream=True)
            response.raise_for_status()
            
            # Проверка типа контента
            content_type = response.headers.get('Content-Type', '')
            if not content_type.startswith('image/'):
                logger.warning(f"Полученный контент не является изображением: {content_type}")
                if attempt < MAX_RETRIES:
                    _add_request_delay(1, 2)
                    continue
            
            # Сохранение изображения
            with open(save_path, 'wb') as f:
                for chunk in response.iter_content(chunk_size=8192):
                    f.write(chunk)
            
            # Проверка размера сохраненного файла
            file_size = save_path.stat().st_size
            if file_size == 0:
                logger.warning(f"Загруженное изображение имеет нулевой размер: {save_path}")
                if attempt < MAX_RETRIES:
                    _add_request_delay(1, 2)
                    continue
                return False
            
            logger.debug(f"Изображение сохранено: {save_path} ({file_size} байт)")
            
            # Небольшая задержка между запросами
            _add_request_delay(0.2, 0.5)
            
            return True
            
        except HTTPError as e:
            status_code = e.response.status_code if hasattr(e, 'response') else 'неизвестен'
            logger.error(f"HTTP ошибка при загрузке изображения {absolute_url}: статус {status_code}")
            if attempt < MAX_RETRIES:
                _add_request_delay(RETRY_DELAY * attempt, RETRY_DELAY * attempt * 2)
                
        except (ConnectionError, Timeout) as e:
            logger.error(f"Ошибка соединения при загрузке изображения {absolute_url}: {str(e)}")
            if attempt < MAX_RETRIES:
                _add_request_delay(RETRY_DELAY * attempt, RETRY_DELAY * attempt * 2)
                
        except Exception as e:
            context = {"url": absolute_url, "attempt": attempt}
            log_error(logger, "Ошибка при загрузке изображения", e, context)
            if attempt < MAX_RETRIES:
                _add_request_delay(RETRY_DELAY * attempt, RETRY_DELAY * attempt * 2)
    
    logger.error(f"Не удалось загрузить изображение {absolute_url} после {MAX_RETRIES} попыток")
    return False


def fetch_multiple_images(image_urls: List[str], 
                         save_dir: Union[str, Path], 
                         base_url: Optional[str] = None,
                         filename_prefix: str = "") -> Dict[str, str]:
    """
    Загружает несколько изображений.
    
    Args:
        image_urls: Список URL изображений
        save_dir: Директория для сохранения изображений
        base_url: Базовый URL для преобразования относительных ссылок
        filename_prefix: Префикс для имен файлов
        
    Returns:
        Dict[str, str]: Словарь {url: локальный_путь} для успешно загруженных изображений
    """
    save_dir = Path(save_dir)
    
    # Создание директории для сохранения
    try:
        os.makedirs(save_dir, exist_ok=True)
    except Exception as e:
        context = {"directory": str(save_dir)}
        log_error(logger, "Не удалось создать директорию для изображений", e, context)
        return {}
    
    results = {}
    total_images = len(image_urls)
    
    for i, url in enumerate(image_urls):
        if i % 5 == 0 or i == total_images - 1:
            logger.info(f"Загрузка изображений: {i+1}/{total_images}")
        
        try:
            filename = _generate_image_filename(url, i, filename_prefix)
            save_path = save_dir / filename
            
            # Загрузка изображения
            success = fetch_image(url, save_path, base_url)
            
            if success:
                results[url] = str(save_path)
        except Exception as e:
            context = {"url": url, "index": i}
            log_error(logger, "Ошибка при обработке изображения", e, context)
    
    logger.info(f"Загружено {len(results)}/{total_images} изображений в {save_dir}")
    return results


def _get_request_headers() -> Dict[str, str]:
    """
    Создает заголовки для HTTP-запроса с имитацией браузера.
    
    Returns:
        Dict[str, str]: Словарь HTTP-заголовков
    """
    return {
        "User-Agent": random.choice(USER_AGENTS),
        "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,*/*;q=0.8",
        "Accept-Language": "ru-RU,ru;q=0.8,en-US;q=0.5,en;q=0.3",
        "Connection": "keep-alive",
        "Upgrade-Insecure-Requests": "1",
        "Cache-Control": "max-age=0",
    }


def _add_request_delay(min_seconds: float, max_seconds: float) -> None:
    """
    Добавляет случайную задержку между запросами для соблюдения вежливости.
    
    Args:
        min_seconds: Минимальное время задержки в секундах
        max_seconds: Максимальное время задержки в секундах
    """
    delay = random.uniform(min_seconds, max_seconds)
    time.sleep(delay)


def _handle_retry_logic(exception: Exception, attempt: int, url: str) -> None:
    """
    Обрабатывает логику повторных попыток при ошибках.
    
    Args:
        exception: Возникшее исключение
        attempt: Номер текущей попытки
        url: URL, для которого выполняется запрос
    """
    if attempt < MAX_RETRIES:
        # Увеличение задержки с каждой попыткой (экспоненциальная задержка)
        wait_time = RETRY_DELAY * (2 ** (attempt - 1))
        logger.warning(f"Ошибка при загрузке {url}: {str(exception)}. Повторная попытка через {wait_time} сек...")
        time.sleep(wait_time)
    else:
        # Исчерпаны все попытки
        logger.error(f"Не удалось загрузить {url} после {MAX_RETRIES} попыток: {str(exception)}")


def _ensure_absolute_url(url: str, base_url: Optional[str] = None) -> str:
    """
    Преобразует относительный URL в абсолютный.
    
    Args:
        url: URL изображения (может быть относительным)
        base_url: Базовый URL для преобразования относительных ссылок
        
    Returns:
        str: Абсолютный URL
    """
    if url.startswith(('http://', 'https://')):
        return url
    
    if not base_url:
        raise ValueError(f"Невозможно преобразовать относительный URL '{url}' без базового URL")
    
    if url.startswith('/'):
        # Извлекаем домен из base_url
        parsed_base = urlparse(base_url)
        domain = f"{parsed_base.scheme}://{parsed_base.netloc}"
        return urljoin(domain, url)
    
    # Относительный путь к текущей странице
    return urljoin(base_url, url)


def _generate_image_filename(url: str, index: int, prefix: str = "") -> str:
    """
    Генерирует имя файла для изображения на основе его URL.
    
    Args:
        url: URL изображения
        index: Индекс изображения в списке
        prefix: Префикс для имени файла
        
    Returns:
        str: Имя файла
    """
    # Получение имени файла из URL
    parsed_url = urlparse(url)
    path = parsed_url.path
    
    # Извлечение имени файла из пути
    filename = os.path.basename(path)
    
    # Если имя файла пустое или не содержит расширения
    if not filename or '.' not in filename:
        filename = f"image_{index}.jpg"
    
    # Добавление префикса
    if prefix:
        filename = f"{prefix}_{filename}"
    
    # Очистка имени файла
    filename = sanitize_filename(filename)
    
    # Ограничение длины имени файла
    if len(filename) > 100:
        name, ext = os.path.splitext(filename)
        filename = f"{name[:90]}{ext}"
    
    return filename


def _validate_html_content(html: str) -> bool:
    """
    Проверяет HTML-контент на базовую валидность.
    
    Args:
        html: HTML-контент для проверки
        
    Returns:
        bool: True, если контент проходит базовую валидацию
    """
    # Проверка на минимальную длину
    if len(html) < 100:
        return False
    
    # Проверка на наличие базовых HTML-тегов
    if not all(tag in html.lower() for tag in ['<html', '<head', '<body']):
        return False
    
    return True