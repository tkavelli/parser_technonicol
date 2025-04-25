"""
Модуль для парсинга HTML-контента статей.
Извлекает основную структуру из страницы, удаляя ненужные элементы.
Также извлекает хлебные крошки для улучшения категоризации и построения метаданных.
"""
import re
import logging
import traceback
import json
import time
from typing import Dict, List, Tuple, Any, Optional, Union, Set
from urllib.parse import urlparse, unquote
from bs4 import BeautifulSoup, Tag, NavigableString

from text_utils import log_error, extract_breadcrumbs_from_url

# Настройка логирования
logger = logging.getLogger(__name__)


def parse_html(html_content: str, url: str) -> Tuple[List[Dict[str, Any]], str, List[str]]:
    """
    Парсит HTML-контент статьи и извлекает структурированное содержимое.
    
    Args:
        html_content: HTML-контент статьи
        url: URL статьи (используется для контекста и относительных ссылок)
    
    Returns:
        Tuple[List[Dict[str, Any]], str, List[str]]: 
            - Список элементов содержимого
            - Заголовок статьи
            - Хлебные крошки (для метаданных и категоризации)
    """
    logger.info(f"Начало парсинга HTML для {url}")
    start_time = time.time()
    
    try:
        # Проверка на пустой или некорректный HTML
        if not html_content or len(html_content) < 100:
            logger.error(f"Получен пустой или слишком короткий HTML для {url}")
            return [], "Без заголовка", []
        
        # Создание объекта BeautifulSoup
        soup = BeautifulSoup(html_content, 'html.parser')
        
        # Удаление ненужных элементов
        _clean_html(soup)
        
        # Попытка извлечь хлебные крошки из URL
        url_breadcrumbs = extract_breadcrumbs_from_url(url)
        if url_breadcrumbs:
            logger.info(f"Извлечены хлебные крошки из URL: {', '.join(url_breadcrumbs)}")
        
        # Извлечение хлебных крошек из HTML (навигационной цепочки)
        html_breadcrumbs = extract_breadcrumbs_from_html(soup)
        
        # Объединяем хлебные крошки с приоритетом HTML
        breadcrumbs = html_breadcrumbs if html_breadcrumbs else url_breadcrumbs
        logger.info(f"Итоговые хлебные крошки: {', '.join(breadcrumbs)}")
        
        # Поиск основного контейнера статьи
        article_container = _find_article_container(soup)
        
        if not article_container:
            logger.warning(f"Не удалось найти основной контейнер статьи для {url}")
            # Используем <body> как запасной вариант
            article_container = soup.body
            if not article_container:
                logger.error(f"HTML не содержит тега <body> для {url}")
                return [], "Без заголовка", breadcrumbs
        
        # Извлечение заголовка статьи (h1)
        title = _extract_title(article_container, soup)
        
        # Извлечение всех структурных элементов
        content_elements = _extract_content_elements(article_container)
        
        # Проверка результатов
        if not content_elements:
            logger.warning(f"Не удалось извлечь содержимое для {url}")
        
        # Сохранение JSON-метаданных в элементах для будущего использования в RAG
        content_elements = _enrich_elements_with_metadata(content_elements, breadcrumbs, url)
        
        parse_time = time.time() - start_time
        logger.info(f"Парсинг завершен за {parse_time:.2f} сек: найдено {len(content_elements)} элементов, "
                   f"заголовок: {title}, хлебные крошки: {breadcrumbs}")
        
        return content_elements, title, breadcrumbs
        
    except Exception as e:
        parse_time = time.time() - start_time
        context = {
            "url": url,
            "html_length": len(html_content) if html_content else 0,
            "parse_time": f"{parse_time:.2f} сек"
        }
        log_error(logger, f"Ошибка при парсинге HTML", e, context)
        return [], "Ошибка парсинга", []


def extract_breadcrumbs_from_html(soup: BeautifulSoup) -> List[str]:
    """
    Извлекает хлебные крошки (навигационную цепочку) из HTML.
    
    Args:
        soup: HTML-документ как объект BeautifulSoup
        
    Returns:
        List[str]: Список хлебных крошек (названий категорий)
    """
    breadcrumbs = []
    
    try:
        # Вариант 1: Стандартный класс "breadcrumbs" для навигации
        breadcrumbs_nav = (
            soup.find('nav', class_=lambda c: c and 'breadcrumb' in c.lower()) or 
            soup.find('div', class_=lambda c: c and 'breadcrumb' in c.lower()) or
            soup.find('ul', class_=lambda c: c and 'breadcrumb' in c.lower())
        )
        
        if breadcrumbs_nav:
            for item in breadcrumbs_nav.find_all(['a', 'li', 'span']):
                breadcrumb_text = item.get_text(strip=True)
                if (breadcrumb_text and 
                    breadcrumb_text.lower() not in ['главная', 'home'] and 
                    breadcrumb_text not in breadcrumbs):
                    breadcrumbs.append(breadcrumb_text)
            
            if breadcrumbs:
                logger.debug(f"Извлечены хлебные крошки из навигации: {breadcrumbs}")
                return breadcrumbs
        
        # Вариант 2: Структура микроразметки Schema.org для хлебных крошек
        breadcrumbs_schema = soup.find(
            ['ol', 'ul'], 
            class_=lambda x: x and ('breadcrumb' in x.lower() or 'navigation' in x.lower())
        )
        
        if breadcrumbs_schema:
            for item in breadcrumbs_schema.find_all('li'):
                link = item.find('a')
                text = link.get_text(strip=True) if link else item.get_text(strip=True)
                if (text and 
                    text.lower() not in ['главная', 'home'] and 
                    text not in breadcrumbs):
                    breadcrumbs.append(text)
            
            if breadcrumbs:
                logger.debug(f"Извлечены хлебные крошки из Schema.org разметки: {breadcrumbs}")
                return breadcrumbs
        
        # Вариант 3: TechnoNICOL-специфичный формат
        for link in soup.find_all('a'):
            href = link.get('href', '')
            if any(path in href for path in ['/knowledge-base/', '/catalog/', '/products/']):
                breadcrumb_text = link.get_text(strip=True)
                if (breadcrumb_text and 
                    breadcrumb_text.lower() not in ['главная', 'home'] and 
                    breadcrumb_text not in breadcrumbs):
                    breadcrumbs.append(breadcrumb_text)
        
        if breadcrumbs:
            logger.debug(f"Извлечены хлебные крошки из ссылок TechnoNICOL: {breadcrumbs}")
            return breadcrumbs
        
        # Вариант 4: Извлечение из мета-тегов
        meta_url = soup.find('meta', property='og:url')
        url_content = meta_url.get('content', '') if meta_url else ''
        
        if url_content:
            breadcrumbs_from_meta = extract_breadcrumbs_from_url(url_content)
            if breadcrumbs_from_meta:
                logger.debug(f"Извлечены хлебные крошки из мета-тега og:url: {breadcrumbs_from_meta}")
                return breadcrumbs_from_meta
    
    except Exception as e:
        context = {"breadcrumbs_found": breadcrumbs}
        log_error(logger, "Ошибка при извлечении хлебных крошек из HTML", e, context)
    
    return breadcrumbs


def _clean_html(soup: BeautifulSoup) -> None:
    """
    Очищает HTML от ненужных элементов.
    
    Args:
        soup: Объект BeautifulSoup для очистки
    """
    try:
        # Удаление скриптов
        for script in soup.find_all('script'):
            script.decompose()
        
        # Удаление стилей
        for style in soup.find_all('style'):
            style.decompose()
        
        # Удаление комментариев
        for comment in soup.find_all(text=lambda text: isinstance(text, str) and text.strip().startswith('<!--')):
            comment.extract()
        
        # Список элементов для удаления
        tags_to_remove = ['nav', 'header', 'footer', 'aside']
        
        # Удаление навигации, хедера, футера и т.п. (но не хлебные крошки)
        for tag_name in tags_to_remove:
            for element in soup.find_all(tag_name):
                # Не удаляем хлебные крошки
                if (tag_name == 'nav' and element.get('class') and
                    any('breadcrumb' in c.lower() for c in element.get('class'))):
                    continue
                element.decompose()
        
        # Список классов, содержащих ненужные элементы
        unwanted_classes = [
            'menu', 'navbar', 'navigation', 'banner', 'advert', 'advertisement', 
            'cookie', 'popup', 'modal', 'sidebar', 'footer', 'header', 
            'related', 'similar', 'recommend', 'share', 'social'
        ]
        
        # Удаление элементов с определенными классами (но не хлебные крошки)
        for unwanted_class in unwanted_classes:
            # Пропускаем класс 'navigation', если в нем могут быть хлебные крошки
            if unwanted_class == 'navigation':
                for element in soup.find_all(class_=lambda c: c and unwanted_class in str(c).lower()):
                    # Проверяем, есть ли в этом элементе хлебные крошки
                    if 'breadcrumb' not in str(element.get('class', '')).lower() and not element.find(class_=lambda c: c and 'breadcrumb' in str(c).lower()):
                        element.decompose()
            else:
                for element in soup.find_all(class_=lambda c: c and unwanted_class in str(c).lower()):
                    # Не удаляем хлебные крошки
                    if 'breadcrumb' not in str(element.get('class', '')).lower() and not element.find(class_=lambda c: c and 'breadcrumb' in str(c).lower()):
                        element.decompose()
        
    except Exception as e:
        log_error(logger, "Ошибка при очистке HTML", e)


def _find_article_container(soup: BeautifulSoup) -> Optional[Tag]:
    """
    Находит основной контейнер статьи в HTML.
    
    Args:
        soup: Объект BeautifulSoup для поиска
        
    Returns:
        Optional[Tag]: Найденный контейнер статьи или None
    """
    # Пробуем различные варианты контейнеров, которые могут содержать основную статью
    container_candidates = [
        # Прямое указание на контейнер статьи
        soup.find('article'),
        soup.find('main', {'role': 'main'}),
        soup.find('div', {'role': 'main'}),
        
        # Классы, обычно содержащие контент статьи
        soup.find('div', class_='article-content'),
        soup.find('div', class_='article'),
        soup.find('div', class_='post-content'),
        soup.find('div', class_='entry-content'),
        soup.find('div', class_='content'),
        soup.find('div', class_='main-content'),
        soup.find('main'),
        
        # Поиск по частичному совпадению класса
        soup.find('div', class_=lambda c: c and ('article' in str(c).lower() or 'content' in str(c).lower())),
        soup.find('div', id=lambda i: i and ('article' in str(i).lower() or 'content' in str(i).lower())),
    ]
    
    # Возвращаем первый найденный контейнер
    for container in container_candidates:
        if container:
            return container
    
    # Альтернативный подход: найти h1 и вернуть его родителя или раздел
    h1 = soup.find('h1')
    if h1:
        # Проверяем ближайший section или div
        section_parent = h1.find_parent(['section', 'div', 'article'])
        if section_parent:
            return section_parent
        return h1.parent
    
    # Не найден подходящий контейнер
    return None


def _extract_title(article_container: Tag, soup: BeautifulSoup) -> str:
    """
    Извлекает заголовок статьи.
    
    Args:
        article_container: Контейнер статьи
        soup: Полный объект BeautifulSoup для запасного поиска
        
    Returns:
        str: Заголовок статьи
    """
    try:
        # Поиск h1 в контейнере статьи
        h1 = article_container.find('h1')
        
        # Если не найден в контейнере, ищем в полном документе
        if not h1:
            h1 = soup.find('h1')
        
        # Если найден h1, извлекаем текст
        if h1:
            title_text = h1.get_text(strip=True)
            if title_text:
                return title_text
        
        # Если h1 не найден или пуст, пробуем заголовок или мета-тег
        title_tag = soup.find('title')
        if title_tag:
            title_text = title_tag.get_text(strip=True)
            # Удаляем общие суффиксы сайта из заголовка
            title_text = re.sub(r'\s+[-|]\s+.*$', '', title_text)
            if title_text:
                return title_text
        
        # Мета-тег title
        meta_title = (
            soup.find('meta', property='og:title') or 
            soup.find('meta', attrs={'name': 'title'})
        )
        if meta_title and meta_title.get('content'):
            return meta_title.get('content')
        
    except Exception as e:
        log_error(logger, "Ошибка при извлечении заголовка статьи", e)
    
    # Если ничего не найдено
    return "Без заголовка"


def _extract_content_elements(container: Tag) -> List[Dict[str, Any]]:
    """
    Извлекает все структурные элементы из контейнера статьи.
    
    Args:
        container: Контейнер статьи
        
    Returns:
        List[Dict[str, Any]]: Список элементов содержимого
    """
    elements = []
    
    try:
        # Обход всех непосредственных потомков контейнера
        for element in container.find_all(recursive=False):
            element_data = _process_element(element)
            if element_data:
                elements.append(element_data)
        
        # Если непосредственных потомков нет или их мало, обрабатываем все вложенные элементы
        if len(elements) < 3:
            elements = []
            
            # Список тегов, которые нас интересуют
            relevant_tags = ['h1', 'h2', 'h3', 'h4', 'h5', 'h6', 'p', 'ul', 'ol', 
                           'table', 'img', 'figure', 'blockquote', 'div']
            
            for element in container.find_all(relevant_tags):
                # Пропускаем вложенные элементы в уже обработанных контейнерах
                if any(parent.name in ['figure', 'blockquote', 'li'] 
                       for parent in element.parents 
                       if parent != container):
                    continue
                
                element_data = _process_element(element)
                if element_data:
                    elements.append(element_data)
        
        # Логирование статистики по типам элементов
        element_types = {}
        for element in elements:
            element_type = element.get("type", "unknown")
            element_types[element_type] = element_types.get(element_type, 0) + 1
        
        logger.debug(f"Извлечено элементов по типам: {element_types}")
        
    except Exception as e:
        log_error(logger, "Ошибка при извлечении содержимого", e)
    
    return elements


def _process_element(element: Tag) -> Optional[Dict[str, Any]]:
    """
    Обрабатывает отдельный HTML-элемент и преобразует его в структурированный формат.
    
    Args:
        element: HTML-элемент для обработки
        
    Returns:
        Optional[Dict[str, Any]]: Структурированное представление элемента или None
    """
    tag_name = element.name
    
    # Пропуск пустых элементов и элементов без имени тега
    if not tag_name:
        return None
    
    try:
        # Пропуск элементов без текста, кроме изображений
        if tag_name not in ['img', 'figure'] and not element.get_text(strip=True):
            return None
        
        # Обработка заголовков
        if tag_name in ['h1', 'h2', 'h3', 'h4', 'h5', 'h6']:
            return _process_heading(element)
        
        # Обработка абзацев
        elif tag_name == 'p':
            return _process_paragraph(element)
        
        # Обработка списков
        elif tag_name in ['ul', 'ol']:
            return _process_list(element)
        
        # Обработка таблиц
        elif tag_name == 'table':
            return _process_table(element)
        
        # Обработка изображений
        elif tag_name == 'img':
            return _process_image(element)
        
        # Обработка фигур (изображения с подписями)
        elif tag_name == 'figure':
            return _process_figure(element)
        
        # Обработка блочных цитат
        elif tag_name == 'blockquote':
            return _process_blockquote(element)
        
        # Обработка div (только если содержит прямой текст)
        elif tag_name == 'div':
            # Проверяем, есть ли прямой текст в div (не в дочерних элементах)
            direct_text = ''.join(str(c) for c in element.contents 
                                 if isinstance(c, NavigableString)).strip()
            
            if direct_text:
                return {
                    "type": "paragraph",
                    "text": element.get_text(strip=True),
                    "html": str(element)
                }
            
    except Exception as e:
        logger.error(f"Ошибка при обработке элемента {tag_name}: {str(e)}")
        logger.debug(f"Трассировка: {traceback.format_exc()}")
    
    return None


def _process_heading(element: Tag) -> Dict[str, Any]:
    """
    Обрабатывает HTML-элемент заголовка.
    
    Args:
        element: HTML-элемент заголовка
        
    Returns:
        Dict[str, Any]: Структурированное представление заголовка
    """
    return {
        "type": "heading",
        "level": int(element.name[1]),
        "text": element.get_text(strip=True),
        "html": str(element)
    }


def _process_paragraph(element: Tag) -> Optional[Dict[str, Any]]:
    """
    Обрабатывает HTML-элемент абзаца.
    
    Args:
        element: HTML-элемент абзаца
        
    Returns:
        Optional[Dict[str, Any]]: Структурированное представление абзаца или None
    """
    text = element.get_text(strip=True)
    if not text:  # Проверка на пустой текст
        return None
    
    # Сохраняем оригинальный HTML для извлечения ссылок
    return {
        "type": "paragraph",
        "text": text,
        "html": str(element)
    }


def _process_list(element: Tag) -> Optional[Dict[str, Any]]:
    """
    Обрабатывает HTML-элемент списка.
    
    Args:
        element: HTML-элемент списка
        
    Returns:
        Optional[Dict[str, Any]]: Структурированное представление списка или None
    """
    items = [li.get_text(strip=True) for li in element.find_all('li') if li.get_text(strip=True)]
    if not items:  # Проверка на пустой список
        return None
    
    return {
        "type": "list",
        "list_type": "ordered" if element.name == 'ol' else "unordered",
        "items": items,
        "html": str(element)
    }


def _process_table(table: Tag) -> Dict[str, Any]:
    """
    Обрабатывает HTML-таблицу.
    
    Args:
        table: HTML-элемент таблицы
        
    Returns:
        Dict[str, Any]: Структурированное представление таблицы
    """
    headers = []
    rows = []
    
    # Извлечение заголовков таблицы
    thead = table.find('thead')
    if thead:
        th_elements = thead.find_all('th')
        if th_elements:
            headers = [th.get_text(strip=True) for th in th_elements]
    
    # Если заголовков нет в thead, ищем в первой строке
    if not headers:
        first_row = table.find('tr')
        if first_row:
            th_elements = first_row.find_all('th')
            if th_elements:
                headers = [th.get_text(strip=True) for th in th_elements]
            else:
                # Используем первую строку как заголовки, если в ней есть td
                td_elements = first_row.find_all('td')
                if td_elements:
                    headers = [td.get_text(strip=True) for td in td_elements]
    
    # Извлечение строк таблицы
    body_rows = table.find_all('tr')
    
    # Пропускаем первую строку, если она была использована как заголовки
    start_index = 1 if headers and len(body_rows) > 0 and not thead else 0
    
    for row in body_rows[start_index:]:
        cells = row.find_all(['td', 'th'])
        if cells:
            row_data = [cell.get_text(strip=True) for cell in cells]
            rows.append(row_data)
    
    # Проверка на пустую таблицу
    if not headers and not rows:
        return {
            "type": "paragraph",
            "text": table.get_text(strip=True),
            "html": str(table)
        }
    
    # Добавляем метаданные о таблице для лучшей индексации числовых данных
    metadata = {
        "has_numeric_data": False,
        "numeric_columns": []
    }
    
    # Обнаружение столбцов с числовыми данными
    numeric_pattern = r'\d+[.,]?\d*\s*(?:[а-яА-Яa-zA-Z°%]+)?'
    for col_idx in range(len(headers) if headers else (len(rows[0]) if rows else 0)):
        # Проверяем заголовок (может содержать указание на единицы измерения)
        if headers and col_idx < len(headers):
            if re.search(r'(?:мм|см|м|км|кг|г|%|°[CС]|\bт\b|шт)', headers[col_idx]):
                metadata["numeric_columns"].append(col_idx)
                metadata["has_numeric_data"] = True
                continue
                
        # Проверяем ячейки столбца
        numeric_count = 0
        total_cells = 0
        
        for row in rows:
            if col_idx < len(row):
                total_cells += 1
                if re.search(numeric_pattern, row[col_idx]):
                    numeric_count += 1
        
        # Если более 50% ячеек содержат числа, считаем столбец числовым
        if total_cells > 0 and numeric_count / total_cells > 0.5:
            metadata["numeric_columns"].append(col_idx)
            metadata["has_numeric_data"] = True
    
    return {
        "type": "table",
        "headers": headers,
        "rows": rows,
        "html": str(table),
        "metadata": metadata
    }


def _process_image(img: Tag) -> Optional[Dict[str, Any]]:
    """
    Обрабатывает HTML-элемент изображения.
    
    Args:
        img: HTML-элемент изображения
        
    Returns:
        Optional[Dict[str, Any]]: Структурированное представление изображения или None
    """
    src = img.get('src', '')
    if not src:
        return None
    
    alt = img.get('alt', '')
    
    # Проверка на декоративные изображения
    if src.lower().endswith(('.gif', '.svg')) and (
        'icon' in src.lower() or 
        'logo' in src.lower() or 
        'separator' in src.lower() or
        'decoration' in src.lower()
    ):
        return None
    
    return {
        "type": "image",
        "src": src,
        "alt": alt,
        "html": str(img)
    }


def _process_figure(figure: Tag) -> Optional[Dict[str, Any]]:
    """
    Обрабатывает HTML-элемент figure (изображение с подписью).
    
    Args:
        figure: HTML-элемент figure
        
    Returns:
        Optional[Dict[str, Any]]: Структурированное представление figure или None
    """
    img = figure.find('img')
    figcaption = figure.find('figcaption')
    
    if not img or not img.get('src'):
        return None
    
    caption = figcaption.get_text(strip=True) if figcaption else ""
    
    return {
        "type": "figure",
        "src": img.get('src', ''),
        "alt": img.get('alt', ''),
        "caption": caption,
        "html": str(figure)
    }


def _process_blockquote(blockquote: Tag) -> Dict[str, Any]:
    """
    Обрабатывает HTML-элемент blockquote (цитата).
    
    Args:
        blockquote: HTML-элемент blockquote
        
    Returns:
        Dict[str, Any]: Структурированное представление цитаты
    """
    text = blockquote.get_text(strip=True)
    
    return {
        "type": "blockquote",
        "text": text,
        "html": str(blockquote)
    }


def _enrich_elements_with_metadata(elements: List[Dict[str, Any]], 
                                 breadcrumbs: List[str], 
                                 url: str) -> List[Dict[str, Any]]:
    """
    Обогащает элементы метаданными для последующего использования в гибридной RAG-системе.
    
    Args:
        elements: Список структурированных элементов контента
        breadcrumbs: Список хлебных крошек
        url: URL статьи
        
    Returns:
        List[Dict[str, Any]]: Обогащенные элементы с метаданными
    """
    from text_utils import extract_numbers
    
    for element in elements:
        # Добавляем базовые метаданные ко всем элементам
        if "metadata" not in element:
            element["metadata"] = {}
        
        # Добавляем информацию о категориях из хлебных крошек
        element["metadata"]["categories"] = breadcrumbs
        element["metadata"]["source_url"] = url
        
        # Добавляем специфичные метаданные в зависимости от типа элемента
        if element["type"] == "table" and "has_numeric_data" not in element["metadata"]:
            # Анализ потенциальных числовых данных в таблице
            element["metadata"]["numerical_columns"] = _detect_numerical_columns(element)
        
        elif element["type"] in ["paragraph", "blockquote"]:
            # Извлечение потенциальных числовых значений и единиц измерения
            numerical_values = extract_numbers(element.get("text", ""))
            if numerical_values:
                element["metadata"]["numerical_values"] = numerical_values
                element["metadata"]["has_numeric_data"] = True
    
    return elements


def _detect_numerical_columns(table_element: Dict[str, Any]) -> List[int]:
    """
    Определяет числовые столбцы в таблице для будущего использования в SQL/JSON поиске.
    
    Args:
        table_element: Структурированное представление таблицы
        
    Returns:
        List[int]: Индексы числовых столбцов
    """
    # Если уже есть метаданные о числовых столбцах, используем их
    if "metadata" in table_element and "numeric_columns" in table_element["metadata"]:
        return table_element["metadata"]["numeric_columns"]
        
    numerical_columns = []
    
    try:
        rows = table_element.get("rows", [])
        if not rows:
            return []
        
        column_count = len(rows[0])
        
        # Для каждого столбца проверяем, содержит ли он числовые данные
        for col_idx in range(column_count):
            numerical_count = 0
            total_count = 0
            
            for row in rows:
                if col_idx < len(row):
                    cell_value = row[col_idx]
                    total_count += 1
                    
                    # Проверка на число с единицей измерения
                    is_numerical = bool(re.search(r'\d+[.,]?\d*\s*[а-яА-Яa-zA-Z°%]+', cell_value) or
                                       re.match(r'^[-+]?\d+[.,]?\d*$', cell_value))
                    
                    if is_numerical:
                        numerical_count += 1
            
            # Если более 50% ячеек в столбце содержат числа, считаем столбец числовым
            if total_count > 0 and numerical_count / total_count > 0.5:
                numerical_columns.append(col_idx)
    
    except Exception as e:
        log_error(logger, "Ошибка при определении числовых столбцов", e)
    
    return numerical_columns