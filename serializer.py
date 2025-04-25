"""
Модуль для сериализации структурированных данных статьи в файлы.
Генерирует YAML, JSON, Markdown и чистый HTML для визуализации результатов.
Поддерживает сохранение метаданных для гибридной RAG-архитектуры.
"""
import os
import json
import yaml
import logging
import traceback
from datetime import datetime
from pathlib import Path
from typing import Dict, Any, Union, List, Optional
from bs4 import BeautifulSoup

# Настройка логирования
logger = logging.getLogger(__name__)


def serialize_article(article_data: Dict[str, Any], output_dir: Union[str, Path]) -> bool:
    """
    Сериализует данные статьи в различные форматы и сохраняет их в указанную директорию.
    
    Args:
        article_data: Структурированные данные статьи
        output_dir: Директория для сохранения файлов
        
    Returns:
        bool: True в случае успешной сериализации, False в случае ошибки
    """
    output_dir = Path(output_dir)
    
    try:
        # Создание директории, если она не существует
        os.makedirs(output_dir, exist_ok=True)
        
        article_title = article_data.get("title", "Без заголовка")
        article_url = article_data.get("url", "")
        
        logger.info(f"Сериализация статьи '{article_title}' в {output_dir}")
        
        # 1. Сериализация в YAML
        yaml_path = output_dir / "structure.yaml"
        yaml_success = _save_yaml(article_data, yaml_path)
        
        # 2. Сериализация в JSON
        json_path = output_dir / "structure.json"
        json_success = _save_json(article_data, json_path)
        
        # 3. Создание чистого HTML
        html_path = output_dir / "clean.html"
        html_success = _save_clean_html(article_data, html_path)
        
        # 4. Генерация Markdown
        md_path = output_dir / "content.md"
        markdown_success = _save_markdown(article_data, md_path)
        
        # 5. Сохранение метаданных для RAG (если есть)
        metadata_path = output_dir / "metadata.json"
        metadata_success = _save_rag_metadata(article_data, metadata_path)
        
        # Проверка успешности всех операций
        all_success = all([yaml_success, json_success, html_success, markdown_success, metadata_success])
        
        if all_success:
            logger.info(f"Сериализация успешно завершена для статьи: {article_title}")
        else:
            logger.warning(f"Сериализация завершена с ошибками для статьи: {article_title}")
        
        return all_success
        
    except Exception as e:
        logger.error(f"Ошибка при сериализации статьи: {str(e)}")
        logger.debug(f"Трассировка: {traceback.format_exc()}")
        return False


def _save_yaml(data: Dict[str, Any], file_path: Union[str, Path]) -> bool:
    """
    Сохраняет данные в YAML-файл.
    
    Args:
        data: Данные для сохранения
        file_path: Путь для сохранения файла
        
    Returns:
        bool: True в случае успешного сохранения, False в случае ошибки
    """
    try:
        # Подготовка данных для YAML
        yaml_data = _prepare_data_for_serialization(data)
        
        # Настройка YAML-дампера для корректной работы с русским языком
        class SafeDumper(yaml.SafeDumper):
            pass
        
        def represent_str(self, data):
            if '\n' in data:
                return self.represent_scalar('tag:yaml.org,2002:str', data, style='|')
            return self.represent_scalar('tag:yaml.org,2002:str', data)
        
        SafeDumper.add_representer(str, represent_str)
        
        # Сохранение в YAML
        with open(file_path, 'w', encoding='utf-8') as yaml_file:
            yaml.dump(yaml_data, yaml_file, default_flow_style=False, 
                    allow_unicode=True, sort_keys=False, Dumper=SafeDumper)
        
        logger.debug(f"YAML сохранен в {file_path}")
        return True
        
    except Exception as e:
        logger.error(f"Ошибка при сохранении YAML: {str(e)}")
        logger.debug(f"Трассировка: {traceback.format_exc()}")
        return False


def _save_json(data: Dict[str, Any], file_path: Union[str, Path]) -> bool:
    """
    Сохраняет данные в JSON-файл.
    
    Args:
        data: Данные для сохранения
        file_path: Путь для сохранения файла
        
    Returns:
        bool: True в случае успешного сохранения, False в случае ошибки
    """
    try:
        # Подготовка данных для JSON
        json_data = _prepare_data_for_serialization(data)
        
        # Сохранение в JSON
        with open(file_path, 'w', encoding='utf-8') as json_file:
            json.dump(json_data, json_file, ensure_ascii=False, indent=2)
        
        logger.debug(f"JSON сохранен в {file_path}")
        return True
        
    except Exception as e:
        logger.error(f"Ошибка при сохранении JSON: {str(e)}")
        logger.debug(f"Трассировка: {traceback.format_exc()}")
        return False


def _save_clean_html(data: Dict[str, Any], file_path: Union[str, Path]) -> bool:
    """
    Создает и сохраняет чистый HTML-файл из структурированных данных.
    
    Args:
        data: Данные статьи
        file_path: Путь для сохранения файла
        
    Returns:
        bool: True в случае успешного сохранения, False в случае ошибки
    """
    try:
        # Создание базового HTML
        html = BeautifulSoup("""
        <!DOCTYPE html>
        <html lang="ru">
        <head>
            <meta charset="UTF-8">
            <meta name="viewport" content="width=device-width, initial-scale=1.0">
            <title>{title}</title>
            <style>
                body { font-family: Arial, sans-serif; line-height: 1.6; max-width: 800px; margin: 0 auto; padding: 20px; }
                h1 { color: #2c3e50; }
                h2 { color: #3498db; margin-top: 30px; }
                h3 { color: #2980b9; }
                h4, h5, h6 { color: #1f618d; }
                img { max-width: 100%; height: auto; }
                table { border-collapse: collapse; width: 100%; margin: 20px 0; }
                th, td { padding: 8px; text-align: left; border: 1px solid #ddd; }
                th { background-color: #f2f2f2; }
                .figure { margin: 20px 0; }
                .figure img { display: block; }
                .figure figcaption { font-size: 0.9em; color: #555; margin-top: 5px; }
                .article-meta { margin-bottom: 20px; color: #666; font-size: 0.9em; }
                .article-source { margin-top: 40px; padding-top: 10px; border-top: 1px solid #eee; font-size: 0.9em; color: #666; }
                .technical-specs { background-color: #f9f9f9; border: 1px solid #eee; padding: 10px; margin: 15px 0; }
                .technical-specs h3 { color: #2c3e50; margin-top: 0; }
                .technical-specs table { margin: 0; }
                .labels { display: flex; flex-wrap: wrap; gap: 5px; margin: 10px 0; }
                .label { background: #e1f5fe; color: #0277bd; padding: 3px 8px; border-radius: 12px; font-size: 0.8em; }
            </style>
        </head>
        <body>
            <article>
                <h1></h1>
                <div class="article-meta">
                    <p>Добавлено: <time>{parsed_at}</time></p>
                </div>
                <div class="article-content">
                </div>
                <div class="article-source">
                    <p>Источник: <a href="{url}" target="_blank">{url}</a></p>
                </div>
            </article>
        </body>
        </html>
        """, 'html.parser')
        
        # Установка заголовка
        html.title.string = data.get("title", "Без заголовка")
        html.find('h1').string = data.get("title", "Без заголовка")
        
        # Установка мета-информации
        if "parsed_at" in data:
            html.find('time').string = data["parsed_at"]
        else:
            html.find('time').string = datetime.now().strftime("%Y-%m-%dT%H:%M:%S")
            
        # Установка URL
        for a in html.find_all('a', href=True):
            a['href'] = data.get("url", "#")
            a.string = data.get("url", "#")
        
        # Добавление хлебных крошек, если они есть
        if "breadcrumbs" in data and data["breadcrumbs"]:
            breadcrumbs_div = html.new_tag("div", attrs={"class": "breadcrumbs"})
            breadcrumbs_div.string = " > ".join(data["breadcrumbs"])
            html.article.insert(1, breadcrumbs_div)
        
        # Получение контейнера для содержимого
        content_div = html.find('div', class_='article-content')
        
        # Добавление содержимого из чанков
        _add_chunks_to_html(data.get("chunks", []), content_div, html)
        
        # Сохранение HTML
        with open(file_path, 'w', encoding='utf-8') as html_file:
            html_file.write(str(html.prettify()))
        
        logger.debug(f"Чистый HTML сохранен в {file_path}")
        return True
        
    except Exception as e:
        logger.error(f"Ошибка при сохранении чистого HTML: {str(e)}")
        logger.debug(f"Трассировка: {traceback.format_exc()}")
        return False


def _add_chunks_to_html(chunks: List[Dict[str, Any]], container: BeautifulSoup, soup: BeautifulSoup, level: int = 0) -> None:
    """
    Рекурсивно добавляет чанки в HTML-документ.
    
    Args:
        chunks: Список чанков для добавления
        container: HTML-контейнер для добавления чанков
        soup: Объект BeautifulSoup для создания новых элементов
        level: Текущий уровень вложенности (для вычисления уровня заголовков)
    """
    for chunk in chunks:
        try:
            # Создаем секцию для чанка
            section = soup.new_tag("section", attrs={"id": chunk.get("id", "")})
            container.append(section)
            
            # Добавление заголовка раздела
            if chunk.get("title") and chunk["title"] != "Введение":
                # Определение уровня заголовка (h1, h2, h3, ...) на основе вложенности
                heading_level = min(6, max(2, chunk.get("level", 2)))
                heading = soup.new_tag(f"h{heading_level}")
                heading.string = chunk["title"]
                section.append(heading)
            
            # Добавление меток, если они есть
            if chunk.get("labels"):
                labels_div = soup.new_tag("div", attrs={"class": "labels"})
                for label in chunk["labels"]:
                    label_span = soup.new_tag("span", attrs={"class": "label"})
                    label_span.string = label
                    labels_div.append(label_span)
                section.append(labels_div)
            
            # Добавление содержимого раздела
            for element in chunk.get("elements", []):
                _add_element_to_html(element, section, soup)
            
            # Добавление технических характеристик, если они есть
            if "metadata" in chunk and "tech_specs" in chunk["metadata"] and chunk["metadata"]["tech_specs"]:
                _add_tech_specs_to_html(chunk["metadata"]["tech_specs"], section, soup)
            
            # Рекурсивно добавляем дочерние чанки
            if chunk.get("children"):
                _add_chunks_to_html(chunk["children"], section, soup, level + 1)
                
        except Exception as e:
            logger.error(f"Ошибка при добавлении чанка {chunk.get('id', 'unknown')} в HTML: {str(e)}")
            logger.debug(f"Трассировка: {traceback.format_exc()}")


def _add_element_to_html(element: Dict[str, Any], container: BeautifulSoup, soup: BeautifulSoup) -> None:
    """
    Добавляет отдельный элемент в HTML-контейнер.
    
    Args:
        element: Элемент контента
        container: HTML-контейнер для добавления элемента
        soup: Объект BeautifulSoup для создания новых элементов
    """
    try:
        element_type = element.get("type", "")
        
        if element_type == "paragraph":
            p = soup.new_tag("p")
            p.string = element.get("text", "")
            container.append(p)
            
        elif element_type == "list":
            list_type = "ol" if element.get("list_type") == "ordered" else "ul"
            list_tag = soup.new_tag(list_type)
            
            for item in element.get("items", []):
                li = soup.new_tag("li")
                li.string = item
                list_tag.append(li)
            
            container.append(list_tag)
            
        elif element_type == "table":
            table = soup.new_tag("table")
            
            # Добавление заголовков таблицы
            if element.get("headers"):
                thead = soup.new_tag("thead")
                tr = soup.new_tag("tr")
                
                for header in element["headers"]:
                    th = soup.new_tag("th")
                    th.string = header
                    tr.append(th)
                
                thead.append(tr)
                table.append(thead)
            
            # Добавление данных таблицы
            tbody = soup.new_tag("tbody")
            
            for row in element.get("rows", []):
                tr = soup.new_tag("tr")
                
                for cell in row:
                    td = soup.new_tag("td")
                    td.string = cell
                    tr.append(td)
                
                tbody.append(tr)
            
            table.append(tbody)
            container.append(table)
            
        elif element_type in ["image", "figure"]:
            if element_type == "figure":
                figure = soup.new_tag("figure", attrs={"class": "figure"})
                
                img = soup.new_tag("img")
                img["src"] = element.get("local_path", element.get("src", "#"))
                if element.get("alt"):
                    img["alt"] = element["alt"]
                
                figure.append(img)
                
                if element.get("caption"):
                    figcaption = soup.new_tag("figcaption")
                    figcaption.string = element["caption"]
                    figure.append(figcaption)
                
                container.append(figure)
            else:
                img = soup.new_tag("img")
                img["src"] = element.get("local_path", element.get("src", "#"))
                if element.get("alt"):
                    img["alt"] = element["alt"]
                container.append(img)
                
        elif element_type == "blockquote":
            blockquote = soup.new_tag("blockquote")
            blockquote.string = element.get("text", "")
            container.append(blockquote)
            
        elif element_type == "heading":
            heading_level = min(6, max(1, element.get("level", 3)))
            heading = soup.new_tag(f"h{heading_level}")
            heading.string = element.get("text", "")
            container.append(heading)
            
    except Exception as e:
        logger.error(f"Ошибка при добавлении элемента типа {element.get('type', 'unknown')} в HTML: {str(e)}")
        logger.debug(f"Трассировка: {traceback.format_exc()}")


def _add_tech_specs_to_html(tech_specs: List[Dict[str, Any]], container: BeautifulSoup, soup: BeautifulSoup) -> None:
    """
    Добавляет технические характеристики в HTML-документ.
    
    Args:
        tech_specs: Список технических характеристик
        container: HTML-контейнер для добавления
        soup: Объект BeautifulSoup для создания новых элементов
    """
    try:
        if not tech_specs:
            return
            
        # Создаем блок для технических характеристик
        specs_div = soup.new_tag("div", attrs={"class": "technical-specs"})
        
        # Добавляем заголовок
        specs_heading = soup.new_tag("h3")
        specs_heading.string = "Технические характеристики"
        specs_div.append(specs_heading)
        
        # Создаем таблицу характеристик
        table = soup.new_tag("table")
        
        # Добавляем заголовки таблицы
        thead = soup.new_tag("thead")
        tr = soup.new_tag("tr")
        
        th_name = soup.new_tag("th")
        th_name.string = "Характеристика"
        tr.append(th_name)
        
        th_value = soup.new_tag("th")
        th_value.string = "Значение"
        tr.append(th_value)
        
        thead.append(tr)
        table.append(thead)
        
        # Добавляем строки с характеристиками
        tbody = soup.new_tag("tbody")
        
        for spec in tech_specs:
            tr = soup.new_tag("tr")
            
            td_name = soup.new_tag("td")
            td_name.string = spec.get("name", "")
            tr.append(td_name)
            
            td_value = soup.new_tag("td")
            td_value.string = spec.get("value", "")
            tr.append(td_value)
            
            tbody.append(tr)
        
        table.append(tbody)
        specs_div.append(table)
        
        # Добавляем блок в контейнер
        container.append(specs_div)
        
    except Exception as e:
        logger.error(f"Ошибка при добавлении технических характеристик в HTML: {str(e)}")
        logger.debug(f"Трассировка: {traceback.format_exc()}")


def _save_markdown(data: Dict[str, Any], file_path: Union[str, Path]) -> bool:
    """
    Создает и сохраняет Markdown-файл из структурированных данных.
    
    Args:
        data: Данные статьи
        file_path: Путь для сохранения файла
        
    Returns:
        bool: True в случае успешного сохранения, False в случае ошибки
    """
    try:
        md_content = []
        
        # Добавление заголовка и мета-информации
        md_content.append(f"# {data.get('title', 'Без заголовка')}\n")
        
        # Добавление хлебных крошек
        if "breadcrumbs" in data and data["breadcrumbs"]:
            md_content.append(f"*Категории: {' > '.join(data['breadcrumbs'])}*\n")
        
        if "parsed_at" in data:
            md_content.append(f"*Дата обработки: {data['parsed_at']}*\n")
        
        md_content.append(f"*Источник: {data.get('url', '')}*\n")
        
        # Рекурсивное добавление чанков
        _chunks_to_markdown(data.get("chunks", []), md_content)
        
        # Сохранение Markdown
        with open(file_path, 'w', encoding='utf-8') as md_file:
            md_file.write("\n".join(md_content))
        
        logger.debug(f"Markdown сохранен в {file_path}")
        return True
        
    except Exception as e:
        logger.error(f"Ошибка при сохранении Markdown: {str(e)}")
        logger.debug(f"Трассировка: {traceback.format_exc()}")
        return False


def _chunks_to_markdown(chunks: List[Dict[str, Any]], md_content: List[str], level: int = 1) -> None:
    """
    Рекурсивно конвертирует чанки в Markdown.
    
    Args:
        chunks: Список чанков для конвертации
        md_content: Список строк Markdown для добавления содержимого
        level: Текущий уровень вложенности (для вычисления уровня заголовков)
    """
    for chunk in chunks:
        try:
            # Добавление заголовка раздела
            if chunk.get("title") and chunk["title"] != "Введение":
                # Заголовок с соответствующим количеством #
                heading_level = min(6, max(1, level + 1))
                md_content.append(f"\n{'#' * heading_level} {chunk['title']}\n")
            
            # Добавление меток, если они есть
            if chunk.get("labels"):
                md_content.append(f"*Метки: {', '.join(chunk['labels'])}*\n")
            
            # Добавление содержимого раздела
            _elements_to_markdown(chunk.get("elements", []), md_content)
            
            # Добавление технических характеристик, если они есть
            if "metadata" in chunk and "tech_specs" in chunk["metadata"] and chunk["metadata"]["tech_specs"]:
                _tech_specs_to_markdown(chunk["metadata"]["tech_specs"], md_content)
            
            # Добавление числовых данных, если они есть
            if chunk.get("numbers"):
                md_content.append("\n**Числовые данные:**")
                for num in chunk["numbers"]:
                    md_content.append(f"- {num.get('raw_text', '')}")
                md_content.append("")
            
            # Добавление исходящих ссылок, если они есть
            if chunk.get("outgoing_links"):
                md_content.append("\n**Связанные статьи:**")
                for link in chunk["outgoing_links"]:
                    md_content.append(f"- [{link}]({link})")
                md_content.append("")
            
            # Рекурсивно добавляем дочерние чанки
            if chunk.get("children"):
                _chunks_to_markdown(chunk["children"], md_content, level + 1)
                
        except Exception as e:
            logger.error(f"Ошибка при конвертации чанка {chunk.get('id', 'unknown')} в Markdown: {str(e)}")
            logger.debug(f"Трассировка: {traceback.format_exc()}")


def _elements_to_markdown(elements: List[Dict[str, Any]], md_content: List[str]) -> None:
    """
    Конвертирует элементы контента в Markdown.
    
    Args:
        elements: Список элементов контента
        md_content: Список строк Markdown для добавления содержимого
    """
    try:
        for element in elements:
            element_type = element.get("type", "")
            
            if element_type == "paragraph":
                md_content.append(f"{element.get('text', '')}\n")
                
            elif element_type == "list":
                md_content.append("")
                for item in element.get("items", []):
                    prefix = "1. " if element.get("list_type") == "ordered" else "- "
                    md_content.append(f"{prefix}{item}")
                md_content.append("")
                
            elif element_type == "table":
                # Добавление заголовков таблицы
                if element.get("headers"):
                    md_content.append("\n| " + " | ".join(element["headers"]) + " |")
                    md_content.append("| " + " | ".join(["---"] * len(element["headers"])) + " |")
                
                # Добавление данных таблицы
                for row in element.get("rows", []):
                    md_content.append("| " + " | ".join(row) + " |")
                
                md_content.append("")
                
            elif element_type in ["image", "figure"]:
                alt_text = element.get("alt", "")
                caption = element.get("caption", "")
                
                image_path = element.get("local_path", element.get("src", "#"))
                
                # В Markdown изображение с подписью
                md_content.append(f"\n![{alt_text}]({image_path})")
                
                if caption:
                    md_content.append(f"*{caption}*\n")
                    
            elif element_type == "blockquote":
                text = element.get("text", "")
                lines = text.split("\n")
                
                md_content.append("")
                for line in lines:
                    md_content.append(f"> {line}")
                md_content.append("")
                
            elif element_type == "heading":
                heading_level = min(6, max(1, element.get("level", 3)))
                md_content.append(f"\n{'#' * heading_level} {element.get('text', '')}\n")
                
    except Exception as e:
        logger.error(f"Ошибка при конвертации элементов в Markdown: {str(e)}")
        logger.debug(f"Трассировка: {traceback.format_exc()}")


def _tech_specs_to_markdown(tech_specs: List[Dict[str, Any]], md_content: List[str]) -> None:
    """
    Конвертирует технические характеристики в Markdown.
    
    Args:
        tech_specs: Список технических характеристик
        md_content: Список строк Markdown для добавления содержимого
    """
    try:
        if not tech_specs:
            return
            
        md_content.append("\n### Технические характеристики\n")
        
        # Создаем заголовки таблицы
        md_content.append("| Характеристика | Значение |")
        md_content.append("|---------------|----------|")
        
        # Добавляем строки с характеристиками
        for spec in tech_specs:
            name = spec.get("name", "").replace("|", "\\|")
            value = spec.get("value", "").replace("|", "\\|")
            md_content.append(f"| {name} | {value} |")
        
        md_content.append("")
        
    except Exception as e:
        logger.error(f"Ошибка при конвертации технических характеристик в Markdown: {str(e)}")
        logger.debug(f"Трассировка: {traceback.format_exc()}")


def _save_rag_metadata(data: Dict[str, Any], file_path: Union[str, Path]) -> bool:
    """
    Сохраняет оптимизированные метаданные для RAG-системы.
    
    Args:
        data: Данные статьи
        file_path: Путь для сохранения файла
        
    Returns:
        bool: True в случае успешного сохранения, False в случае ошибки
    """
    try:
        # Извлечение основных метаданных
        metadata = {
            "title": data.get("title", ""),
            "url": data.get("url", ""),
            "breadcrumbs": data.get("breadcrumbs", []),
            "parsed_at": data.get("parsed_at", datetime.now().isoformat()),
            "chunk_count": 0,
            "paragraph_count": 0,
            "labels": set(),
            "numerical_data": {
                "units": set(),
                "has_values": False
            },
            "tech_specs": [],
            "content_types": set(),
            "rag_version": "1.0"
        }
        
        # Обработка чанков
        flat_chunks = _flatten_chunks(data.get("chunks", []))
        metadata["chunk_count"] = len(flat_chunks)
        
        # Сбор статистики по всем чанкам
        all_paragraphs = []
        
        for chunk in flat_chunks:
            # Сбор меток
            if chunk.get("labels"):
                metadata["labels"].update(chunk["labels"])
            
            # Сбор параграфов
            if chunk.get("paragraphs"):
                all_paragraphs.extend(chunk["paragraphs"])
            
            # Сбор числовых данных
            if chunk.get("numbers"):
                metadata["numerical_data"]["has_values"] = True
                for num in chunk["numbers"]:
                    if "unit" in num:
                        metadata["numerical_data"]["units"].add(num["unit"])
            
            # Сбор технических характеристик
            if "metadata" in chunk and "tech_specs" in chunk["metadata"]:
                metadata["tech_specs"].extend(chunk["metadata"]["tech_specs"])
            
            # Сбор типов контента
            if "metadata" in chunk and "content_types" in chunk["metadata"]:
                metadata["content_types"].update(chunk["metadata"]["content_types"])
        
        # Добавление количества параграфов
        metadata["paragraph_count"] = len(all_paragraphs)
        
        # Преобразование множеств в списки для сериализации
        metadata["labels"] = list(metadata["labels"])
        metadata["numerical_data"]["units"] = list(metadata["numerical_data"]["units"])
        metadata["content_types"] = list(metadata["content_types"])
        
        # Ограничение количества технических характеристик
        if len(metadata["tech_specs"]) > 100:
            metadata["tech_specs"] = metadata["tech_specs"][:100]
        
        # Сохранение метаданных
        with open(file_path, 'w', encoding='utf-8') as f:
            json.dump(metadata, f, ensure_ascii=False, indent=2)
        
        logger.debug(f"Метаданные RAG сохранены в {file_path}")
        return True
        
    except Exception as e:
        logger.error(f"Ошибка при сохранении метаданных RAG: {str(e)}")
        logger.debug(f"Трассировка: {traceback.format_exc()}")
        return False


def _flatten_chunks(chunks: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
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


def _prepare_data_for_serialization(data: Dict[str, Any]) -> Dict[str, Any]:
    """
    Подготавливает данные для сериализации, удаляя или преобразуя
    элементы, которые могут вызвать проблемы.
    
    Args:
        data: Исходные данные
        
    Returns:
        Dict[str, Any]: Подготовленные данные
    """
    try:
        # Создаем глубокую копию данных
        prepared_data = {}
        
        # Копируем основные метаданные
        for key in ["title", "url", "slug", "breadcrumbs", "parsed_at"]:
            if key in data:
                prepared_data[key] = data[key]
        
        # Обработка метаданных
        if "metadata" in data:
            prepared_data["metadata"] = data["metadata"]
        
        # Обработка чанков
        if "chunks" in data:
            prepared_data["chunks"] = _prepare_chunks_for_serialization(data["chunks"])
        
        return prepared_data
        
    except Exception as e:
        logger.error(f"Ошибка при подготовке данных для сериализации: {str(e)}")
        logger.debug(f"Трассировка: {traceback.format_exc()}")
        # В случае ошибки возвращаем исходные данные
        return data


def _prepare_chunks_for_serialization(chunks: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """
    Подготавливает чанки для сериализации.
    
    Args:
        chunks: Список чанков
        
    Returns:
        List[Dict[str, Any]]: Подготовленные чанки
    """
    prepared_chunks = []
    
    try:
        for chunk in chunks:
            prepared_chunk = {}
            
            # Копируем основные поля
            for key in ["id", "title", "level", "labels", "numbers", "outgoing_links", "images"]:
                if key in chunk:
                    prepared_chunk[key] = chunk[key]
            
            # Копируем метаданные, если они есть
            if "metadata" in chunk:
                prepared_chunk["metadata"] = chunk["metadata"]
            
            # Обработка элементов содержимого
            if "elements" in chunk:
                prepared_elements = []
                for element in chunk["elements"]:
                    # Для элементов удаляем лишние поля и преобразуем при необходимости
                    prepared_element = {key: value for key, value in element.items() 
                                    if key not in ["html", "soup"]}
                    prepared_elements.append(prepared_element)
                prepared_chunk["elements"] = prepared_elements
            
            # Копируем параграфы
            if "paragraphs" in chunk:
                prepared_chunk["paragraphs"] = chunk["paragraphs"]
            
            # Копируем словарь меток параграфов
            if "paragraph_labels" in chunk:
                prepared_chunk["paragraph_labels"] = chunk["paragraph_labels"]
            
            # Рекурсивная обработка дочерних чанков
            if "children" in chunk and chunk["children"]:
                prepared_chunk["children"] = _prepare_chunks_for_serialization(chunk["children"])
            else:
                prepared_chunk["children"] = []
            
            prepared_chunks.append(prepared_chunk)
            
    except Exception as e:
        logger.error(f"Ошибка при подготовке чанков для сериализации: {str(e)}")
        logger.debug(f"Трассировка: {traceback.format_exc()}")
    
    return prepared_chunks