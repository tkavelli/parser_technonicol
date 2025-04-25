"""
Основной модуль, координирующий весь процесс обработки статей.
Поддерживает параллельную обработку и отслеживание прогресса.
"""
import os
import sys
import json
import signal
import logging
import argparse
import pandas as pd
import time
import traceback
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
from typing import List, Dict, Any, Optional, Tuple
from datetime import datetime

# Добавляем путь к текущей директории в PYTHONPATH
sys.path.append(str(Path(__file__).parent))

# Импорт компонентов пайплайна
from fetcher import fetch_article
from parser import parse_html
from chunker import create_chunks
from extractor import extract_data
from serializer import serialize_article
from db import init_db, save_to_db
from vectorizer import vectorize_article
from state_manager import StateManager
from utils import setup_logging, get_article_slug, ensure_dir
from text_utils import log_error, extract_breadcrumbs_from_url

# Константы
OUTPUT_DIR = Path("output")
DATA_DIR = Path("data")
LOGS_DIR = OUTPUT_DIR / "logs"
DB_PATH = OUTPUT_DIR / "articles.db"
PROGRESS_FILE = OUTPUT_DIR / "progress.json"
DEBUG_DIR = OUTPUT_DIR / "debug"
DEFAULT_MAX_WORKERS = 4

# Настройка логирования
logger = setup_logging()


class ProcessingManager:
    """Менеджер обработки статей с отслеживанием прогресса и возможностью возобновления."""
    
    def __init__(self, input_file: str, max_workers: int = DEFAULT_MAX_WORKERS, 
                 force_restart: bool = False, skip_vectorization: bool = False,
                 enable_debug: bool = True):
        """
        Инициализация менеджера обработки.
        
        Args:
            input_file: Путь к Excel-файлу со списком статей
            max_workers: Максимальное количество параллельных процессов
            force_restart: Принудительно начать обработку заново для всех статей
            skip_vectorization: Пропустить этап векторизации (для отладки)
            enable_debug: Включить сохранение отладочных данных
        """
        self.input_file = input_file
        self.max_workers = max_workers
        self.force_restart = force_restart
        self.skip_vectorization = skip_vectorization
        self.enable_debug = enable_debug
        self.stop_requested = False
        
        # Настройка обработчика сигналов для корректного завершения
        signal.signal(signal.SIGINT, self.handle_shutdown)
        signal.signal(signal.SIGTERM, self.handle_shutdown)
        
        # Создание необходимых директорий
        self._create_required_directories()
        
        # Инициализация менеджера состояния
        self.state_manager = StateManager(PROGRESS_FILE)
        
        # Инициализация базы данных
        init_db(DB_PATH)
        
        logger.info(f"ProcessingManager инициализирован: input={input_file}, "
                   f"workers={max_workers}, force_restart={force_restart}, "
                   f"skip_vectorization={skip_vectorization}, debug={enable_debug}")
    
    def _create_required_directories(self):
        """Создает необходимые директории для работы пайплайна."""
        ensure_dir(OUTPUT_DIR)
        ensure_dir(DATA_DIR)
        ensure_dir(LOGS_DIR)
        ensure_dir(DATA_DIR / "html")
        ensure_dir(DATA_DIR / "images")
        
        if self.enable_debug:
            ensure_dir(DEBUG_DIR)
    
    def handle_shutdown(self, signum, frame):
        """Обработчик сигналов завершения для корректного завершения работы."""
        logger.info(f"Получен сигнал завершения {signum}. Завершаем текущие задачи...")
        self.stop_requested = True
    
    def save_debug_snapshot(self, article_data: Dict[str, Any], stage: str) -> None:
        """
        Сохраняет промежуточные данные для отладки.
        
        Args:
            article_data: Данные статьи
            stage: Название этапа обработки
        """
        if not self.enable_debug:
            return
            
        try:
            slug = article_data.get("slug", "unknown")
            debug_dir = DEBUG_DIR / slug / stage
            ensure_dir(debug_dir)
            
            # Подготовка данных для сериализации
            snapshot = {
                "timestamp": datetime.now().isoformat(),
                "stage": stage,
                "article_url": article_data.get("url", ""),
                "article_title": article_data.get("title", "")
            }
            
            # Сохраняем метаданные
            with open(debug_dir / "metadata.json", "w", encoding="utf-8") as f:
                json.dump(snapshot, f, ensure_ascii=False, indent=2)
            
            # Сохраняем полные данные статьи
            try:
                with open(debug_dir / "data.json", "w", encoding="utf-8") as f:
                    json.dump(article_data, f, ensure_ascii=False, indent=2, default=str)
            except Exception as e:
                logger.warning(f"Не удалось сохранить полные данные: {str(e)}")
                # Сохраняем основные поля в случае ошибки
                simple_data = {
                    "url": article_data.get("url", ""),
                    "title": article_data.get("title", ""),
                    "slug": slug,
                    "breadcrumbs": article_data.get("breadcrumbs", []),
                    "parsed_at": article_data.get("parsed_at", "")
                }
                with open(debug_dir / "data_simple.json", "w", encoding="utf-8") as f:
                    json.dump(simple_data, f, ensure_ascii=False, indent=2)
            
            logger.debug(f"Создан отладочный снимок: {debug_dir}")
            
        except Exception as e:
            logger.warning(f"Не удалось сохранить отладочный снимок: {str(e)}")
    
    def load_articles(self) -> List[Dict[str, Any]]:
        """
        Загрузка списка статей из Excel-файла.
        
        Returns:
            List[Dict[str, Any]]: Список статей с их URL и метаданными
        """
        try:
            logger.info(f"Загрузка списка статей из {self.input_file}")
            
            # Чтение Excel-файла
            df = pd.read_excel(self.input_file)
            
            # Проверка наличия необходимых колонок
            if 'URL' not in df.columns:
                error_msg = "В Excel-файле отсутствует обязательная колонка 'URL'"
                logger.error(error_msg)
                raise ValueError(error_msg)
            
            # Преобразование в список словарей
            articles = []
            for _, row in df.iterrows():
                url = row['URL']
                
                # Пропуск пустых URL
                if pd.isna(url) or not url.strip():
                    continue
                
                # Создание записи о статье
                article = {
                    "url": url.strip(),
                    "title": row.get('Title', '') if 'Title' in df.columns and not pd.isna(row.get('Title', '')) else '',
                    "status": "not_started"
                }
                
                # Генерация slug из URL
                article["slug"] = get_article_slug(url)
                
                # Извлечение хлебных крошек из URL
                article["breadcrumbs_from_url"] = extract_breadcrumbs_from_url(url)
                
                articles.append(article)
            
            # Удаление дубликатов по URL
            unique_articles = self._deduplicate_articles(articles)
            
            logger.info(f"Загружено {len(unique_articles)} уникальных статей")
            return unique_articles
            
        except Exception as e:
            context = {"input_file": self.input_file}
            log_error(logger, "Ошибка при загрузке статей", e, context)
            raise
    
    def _deduplicate_articles(self, articles: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Удаляет дубликаты статей по URL.
        
        Args:
            articles: Список статей с возможными дубликатами
            
        Returns:
            List[Dict[str, Any]]: Список уникальных статей
        """
        unique_articles = []
        seen_urls = set()
        
        for article in articles:
            if article["url"] not in seen_urls:
                seen_urls.add(article["url"])
                unique_articles.append(article)
            else:
                logger.warning(f"Обнаружен дубликат URL: {article['url']} (пропущен)")
        
        return unique_articles
    
    def _fetch_and_parse_article(self, url: str, slug: str, html_path: Path) -> Tuple[List[Dict[str, Any]], str, List[str]]:
        """
        Загружает и парсит HTML-контент статьи.
        
        Args:
            url: URL статьи
            slug: Slug статьи
            html_path: Путь для сохранения HTML
            
        Returns:
            Tuple[List[Dict[str, Any]], str, List[str]]: Контент, заголовок и хлебные крошки
        """
        start_time = time.time()
        logger.info(f"[{slug}] Загрузка HTML...")
        
        html_content = fetch_article(url, html_path)
        download_time = time.time() - start_time
        logger.info(f"[{slug}] Загрузка HTML выполнена за {download_time:.2f} сек")
        
        if not html_content:
            error_msg = f"Получен пустой HTML-контент для {url}"
            logger.error(error_msg)
            raise ValueError(error_msg)
        
        start_time = time.time()
        logger.info(f"[{slug}] Парсинг HTML...")
        
        parsed_content, extracted_title, breadcrumbs = parse_html(html_content, url)
        parsing_time = time.time() - start_time
        logger.info(f"[{slug}] Парсинг HTML выполнен за {parsing_time:.2f} сек. "
                   f"Извлечено элементов: {len(parsed_content)}, "
                   f"заголовок: '{extracted_title}', хлебные крошки: {breadcrumbs}")
        
        if not parsed_content:
            logger.warning(f"[{slug}] Парсинг HTML не дал результатов. Проверьте HTML и селекторы.")
        
        return parsed_content, extracted_title, breadcrumbs
    
    def _process_content(self, parsed_content: List[Dict[str, Any]], slug: str, url: str, 
                        images_dir: Path, breadcrumbs: List[str] = None) -> List[Dict[str, Any]]:
        """
        Обрабатывает контент статьи: создает чанки и извлекает данные.
        
        Args:
            parsed_content: Контент статьи из парсера
            slug: Slug статьи
            url: URL статьи
            images_dir: Директория для изображений
            breadcrumbs: Хлебные крошки статьи
            
        Returns:
            List[Dict[str, Any]]: Обогащенные чанки статьи
        """
        # Создание чанков
        start_time = time.time()
        logger.info(f"[{slug}] Создание чанков...")
        
        chunks = create_chunks(parsed_content, slug, breadcrumbs)
        chunking_time = time.time() - start_time
        
        chunks_count = len(chunks)
        logger.info(f"[{slug}] Создание чанков выполнено за {chunking_time:.2f} сек. "
                   f"Создано чанков: {chunks_count}")
        
        if not chunks:
            logger.warning(f"[{slug}] Создание чанков не дало результатов. Проверьте parsed_content.")
        
        # Извлечение данных
        start_time = time.time()
        logger.info(f"[{slug}] Извлечение данных из чанков...")
        
        enriched_chunks = extract_data(chunks, url, images_dir)
        extraction_time = time.time() - start_time
        
        from text_utils import flatten_chunks
        flat_chunks = flatten_chunks(enriched_chunks)
        total_paragraphs = sum(len(chunk.get("paragraphs", [])) for chunk in flat_chunks)
        total_numbers = sum(len(chunk.get("numbers", [])) for chunk in flat_chunks)
        
        logger.info(f"[{slug}] Извлечение данных выполнено за {extraction_time:.2f} сек. "
                   f"Чанков: {len(flat_chunks)}, параграфов: {total_paragraphs}, "
                   f"числовых данных: {total_numbers}")
        
        return enriched_chunks
    
    def process_article(self, article: Dict[str, Any]) -> Tuple[Dict[str, Any], bool]:
        """
        Обработка одной статьи.
        
        Выполняет все этапы обработки: загрузка HTML, парсинг, создание чанков,
        извлечение данных, сериализация, сохранение в БД и векторизация.
        
        Args:
            article: Словарь с информацией о статье
            
        Returns:
            Tuple[Dict, bool]: Обновленная информация о статье и флаг успешности
        """
        url = article["url"]
        slug = article["slug"]
        title = article["title"]
        
        article_dir = OUTPUT_DIR / slug
        html_path = DATA_DIR / "html" / f"{slug}.html"
        images_dir = DATA_DIR / "images" / slug
        
        # Общее время обработки
        total_start_time = time.time()
        logger.info(f"Начало обработки статьи: {url}")
        
        try:
            # 1. Загрузка и парсинг HTML
            parsed_content, extracted_title, breadcrumbs = self._fetch_and_parse_article(url, slug, html_path)
            
            # Объединяем хлебные крошки из URL и HTML, если нужно
            breadcrumbs_from_url = article.get("breadcrumbs_from_url", [])
            if not breadcrumbs and breadcrumbs_from_url:
                logger.info(f"[{slug}] Используем хлебные крошки из URL: {breadcrumbs_from_url}")
                breadcrumbs = breadcrumbs_from_url
            
            # Обновляем заголовок, если он не был указан в Excel
            if not title and extracted_title:
                title = extracted_title
                article["title"] = title
            
            # 2. Обработка контента
            enriched_chunks = self._process_content(parsed_content, slug, url, images_dir, breadcrumbs)
            
            # 3. Подготовка полных данных статьи
            current_time = datetime.now().strftime("%Y-%m-%dT%H:%M:%S")
            article_data = {
                "title": title,
                "url": url,
                "slug": slug,
                "chunks": enriched_chunks,
                "breadcrumbs": breadcrumbs,
                "parsed_at": current_time
            }
            
            # Сохранение промежуточных результатов для отладки
            self.save_debug_snapshot(article_data, "after_extraction")
            
            # 4. Сериализация в YAML/JSON
            serialization_start = time.time()
            logger.info(f"[{slug}] Сериализация в YAML/JSON...")
            
            serialize_article(article_data, article_dir)
            
            serialization_time = time.time() - serialization_start
            logger.info(f"[{slug}] Сериализация выполнена за {serialization_time:.2f} сек.")
            
            # 5. Сохранение в базу данных
            db_start = time.time()
            logger.info(f"[{slug}] Сохранение в базу данных...")
            
            article_id = save_to_db(article_data, DB_PATH)
            
            db_time = time.time() - db_start
            logger.info(f"[{slug}] Сохранение в БД выполнено за {db_time:.2f} сек. ID статьи: {article_id}")
            
            # 6. Векторизация (если не отключена)
            if not self.skip_vectorization:
                vectorization_start = time.time()
                logger.info(f"[{slug}] Векторизация чанков...")
                
                chunks_indexed, paragraphs_indexed = vectorize_article(
                    enriched_chunks, article_id, title, url, breadcrumbs
                )
                
                vectorization_time = time.time() - vectorization_start
                logger.info(f"[{slug}] Векторизация выполнена за {vectorization_time:.2f} сек. "
                          f"Проиндексировано чанков: {chunks_indexed}, параграфов: {paragraphs_indexed}")
            
            total_time = time.time() - total_start_time
            logger.info(f"Обработка статьи '{title}' ({url}) завершена за {total_time:.2f} сек.")
            
            article["status"] = "done"
            article["processing_time"] = total_time
            return article, True
            
        except Exception as e:
            total_time = time.time() - total_start_time
            context = {
                "url": url,
                "slug": slug,
                "title": title,
                "time_spent": f"{total_time:.2f} сек"
            }
            log_error(logger, f"Ошибка при обработке статьи", e, context)
            
            article["status"] = "error"
            article["error"] = str(e)
            article["processing_time"] = total_time
            return article, False
    
    def _prepare_articles_for_processing(self, all_articles: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Подготавливает список статей для обработки, учитывая текущий прогресс.
        
        Args:
            all_articles: Полный список статей
            
        Returns:
            List[Dict[str, Any]]: Список статей для обработки
        """
        # Загрузка текущего прогресса
        current_progress = self.state_manager.load_progress()
        
        # Обновление статусов из прогресса, если не требуется принудительный перезапуск
        if not self.force_restart and current_progress:
            progress_by_url = {a["url"]: a for a in current_progress if "url" in a}
            
            for article in all_articles:
                if article["url"] in progress_by_url:
                    # Сохраняем текущий статус из файла прогресса
                    progress_item = progress_by_url[article["url"]]
                    current_status = progress_item.get("status")
                    
                    if current_status == "done":
                        logger.info(f"Статья уже обработана: {article['url']} (статус: {current_status})")
                        article["status"] = current_status
                        # Копируем дополнительные поля из прогресса, если они есть
                        for key in ["processing_time", "processed_at"]:
                            if key in progress_item:
                                article[key] = progress_item[key]
                    elif current_status == "in_progress":
                        # Статья была прервана во время обработки, начнем заново
                        logger.warning(f"Статья была прервана: {article['url']} (будет обработана заново)")
                        article["status"] = "not_started"
                    elif current_status == "error":
                        logger.warning(f"Статья завершилась с ошибкой: {article['url']} (будет предпринята новая попытка)")
                        article["status"] = "not_started"
                        if "error" in progress_item:
                            logger.warning(f"Предыдущая ошибка: {progress_item['error']}")
        
        # Отбор статей для обработки (те, что не помечены как "done")
        articles_to_process = [a for a in all_articles if a["status"] != "done"]
        
        # Если включен режим принудительного перезапуска, все статьи будут обработаны заново
        if self.force_restart:
            logger.info("Включен режим принудительного перезапуска - все статьи будут обработаны заново")
            return all_articles
            
        return articles_to_process
    
    def _log_processing_statistics(self, all_articles: List[Dict[str, Any]], 
                                 articles_to_process: List[Dict[str, Any]]):
        """
        Выводит статистику по обработке статей.
        
        Args:
            all_articles: Полный список статей
            articles_to_process: Список статей для обработки
        """
        logger.info("=" * 50)
        logger.info(f"Всего статей: {len(all_articles)}")
        logger.info(f"Уже обработано: {len(all_articles) - len(articles_to_process)}")
        logger.info(f"Осталось обработать: {len(articles_to_process)}")
        
        # Вывести статистику времени обработки, если есть обработанные статьи
        processed_times = [a.get("processing_time", 0) for a in all_articles if a.get("status") == "done" and "processing_time" in a]
        if processed_times:
            avg_time = sum(processed_times) / len(processed_times)
            min_time = min(processed_times)
            max_time = max(processed_times)
            logger.info(f"Среднее время обработки: {avg_time:.2f} сек (мин: {min_time:.2f}, макс: {max_time:.2f})")
            
            # Приблизительное время окончания
            if articles_to_process:
                eta = avg_time * len(articles_to_process) / self.max_workers
                eta_minutes = int(eta / 60)
                eta_seconds = int(eta % 60)
                logger.info(f"Приблизительное время до завершения: {eta_minutes} мин {eta_seconds} сек")
        
        logger.info("=" * 50)
    
    def _process_futures_results(self, futures_dict, total_articles, all_articles):
        """
        Обрабатывает результаты выполнения задач в пуле процессов.
        
        Args:
            futures_dict: Словарь {future: article}
            total_articles: Общее количество статей для обработки
            all_articles: Полный список статей
            
        Returns:
            Tuple[int, int, int]: Количество обработанных, успешных и неудачных статей
        """
        processed = 0
        successful = 0
        failed = 0
        
        for future in as_completed(futures_dict):
            article = futures_dict[future]
            url = article.get("url", "unknown")
            
            try:
                updated_article, success = future.result()
                processed += 1
                
                if success:
                    successful += 1
                    # Добавляем дату и время успешной обработки
                    updated_article["processed_at"] = datetime.now().isoformat()
                else:
                    failed += 1
                
                # Обновление прогресса
                self.state_manager.update_article(updated_article)
                
                # Вывод прогресса
                self._log_progress(processed, total_articles, successful, failed)
                
            except Exception as e:
                failed += 1
                processed += 1
                
                context = {"url": url, "slug": article.get("slug", "unknown")}
                log_error(logger, "Необработанная ошибка в процессе обработки статьи", e, context)
                
                # Обновление статуса с ошибкой
                for a in all_articles:
                    if a.get("url") == url:
                        a["status"] = "error"
                        a["error"] = f"Неперехваченное исключение: {str(e)}"
                        self.state_manager.update_article(a)
                        break
                
                # Вывод прогресса
                self._log_progress(processed, total_articles, successful, failed)
        
        return processed, successful, failed
    
    def _log_progress(self, processed: int, total: int, successful: int, failed: int):
        """
        Выводит информацию о текущем прогрессе обработки.
        
        Args:
            processed: Количество обработанных статей
            total: Общее количество статей
            successful: Количество успешно обработанных статей
            failed: Количество статей с ошибками
        """
        progress_percent = (processed / total) * 100 if total > 0 else 0
        logger.info(f"Прогресс: {processed}/{total} ({progress_percent:.1f}%) - "
                   f"Успешно: {successful}, Ошибок: {failed}")
    
    def _log_final_statistics(self, processed: int, total: int, successful: int, failed: int,
                             start_time: float):
        """
        Выводит итоговую статистику обработки.
        
        Args:
            processed: Количество обработанных статей
            total: Общее количество статей
            successful: Количество успешно обработанных статей
            failed: Количество статей с ошибками
            start_time: Время начала обработки
        """
        total_time = time.time() - start_time
        minutes = int(total_time // 60)
        seconds = int(total_time % 60)
        
        logger.info("=" * 50)
        logger.info("Обработка завершена!")
        logger.info(f"Всего обработано: {processed}/{total}")
        logger.info(f"Успешно: {successful}")
        logger.info(f"С ошибками: {failed}")
        logger.info(f"Общее время: {minutes} мин {seconds} сек")
        
        # Статистика скорости обработки
        if processed > 0:
            avg_time = total_time / processed
            logger.info(f"Среднее время на статью: {avg_time:.2f} сек")
        logger.info("=" * 50)
    
    def run(self):
        """Запуск полного пайплайна обработки статей."""
        start_time = time.time()
        
        try:
            # Загрузка списка статей
            all_articles = self.load_articles()
            
            # Подготовка статей для обработки
            articles_to_process = self._prepare_articles_for_processing(all_articles)
            
            # Вывод статистики
            self._log_processing_statistics(all_articles, articles_to_process)
            
            if not articles_to_process:
                logger.info("Все статьи уже обработаны!")
                return
            
            # Инициализация счетчиков
            total = len(articles_to_process)
            processed = 0
            successful = 0
            failed = 0
            
            # Параллельная обработка статей
            logger.info(f"Запуск параллельной обработки с {self.max_workers} процессами...")
            
            with ProcessPoolExecutor(max_workers=self.max_workers) as executor:
                # Отправка статей на обработку
                futures = {}
                for article in articles_to_process:
                    # Пропуск, если получен сигнал остановки
                    if self.stop_requested:
                        logger.info("Получен сигнал остановки. Прекращаем добавление новых задач.")
                        break
                    
                    # Пометка статьи как "in_progress"
                    article["status"] = "in_progress"
                    self.state_manager.update_article(article)
                    
                    # Отправка на обработку
                    future = executor.submit(self.process_article, article.copy())
                    futures[future] = article
                
                # Обработка результатов
                processed, successful, failed = self._process_futures_results(futures, total, all_articles)
            
            # Итоговая статистика
            self._log_final_statistics(processed, total, successful, failed, start_time)
            
        except KeyboardInterrupt:
            logger.info("Обработка прервана пользователем.")
            # Сколько статей успели обработать
            articles_done = len([a for a in all_articles if a.get("status") == "done"])
            articles_error = len([a for a in all_articles if a.get("status") == "error"])
            logger.info(f"Успешно обработано: {articles_done}, с ошибками: {articles_error}")
            
        except Exception as e:
            log_error(logger, "Критическая ошибка при выполнении пайплайна", e)
            raise
        finally:
            # Всегда выводим время работы
            total_time = time.time() - start_time
            minutes = int(total_time // 60)
            seconds = int(total_time % 60)
            logger.info(f"Общее время работы: {minutes} мин {seconds} сек")


if __name__ == "__main__":
    # Парсинг аргументов командной строки
    parser = argparse.ArgumentParser(description="Пайплайн обработки статей")
    parser.add_argument("--input", "-i", type=str, default="articles.xlsx", 
                        help="Путь к Excel-файлу со списком статей")
    parser.add_argument("--workers", "-w", type=int, default=DEFAULT_MAX_WORKERS, 
                        help=f"Количество параллельных процессов (по умолчанию: {DEFAULT_MAX_WORKERS})")
    parser.add_argument("--force-restart", "-f", action="store_true", 
                        help="Принудительно начать обработку заново для всех статей")
    parser.add_argument("--skip-vectorization", "-s", action="store_true", 
                        help="Пропустить этап векторизации (для отладки)")
    parser.add_argument("--disable-debug", "-d", action="store_true",
                        help="Отключить сохранение отладочных данных")
    parser.add_argument("--log-level", "-l", choices=["DEBUG", "INFO", "WARNING", "ERROR"], 
                        default="INFO", help="Уровень логирования")
    
    args = parser.parse_args()
    
    # Установка уровня логирования
    numeric_level = getattr(logging, args.log_level.upper(), None)
    if numeric_level:
        logger.setLevel(numeric_level)
        # Обновление уровня для существующих обработчиков
        for handler in logger.handlers:
            handler.setLevel(numeric_level)
    
    # Запуск пайплайна
    manager = ProcessingManager(
        input_file=args.input, 
        max_workers=args.workers, 
        force_restart=args.force_restart, 
        skip_vectorization=args.skip_vectorization,
        enable_debug=not args.disable_debug
    )
    
    manager.run()