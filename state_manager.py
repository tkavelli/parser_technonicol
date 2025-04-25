"""
Модуль для управления состоянием и прогрессом обработки статей.
Позволяет сохранять и восстанавливать прогресс выполнения пайплайна.
"""
import os
import json
import logging
import threading
from pathlib import Path
from typing import List, Dict, Any, Optional, Union, Set

# Настройка логирования
logger = logging.getLogger(__name__)


class StateManager:
    """
    Менеджер состояния для отслеживания прогресса обработки статей.
    
    Обеспечивает потокобезопасную работу с файлом прогресса и предоставляет
    методы для управления статусами обработки статей.
    """
    
    def __init__(self, progress_file: Union[str, Path]):
        """
        Инициализация менеджера состояния.
        
        Args:
            progress_file: Путь к файлу прогресса
        """
        self.progress_file = Path(progress_file)
        self.lock = threading.Lock()  # Блокировка для безопасности в многопоточной среде
        
        # Создание директории для файла прогресса, если не существует
        os.makedirs(self.progress_file.parent, exist_ok=True)
        
        # Инициализация файла прогресса, если он не существует
        if not self.progress_file.exists():
            self._save_progress([])
    
    def load_progress(self) -> List[Dict[str, Any]]:
        """
        Загружает текущий прогресс обработки.
        
        Returns:
            List[Dict[str, Any]]: Список статей с их статусами
        """
        with self.lock:
            try:
                if not self.progress_file.exists():
                    logger.info(f"Файл прогресса не найден: {self.progress_file}")
                    return []
                
                with open(self.progress_file, 'r', encoding='utf-8') as f:
                    progress = json.load(f)
                
                logger.info(f"Загружены данные о прогрессе: {len(progress)} статей")
                return progress
                
            except Exception as e:
                logger.error(f"Ошибка при загрузке прогресса: {str(e)}")
                return []
    
    def _save_progress(self, progress: List[Dict[str, Any]]) -> bool:
        """
        Сохраняет прогресс в файл.
        
        Args:
            progress: Список статей с их статусами
            
        Returns:
            bool: True, если сохранение успешно
        """
        try:
            with open(self.progress_file, 'w', encoding='utf-8') as f:
                json.dump(progress, f, ensure_ascii=False, indent=2)
            
            return True
            
        except Exception as e:
            logger.error(f"Ошибка при сохранении прогресса: {str(e)}")
            return False
    
    def update_article(self, article: Dict[str, Any]) -> bool:
        """
        Обновляет статус статьи в прогрессе.
        
        Args:
            article: Информация о статье с обновленным статусом
            
        Returns:
            bool: True, если обновление успешно
        """
        if "url" not in article:
            logger.error("Невозможно обновить статью без URL")
            return False
            
        with self.lock:
            try:
                # Загрузка текущего прогресса
                progress = self.load_progress()
                
                # Поиск статьи в прогрессе
                found = False
                for idx, item in enumerate(progress):
                    if item.get("url") == article.get("url"):
                        # Обновление существующей статьи
                        progress[idx] = article
                        found = True
                        break
                
                # Добавление новой статьи, если не найдена
                if not found:
                    progress.append(article)
                
                # Сохранение обновленного прогресса
                result = self._save_progress(progress)
                
                if result:
                    article_title = article.get('title', article.get('url', 'Неизвестная статья'))
                    article_status = article.get('status', 'неизвестно')
                    logger.debug(f"Статус статьи '{article_title}' обновлен на '{article_status}'")
                
                return result
                
            except Exception as e:
                logger.error(f"Ошибка при обновлении статуса статьи: {str(e)}")
                return False
    
    def update_multiple_articles(self, articles: List[Dict[str, Any]]) -> bool:
        """
        Обновляет статусы нескольких статей в прогрессе.
        
        Args:
            articles: Список статей с обновленными статусами
            
        Returns:
            bool: True, если обновление успешно
        """
        if not articles:
            logger.warning("Передан пустой список статей для обновления")
            return True
            
        with self.lock:
            try:
                # Загрузка текущего прогресса
                progress = self.load_progress()
                
                # Создание словаря URL -> статья для быстрого поиска
                articles_by_url = {article.get("url"): article 
                                  for article in articles 
                                  if article.get("url")}
                
                if not articles_by_url:
                    logger.warning("Ни одна из переданных статей не содержит URL")
                    return False
                
                # Обновление существующих статей
                for idx, item in enumerate(progress):
                    if item.get("url") in articles_by_url:
                        progress[idx] = articles_by_url[item.get("url")]
                        del articles_by_url[item.get("url")]
                
                # Добавление оставшихся новых статей
                progress.extend(articles_by_url.values())
                
                # Сохранение обновленного прогресса
                result = self._save_progress(progress)
                
                if result:
                    logger.debug(f"Обновлены статусы {len(articles)} статей")
                
                return result
                
            except Exception as e:
                logger.error(f"Ошибка при обновлении статусов статей: {str(e)}")
                return False
    
    def get_article_status(self, url: str) -> Optional[str]:
        """
        Получает текущий статус статьи.
        
        Args:
            url: URL статьи
            
        Returns:
            Optional[str]: Статус статьи или None, если статья не найдена
        """
        with self.lock:
            try:
                progress = self.load_progress()
                
                for item in progress:
                    if item.get("url") == url:
                        return item.get("status")
                
                return None
                
            except Exception as e:
                logger.error(f"Ошибка при получении статуса статьи: {str(e)}")
                return None
    
    def get_statistics(self) -> Dict[str, int]:
        """
        Получает статистику обработки статей.
        
        Returns:
            Dict[str, int]: Словарь с количеством статей по статусам
        """
        with self.lock:
            try:
                progress = self.load_progress()
                
                # Подсчет статей по статусам
                stats = {
                    "total": len(progress),
                    "not_started": 0,
                    "in_progress": 0,
                    "done": 0,
                    "error": 0,
                    "validated": 0
                }
                
                # Добавляем все возможные статусы в статистику
                all_statuses: Set[str] = set()
                
                for item in progress:
                    status = item.get("status", "unknown")
                    all_statuses.add(status)
                    stats[status] = stats.get(status, 0) + 1
                
                return stats
                
            except Exception as e:
                logger.error(f"Ошибка при получении статистики: {str(e)}")
                return {"total": 0, "error": "Ошибка при получении статистики"}
    
    def reset_status(self, url: Optional[str] = None, 
                    status: Optional[str] = None, 
                    new_status: str = "not_started") -> int:
        """
        Сбрасывает статус статей.
        
        Args:
            url: URL статьи для сброса (если None, обрабатываются все статьи)
            status: Фильтр статуса для сброса (если None, фильтр не применяется)
            new_status: Новый статус для статей
            
        Returns:
            int: Количество обновленных статей
        """
        with self.lock:
            try:
                progress = self.load_progress()
                updated_count = 0
                
                for item in progress:
                    # Применение фильтров
                    if url is not None and item.get("url") != url:
                        continue
                    
                    if status is not None and item.get("status") != status:
                        continue
                    
                    # Обновление статуса
                    item["status"] = new_status
                    updated_count += 1
                
                # Сохранение изменений только если были обновления
                if updated_count > 0:
                    self._save_progress(progress)
                    logger.info(f"Сброшены статусы {updated_count} статей на '{new_status}'")
                
                return updated_count
                
            except Exception as e:
                logger.error(f"Ошибка при сбросе статусов: {str(e)}")
                return 0
    
    def remove_article(self, url: str) -> bool:
        """
        Удаляет статью из прогресса.
        
        Args:
            url: URL статьи для удаления
            
        Returns:
            bool: True, если статья успешно удалена
        """
        with self.lock:
            try:
                progress = self.load_progress()
                original_len = len(progress)
                
                # Фильтрация списка для удаления статьи
                progress = [item for item in progress if item.get("url") != url]
                
                # Если длина не изменилась, статья не была найдена
                if len(progress) == original_len:
                    logger.warning(f"Статья с URL '{url}' не найдена в прогрессе")
                    return False
                
                # Сохранение обновленного прогресса
                result = self._save_progress(progress)
                
                if result:
                    logger.info(f"Статья с URL '{url}' удалена из прогресса")
                
                return result
                
            except Exception as e:
                logger.error(f"Ошибка при удалении статьи: {str(e)}")
                return False