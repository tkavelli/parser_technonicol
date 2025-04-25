"""
Модуль для классификации текстовых фрагментов с использованием различных моделей.
Поддерживает классификацию на уровне чанков и отдельных параграфов.
Обеспечивает мультиуровневую классификацию для гибридной RAG-системы.
"""
import logging
import json
import os
import traceback
import re
from pathlib import Path
from typing import List, Dict, Any, Optional, Union, Tuple, Set, Callable
import requests
from urllib.parse import quote

# Опциональные импорты для моделей ML
try:
    import torch
    from transformers import AutoTokenizer, AutoModelForSequenceClassification
    transformers_available = True
except ImportError:
    transformers_available = False

# Импорт функций из других модулей
from chunker import get_chunk_text, flatten_chunks, flatten_paragraphs

# Настройка логирования
logger = logging.getLogger(__name__)


class ContentClassifier:
    """
    Базовый класс для классификации контента.
    Определяет интерфейс для различных реализаций классификаторов.
    """
    
    def classify_chunk(self, chunk: Dict[str, Any]) -> List[str]:
        """
        Классифицирует чанк статьи.
        
        Args:
            chunk: Данные чанка
            
        Returns:
            List[str]: Список меток
        """
        raise NotImplementedError("Subclasses must implement classify_chunk")
    
    def classify_paragraph(self, paragraph: Dict[str, Any], 
                          chunk_context: Optional[Dict[str, Any]] = None) -> List[str]:
        """
        Классифицирует отдельный параграф.
        
        Args:
            paragraph: Данные параграфа
            chunk_context: Контекст чанка, которому принадлежит параграф
            
        Returns:
            List[str]: Список меток
        """
        raise NotImplementedError("Subclasses must implement classify_paragraph")
    
    def get_classifier_name(self) -> str:
        """
        Возвращает имя классификатора.
        
        Returns:
            str: Имя классификатора
        """
        return self.__class__.__name__


class KeywordClassifier(ContentClassifier):
    """
    Классификатор на основе ключевых слов.
    Простой подход, основанный на поиске заданных ключевых слов в тексте.
    """
    
    def __init__(self, keywords_file: Optional[str] = None):
        """
        Инициализация классификатора ключевых слов.
        
        Args:
            keywords_file: Путь к JSON-файлу с ключевыми словами по категориям
        """
        self.label_keywords = self._load_keywords(keywords_file)
        logger.info(f"Инициализирован KeywordClassifier с {len(self.label_keywords)} категориями")
    
    def _load_keywords(self, keywords_file: Optional[str]) -> Dict[str, List[str]]:
        """
        Загружает ключевые слова из файла или использует встроенные.
        
        Args:
            keywords_file: Путь к JSON-файлу с ключевыми словами
            
        Returns:
            Dict[str, List[str]]: Словарь меток и их ключевых слов
        """
        try:
            if keywords_file and os.path.exists(keywords_file):
                try:
                    with open(keywords_file, 'r', encoding='utf-8') as f:
                        keywords_dict = json.load(f)
                    logger.info(f"Загружены ключевые слова из {keywords_file}")
                    return keywords_dict
                except Exception as e:
                    logger.error(f"Ошибка при загрузке ключевых слов из {keywords_file}: {str(e)}")
                    logger.debug(f"Трассировка: {traceback.format_exc()}")
            
            # Встроенный набор ключевых слов (можно расширить)
            logger.info("Используются встроенные ключевые слова")
            return {
                "installation_type": [
                    "монтаж", "установка", "укладка", "способ монтажа", "схема монтажа", 
                    "крепление", "закрепление", "установить", "смонтировать", "монтироваться",
                    "инструкция по монтажу", "монтируется", "производить монтаж", "правила монтажа"
                ],
                "material": [
                    "материал", "вид материала", "тип материала", "состав", "сырье", 
                    "древесина", "бетон", "утеплитель", "кровля", "изоляция",
                    "мастика", "праймер", "битум", "мембрана", "плита",
                    "ячеистый бетон", "кирпич", "штукатурка", "раствор", "смесь"
                ],
                "method": [
                    "метод", "способ", "процедура", "технология", "этап", "шаг", 
                    "процесс", "последовательность", "алгоритм", "инструкция",
                    "руководство", "методика", "техника", "применение", "выполнение",
                    "технология производства", "технология выполнения", "требования"
                ],
                "layer": [
                    "слой", "многослойный", "структура слоя", "толщина слоя", "нанесение слоя", 
                    "покрытие", "слоистый", "прослойка", "слоями", "наслоение",
                    "наносить слой", "верхний слой", "нижний слой", "гидроизоляционный слой"
                ],
                "overlap_type": [
                    "нахлест", "перекрытие", "стык", "перехлест", "соединение", 
                    "шов", "склеивание", "стыковка", "нахлестка", "соединить",
                    "шириной нахлеста", "с нахлестом", "стыковать", "зазор"
                ],
                "technical_characteristics": [
                    "характеристика", "свойство", "параметр", "показатель", "значение",
                    "прочность", "устойчивость", "эластичность", "гибкость", "твердость",
                    "теплопроводность", "влагостойкость", "морозостойкость", "долговечность",
                    "паропроницаемость", "водонепроницаемость", "герметичность"
                ],
                "tools_equipment": [
                    "инструмент", "оборудование", "приспособление", "механизм", "станок",
                    "дрель", "шуруповерт", "ножницы", "пила", "молоток", "правило",
                    "мешалка", "затирка", "кельма", "уровень", "лопата", "швабра"
                ],
                "safety": [
                    "безопасность", "защита", "меры предосторожности", "техника безопасности",
                    "защитные", "опасность", "риск", "травма", "несчастный случай",
                    "средства индивидуальной защиты", "СИЗ", "противопожарный", "огнезащита"
                ],
                "maintenance": [
                    "обслуживание", "уход", "эксплуатация", "чистка", "ремонт", 
                    "восстановление", "обновление", "замена", "диагностика",
                    "периодичность", "сервис", "профилактика", "техническое обслуживание"
                ]
            }
        except Exception as e:
            logger.error(f"Ошибка при загрузке ключевых слов: {str(e)}")
            logger.debug(f"Трассировка: {traceback.format_exc()}")
            # Возвращаем пустой словарь в случае ошибки
            return {}
    
    def classify_chunk(self, chunk: Dict[str, Any]) -> List[str]:
        """
        Классифицирует чанк на основе ключевых слов.
        
        Args:
            chunk: Данные чанка
            
        Returns:
            List[str]: Список меток
        """
        try:
            # Объединяем заголовок и текст чанка
            chunk_text = chunk.get("title", "") + " " + get_chunk_text(chunk)
            chunk_text = chunk_text.lower()
            
            # Поиск ключевых слов
            labels = []
            for label, keywords in self.label_keywords.items():
                for keyword in keywords:
                    if keyword.lower() in chunk_text:
                        labels.append(label)
                        break
            
            # Добавление меток из хлебных крошек, если есть
            breadcrumbs = []
            if "breadcrumbs" in chunk:
                breadcrumbs = chunk["breadcrumbs"]
            elif "metadata" in chunk and "breadcrumbs" in chunk["metadata"]:
                breadcrumbs = chunk["metadata"]["breadcrumbs"]
                
            if breadcrumbs:
                breadcrumb_labels = self._extract_labels_from_breadcrumbs(breadcrumbs)
                labels.extend(breadcrumb_labels)
            
            # Сохраняем источник меток для гибридной RAG-системы
            labeled_entities = []
            for label in labels:
                labeled_entities.append({
                    "category": label,
                    "source": "keyword_classifier",
                    "confidence": 1.0
                })
            
            # Добавляем метаданные о классификации, если есть место для них
            if "metadata" in chunk:
                chunk["metadata"]["classified_by"] = chunk["metadata"].get("classified_by", []) + ["keyword"]
                chunk["metadata"]["labeled_entities"] = chunk["metadata"].get("labeled_entities", []) + labeled_entities
            
            return list(set(labels))  # Удаление дубликатов
            
        except Exception as e:
            logger.error(f"Ошибка при классификации чанка {chunk.get('id', 'unknown')}: {str(e)}")
            logger.debug(f"Трассировка: {traceback.format_exc()}")
            return []
    
    def classify_paragraph(self, paragraph: Dict[str, Any], 
                          chunk_context: Optional[Dict[str, Any]] = None) -> List[str]:
        """
        Классифицирует параграф на основе ключевых слов.
        
        Args:
            paragraph: Данные параграфа
            chunk_context: Контекст чанка, которому принадлежит параграф
            
        Returns:
            List[str]: Список меток
        """
        try:
            # Получаем текст параграфа
            paragraph_text = paragraph.get("text", "").lower()
            
            # Поиск ключевых слов
            labels = []
            for label, keywords in self.label_keywords.items():
                for keyword in keywords:
                    if keyword.lower() in paragraph_text:
                        labels.append(label)
                        break
            
            # Дополнительно добавляем некоторые метки из контекста чанка, если он есть
            if chunk_context and "labels" in chunk_context:
                # Добавляем только важные метки, передающиеся по наследству
                inheritable_labels = {"material", "technical_characteristics", "safety"}
                inherited_labels = [l for l in chunk_context["labels"] if l in inheritable_labels]
                labels.extend(inherited_labels)
            
            # Сохраняем источник меток для гибридной RAG-системы
            labeled_entities = []
            for label in labels:
                labeled_entities.append({
                    "category": label,
                    "source": "keyword_classifier",
                    "confidence": 1.0
                })
            
            # Добавляем метаданные о классификации, если есть место для них
            if "metadata" in paragraph:
                paragraph["metadata"]["classified_by"] = paragraph["metadata"].get("classified_by", []) + ["keyword"]
                paragraph["metadata"]["labeled_entities"] = paragraph["metadata"].get("labeled_entities", []) + labeled_entities
            
            return list(set(labels))  # Удаление дубликатов
            
        except Exception as e:
            logger.error(f"Ошибка при классификации параграфа {paragraph.get('id', 'unknown')}: {str(e)}")
            logger.debug(f"Трассировка: {traceback.format_exc()}")
            return []
    
    def _extract_labels_from_breadcrumbs(self, breadcrumbs: List[str]) -> List[str]:
        """
        Извлекает метки из хлебных крошек.
        
        Args:
            breadcrumbs: Список хлебных крошек
            
        Returns:
            List[str]: Список извлеченных меток
        """
        labels = []
        
        try:
            # Отображение ключевых слов из хлебных крошек в метки
            breadcrumb_mappings = {
                "материалы": "material",
                "звукоизоляционные материалы": "sound_insulation",
                "теплоизоляционные материалы": "thermal_insulation",
                "гидроизоляция": "waterproofing",
                "кровля": "roofing",
                "фасады": "facade",
                "фундаменты": "foundation",
                "инструменты": "tools_equipment",
                "технологии": "method",
                "монтаж": "installation_type"
            }
            
            for crumb in breadcrumbs:
                crumb_lower = crumb.lower()
                for key, label in breadcrumb_mappings.items():
                    if key in crumb_lower:
                        labels.append(label)
        
        except Exception as e:
            logger.error(f"Ошибка при извлечении меток из хлебных крошек: {str(e)}")
            logger.debug(f"Трассировка: {traceback.format_exc()}")
        
        return labels


class BertClassifier(ContentClassifier):
    """
    Классификатор на основе BERT-модели.
    Использует предобученную модель для классификации текста.
    """
    
    def __init__(self, model_name: str = "DeepPavlov/rubert-base-cased"):
        """
        Инициализация BERT-классификатора.
        
        Args:
            model_name: Имя предобученной модели
        """
        self.model_name = model_name
        self.model = None
        self.tokenizer = None
        self.label_mapping = self._get_label_mapping()
        
        if not transformers_available:
            logger.warning("Библиотека transformers не установлена. BERT-классификатор недоступен.")
            return
        
        try:
            self.tokenizer = AutoTokenizer.from_pretrained(model_name)
            self.model = AutoModelForSequenceClassification.from_pretrained(model_name)
            logger.info(f"BERT-классификатор инициализирован с моделью {model_name}")
        except Exception as e:
            logger.error(f"Ошибка при инициализации BERT-классификатора: {str(e)}")
            logger.debug(f"Трассировка: {traceback.format_exc()}")
            self.model = None
            self.tokenizer = None
    
    def _get_label_mapping(self) -> Dict[int, str]:
        """
        Создает отображение от индексов выходов модели к текстовым меткам.
        
        Returns:
            Dict[int, str]: Отображение индекс -> метка
        """
        # Примечание: это примерное отображение, нужно настроить под конкретную модель
        return {
            0: "installation_type",
            1: "material",
            2: "method",
            3: "technical_characteristics",
            4: "tools_equipment",
            5: "safety",
            6: "maintenance"
        }
    
    def classify_chunk(self, chunk: Dict[str, Any]) -> List[str]:
        """
        Классифицирует чанк с использованием BERT-модели.
        
        Args:
            chunk: Данные чанка
            
        Returns:
            List[str]: Список меток
        """
        if not self.model or not self.tokenizer:
            return []
        
        try:
            # Объединяем заголовок и текст чанка
            chunk_text = chunk.get("title", "") + " " + get_chunk_text(chunk)
            
            # Ограничиваем длину текста, если он слишком длинный
            if len(chunk_text) > 1024:
                chunk_text = chunk_text[:1024]
            
            # Токенизация и предсказание
            inputs = self.tokenizer(chunk_text, return_tensors="pt", truncation=True, max_length=512)
            outputs = self.model(**inputs)
            predictions = torch.nn.functional.softmax(outputs.logits, dim=-1)
            
            # Преобразование предсказаний в метки
            labels = self._convert_predictions_to_labels(predictions)
            
            # Сохраняем источник меток для гибридной RAG-системы
            labeled_entities = []
            for label, confidence in zip(labels, predictions[0].tolist()):
                labeled_entities.append({
                    "category": label,
                    "source": "bert_classifier",
                    "confidence": float(confidence)
                })
            
            # Добавляем метаданные о классификации, если есть место для них
            if "metadata" in chunk:
                chunk["metadata"]["classified_by"] = chunk["metadata"].get("classified_by", []) + ["bert"]
                chunk["metadata"]["labeled_entities"] = chunk["metadata"].get("labeled_entities", []) + labeled_entities
            
            return labels
            
        except Exception as e:
            logger.error(f"Ошибка при BERT-классификации чанка {chunk.get('id', 'unknown')}: {str(e)}")
            logger.debug(f"Трассировка: {traceback.format_exc()}")
            return []
    
    def classify_paragraph(self, paragraph: Dict[str, Any], 
                          chunk_context: Optional[Dict[str, Any]] = None) -> List[str]:
        """
        Классифицирует параграф с использованием BERT-модели.
        
        Args:
            paragraph: Данные параграфа
            chunk_context: Контекст чанка, которому принадлежит параграф
            
        Returns:
            List[str]: Список меток
        """
        if not self.model or not self.tokenizer:
            return []
        
        try:
            # Для параграфа можно добавить контекст из заголовка чанка
            context_prefix = ""
            if chunk_context and "title" in chunk_context:
                context_prefix = chunk_context["title"] + ": "
            
            paragraph_text = context_prefix + paragraph.get("text", "")
            
            # Токенизация и предсказание
            inputs = self.tokenizer(paragraph_text, return_tensors="pt", truncation=True, max_length=512)
            outputs = self.model(**inputs)
            predictions = torch.nn.functional.softmax(outputs.logits, dim=-1)
            
            # Преобразование предсказаний в метки
            labels = self._convert_predictions_to_labels(predictions)
            
            # Сохраняем источник меток для гибридной RAG-системы
            labeled_entities = []
            for label, confidence in zip(labels, predictions[0].tolist()):
                labeled_entities.append({
                    "category": label,
                    "source": "bert_classifier",
                    "confidence": float(confidence)
                })
            
            # Добавляем метаданные о классификации, если есть место для них
            if "metadata" in paragraph:
                paragraph["metadata"]["classified_by"] = paragraph["metadata"].get("classified_by", []) + ["bert"]
                paragraph["metadata"]["labeled_entities"] = paragraph["metadata"].get("labeled_entities", []) + labeled_entities
            
            return labels
            
        except Exception as e:
            logger.error(f"Ошибка при BERT-классификации параграфа {paragraph.get('id', 'unknown')}: {str(e)}")
            logger.debug(f"Трассировка: {traceback.format_exc()}")
            return []
    
    def _convert_predictions_to_labels(self, predictions) -> List[str]:
        """
        Преобразует выходы модели в метки.
        
        Args:
            predictions: Выходы модели
            
        Returns:
            List[str]: Список меток
        """
        try:
            # Получение индексов с наибольшей вероятностью
            confidences, indices = torch.topk(predictions[0], k=3)
            
            # Пороговое значение для включения метки
            threshold = 0.3
            
            # Преобразование индексов в метки
            labels = []
            for confidence, idx in zip(confidences, indices):
                if confidence >= threshold and idx.item() in self.label_mapping:
                    labels.append(self.label_mapping[idx.item()])
            
            return labels
            
        except Exception as e:
            logger.error(f"Ошибка при преобразовании предсказаний в метки: {str(e)}")
            logger.debug(f"Трассировка: {traceback.format_exc()}")
            return []


class GPTClassifier(ContentClassifier):
    """
    Классификатор на основе GPT API.
    Использует внешний API для классификации текста.
    """
    
    def __init__(self, api_key: Optional[str] = None, api_url: Optional[str] = None):
        """
        Инициализация GPT-классификатора.
        
        Args:
            api_key: Ключ API (можно указать через переменную окружения GPT_API_KEY)
            api_url: URL API (по умолчанию OpenAI API)
        """
        self.api_key = api_key or os.environ.get("GPT_API_KEY")
        self.api_url = api_url or "https://api.openai.com/v1/chat/completions"
        self.model = "gpt-3.5-turbo"
        
        if not self.api_key:
            logger.warning("API ключ для GPT не указан. GPT-классификатор будет недоступен.")
    
    def classify_chunk(self, chunk: Dict[str, Any]) -> List[str]:
        """
        Классифицирует чанк с использованием GPT API.
        
        Args:
            chunk: Данные чанка
            
        Returns:
            List[str]: Список меток
        """
        if not self.api_key:
            return []
        
        try:
            # Объединяем заголовок и текст чанка
            chunk_text = chunk.get("title", "") + " " + get_chunk_text(chunk)
            
            # Ограничение длины текста
            if len(chunk_text) > 1500:
                chunk_text = chunk_text[:1500] + "..."
            
            # Формирование промпта для GPT
            prompt = f"""
            Определи наиболее подходящие категории для следующего текста о строительных материалах и технологиях:
            
            {chunk_text}
            
            Выбери не более 3 категорий из следующего списка:
            - installation_type (монтаж, установка)
            - material (материалы, свойства материалов)
            - method (методы, способы, процедуры)
            - layer (слои, покрытия)
            - overlap_type (нахлесты, стыки)
            - technical_characteristics (технические характеристики)
            - tools_equipment (инструменты, оборудование)
            - safety (безопасность, защита)
            - maintenance (обслуживание, уход)
            
            Формат ответа: только список категорий, разделенных запятыми, без пояснений.
            """
            
            # Запрос к API
            labels, confidences = self._call_gpt_api(prompt)
            
            # Сохраняем источник меток для гибридной RAG-системы
            labeled_entities = []
            for i, label in enumerate(labels):
                confidence = confidences[i] if i < len(confidences) else 0.8
                labeled_entities.append({
                    "category": label,
                    "source": "gpt_classifier",
                    "confidence": confidence
                })
            
            # Добавляем метаданные о классификации, если есть место для них
            if "metadata" in chunk:
                chunk["metadata"]["classified_by"] = chunk["metadata"].get("classified_by", []) + ["gpt"]
                chunk["metadata"]["labeled_entities"] = chunk["metadata"].get("labeled_entities", []) + labeled_entities
            
            return labels
            
        except Exception as e:
            logger.error(f"Ошибка при GPT-классификации чанка {chunk.get('id', 'unknown')}: {str(e)}")
            logger.debug(f"Трассировка: {traceback.format_exc()}")
            return []
    
    def classify_paragraph(self, paragraph: Dict[str, Any], 
                          chunk_context: Optional[Dict[str, Any]] = None) -> List[str]:
        """
        Классифицирует параграф с использованием GPT API.
        
        Args:
            paragraph: Данные параграфа
            chunk_context: Контекст чанка, которому принадлежит параграф
            
        Returns:
            List[str]: Список меток
        """
        if not self.api_key:
            return []
        
        try:
            # Формирование контекста
            context = ""
            if chunk_context and "title" in chunk_context:
                context = f"Контекст (заголовок раздела): {chunk_context['title']}\n\n"
            
            # Формирование промпта для GPT
            prompt = f"""
            {context}Определи наиболее подходящие категории для следующего абзаца текста о строительных материалах и технологиях:
            
            {paragraph.get("text", "")}
            
            Выбери не более 2 категорий из следующего списка:
            - installation_type (монтаж, установка)
            - material (материалы, свойства материалов)
            - method (методы, способы, процедуры)
            - layer (слои, покрытия)
            - overlap_type (нахлесты, стыки)
            - technical_characteristics (технические характеристики)
            - tools_equipment (инструменты, оборудование)
            - safety (безопасность, защита)
            - maintenance (обслуживание, уход)
            
            Формат ответа: только список категорий, разделенных запятыми, без пояснений.
            """
            
            # Запрос к API
            labels, confidences = self._call_gpt_api(prompt)
            
            # Сохраняем источник меток для гибридной RAG-системы
            labeled_entities = []
            for i, label in enumerate(labels):
                confidence = confidences[i] if i < len(confidences) else 0.8
                labeled_entities.append({
                    "category": label,
                    "source": "gpt_classifier",
                    "confidence": confidence
                })
            
            # Добавляем метаданные о классификации, если есть место для них
            if "metadata" in paragraph:
                paragraph["metadata"]["classified_by"] = paragraph["metadata"].get("classified_by", []) + ["gpt"]
                paragraph["metadata"]["labeled_entities"] = paragraph["metadata"].get("labeled_entities", []) + labeled_entities
            
            return labels
            
        except Exception as e:
            logger.error(f"Ошибка при GPT-классификации параграфа {paragraph.get('id', 'unknown')}: {str(e)}")
            logger.debug(f"Трассировка: {traceback.format_exc()}")
            return []
    
    def _call_gpt_api(self, prompt: str) -> Tuple[List[str], List[float]]:
        """
        Вызывает GPT API для классификации.
        
        Args:
            prompt: Промпт для API
            
        Returns:
            Tuple[List[str], List[float]]: Список меток и список уверенностей
        """
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json"
        }
        
        data = {
            "model": self.model,
            "messages": [{"role": "user", "content": prompt}],
            "temperature": 0.1,
            "max_tokens": 50
        }
        
        try:
            response = requests.post(
                self.api_url, 
                headers=headers, 
                json=data,
                timeout=10  # 10 секунд таймаут
            )
            
            if response.status_code != 200:
                logger.error(f"Ошибка API: {response.status_code}, {response.text}")
                return [], []
            
            result = response.json()
            response_text = result["choices"][0]["message"]["content"].strip()
            
            # Парсинг ответа (ожидается список категорий, разделенных запятыми)
            labels = [label.strip() for label in response_text.split(",")]
            
            # Генерация уверенностей (GPT не предоставляет их напрямую)
            # Но мы можем использовать позицию в списке как признак уверенности
            confidences = [0.9 - i * 0.1 for i in range(len(labels))]
            
            return labels, confidences
            
        except requests.exceptions.RequestException as e:
            logger.error(f"Ошибка при вызове GPT API: {str(e)}")
            logger.debug(f"Трассировка: {traceback.format_exc()}")
            return [], []


class HybridClassifier(ContentClassifier):
    """
    Гибридный классификатор, объединяющий несколько подходов.
    Использует доступные классификаторы и объединяет их результаты.
    Оптимизирован для гибридной RAG-архитектуры.
    """
    
    def __init__(self, keywords_file: Optional[str] = None, 
                use_bert: bool = False, use_gpt: bool = False,
                gpt_api_key: Optional[str] = None, 
                weighting: Optional[Dict[str, float]] = None):
        """
        Инициализация гибридного классификатора.
        
        Args:
            keywords_file: Путь к файлу с ключевыми словами
            use_bert: Использовать ли BERT-классификатор
            use_gpt: Использовать ли GPT-классификатор
            gpt_api_key: Ключ API для GPT
            weighting: Веса для различных классификаторов (по умолчанию равные веса)
        """
        # Инициализация доступных классификаторов
        self.classifiers = []
        self.classifier_weights = weighting or {
            "KeywordClassifier": 1.0,
            "BertClassifier": 0.8,
            "GPTClassifier": 1.2
        }
        
        # Всегда добавляем базовый классификатор по ключевым словам
        self.keyword_classifier = KeywordClassifier(keywords_file)
        self.classifiers.append(self.keyword_classifier)
        
        # Добавляем BERT, если запрошен и доступен
        self.bert_classifier = None
        if use_bert and transformers_available:
            self.bert_classifier = BertClassifier()
            if self.bert_classifier.model:
                self.classifiers.append(self.bert_classifier)
        
        # Добавляем GPT, если запрошен
        self.gpt_classifier = None
        if use_gpt:
            self.gpt_classifier = GPTClassifier(gpt_api_key)
            if self.gpt_classifier.api_key:
                self.classifiers.append(self.gpt_classifier)
        
        logger.info(f"Гибридный классификатор инициализирован с {len(self.classifiers)} классификаторами")
    
    def classify_chunk(self, chunk: Dict[str, Any]) -> List[str]:
        """
        Классифицирует чанк, используя все доступные классификаторы.
        
        Args:
            chunk: Данные чанка
            
        Returns:
            List[str]: Объединенный список меток
        """
        all_labels = []
        label_scores = {}
        
        try:
            # Получение меток от всех классификаторов
            for classifier in self.classifiers:
                try:
                    labels = classifier.classify_chunk(chunk)
                    classifier_name = classifier.get_classifier_name()
                    weight = self.classifier_weights.get(classifier_name, 1.0)
                    
                    # Учитываем веса классификаторов
                    for label in labels:
                        if label not in label_scores:
                            label_scores[label] = 0
                        label_scores[label] += weight
                        
                    all_labels.extend(labels)
                except Exception as e:
                    logger.error(f"Ошибка в классификаторе {classifier.__class__.__name__}: {str(e)}")
                    logger.debug(f"Трассировка: {traceback.format_exc()}")
            
            # Удаление дубликатов и сортировка по весу
            unique_labels = sorted(
                list(set(all_labels)), 
                key=lambda x: label_scores.get(x, 0), 
                reverse=True
            )
            
            # Добавляем метаданные о комбинированных результатах для гибридной RAG-системы
            if "metadata" in chunk:
                chunk["metadata"]["label_scores"] = label_scores
            
            return unique_labels
            
        except Exception as e:
            logger.error(f"Ошибка при гибридной классификации чанка {chunk.get('id', 'unknown')}: {str(e)}")
            logger.debug(f"Трассировка: {traceback.format_exc()}")
            return []
    
    def classify_paragraph(self, paragraph: Dict[str, Any], 
                          chunk_context: Optional[Dict[str, Any]] = None) -> List[str]:
        """
        Классифицирует параграф, используя все доступные классификаторы.
        
        Args:
            paragraph: Данные параграфа
            chunk_context: Контекст чанка, которому принадлежит параграф
            
        Returns:
            List[str]: Объединенный список меток
        """
        all_labels = []
        label_scores = {}
        
        try:
            # Получение меток от всех классификаторов
            for classifier in self.classifiers:
                try:
                    labels = classifier.classify_paragraph(paragraph, chunk_context)
                    classifier_name = classifier.get_classifier_name()
                    weight = self.classifier_weights.get(classifier_name, 1.0)
                    
                    # Учитываем веса классификаторов
                    for label in labels:
                        if label not in label_scores:
                            label_scores[label] = 0
                        label_scores[label] += weight
                        
                    all_labels.extend(labels)
                except Exception as e:
                    logger.error(f"Ошибка в классификаторе {classifier.__class__.__name__}: {str(e)}")
                    logger.debug(f"Трассировка: {traceback.format_exc()}")
            
            # Удаление дубликатов и сортировка по весу
            unique_labels = sorted(
                list(set(all_labels)), 
                key=lambda x: label_scores.get(x, 0), 
                reverse=True
            )
            
            # Добавляем метаданные о комбинированных результатах для гибридной RAG-системы
            if "metadata" in paragraph:
                paragraph["metadata"]["label_scores"] = label_scores
            
            return unique_labels
            
        except Exception as e:
            logger.error(f"Ошибка при гибридной классификации параграфа {paragraph.get('id', 'unknown')}: {str(e)}")
            logger.debug(f"Трассировка: {traceback.format_exc()}")
            return []


# Функция для получения глобального экземпляра классификатора
_classifier_instance = None

def get_classifier(keywords_file: Optional[str] = None, 
                  use_bert: bool = False, use_gpt: bool = False,
                  gpt_api_key: Optional[str] = None,
                  weighting: Optional[Dict[str, float]] = None) -> ContentClassifier:
    """
    Получает экземпляр классификатора для использования в приложении.
    
    Args:
        keywords_file: Путь к файлу с ключевыми словами
        use_bert: Использовать ли BERT-классификатор
        use_gpt: Использовать ли GPT-классификатор
        gpt_api_key: Ключ API для GPT
        weighting: Веса для различных классификаторов
        
    Returns:
        ContentClassifier: Экземпляр классификатора
    """
    global _classifier_instance
    
    try:
        if _classifier_instance is None:
            logger.info("Создание нового экземпляра классификатора")
            _classifier_instance = HybridClassifier(
                keywords_file, use_bert, use_gpt, gpt_api_key, weighting
            )
        else:
            logger.debug("Использование существующего экземпляра классификатора")
        
        return _classifier_instance
        
    except Exception as e:
        logger.error(f"Ошибка при получении экземпляра классификатора: {str(e)}")
        logger.debug(f"Трассировка: {traceback.format_exc()}")
        
        # В случае ошибки, возвращаем простой классификатор по ключевым словам
        return KeywordClassifier()


def classify_article(chunks: List[Dict[str, Any]], classifier: Optional[ContentClassifier] = None) -> Tuple[List[Dict[str, Any]], Dict[str, List[str]]]:
    """
    Классифицирует все чанки и параграфы статьи.
    
    Args:
        chunks: Список чанков статьи
        classifier: Классификатор для использования (если None, будет создан новый)
        
    Returns:
        Tuple[List[Dict[str, Any]], Dict[str, List[str]]]: Обновленные чанки и словарь меток параграфов
    """
    try:
        # Получение классификатора
        if classifier is None:
            classifier = get_classifier()
        
        logger.info(f"Начало классификации статьи с {len(chunks)} чанками")
        
        # Получение плоского списка чанков
        flat_chunks = flatten_chunks(chunks)
        
        # Классификация каждого чанка
        for chunk in flat_chunks:
            chunk_id = chunk.get("id", "unknown")
            try:
                logger.debug(f"Классификация чанка {chunk_id}")
                chunk["labels"] = classifier.classify_chunk(chunk)
            except Exception as e:
                logger.error(f"Ошибка при классификации чанка {chunk_id}: {str(e)}")
                logger.debug(f"Трассировка: {traceback.format_exc()}")
                chunk["labels"] = []
        
        # Классификация каждого параграфа
        paragraph_labels = {}
        for chunk in flat_chunks:
            chunk_id = chunk.get("id", "unknown")
            for paragraph in chunk.get("paragraphs", []):
                paragraph_id = paragraph.get("id", "unknown")
                try:
                    logger.debug(f"Классификация параграфа {paragraph_id}")
                    labels = classifier.classify_paragraph(paragraph, chunk)
                    paragraph_labels[paragraph_id] = labels
                    
                    # Обновляем словарь меток параграфов в чанке
                    if "paragraph_labels" not in chunk:
                        chunk["paragraph_labels"] = {}
                    chunk["paragraph_labels"][paragraph_id] = labels
                except Exception as e:
                    logger.error(f"Ошибка при классификации параграфа {paragraph_id}: {str(e)}")
                    logger.debug(f"Трассировка: {traceback.format_exc()}")
                    paragraph_labels[paragraph_id] = []
                    chunk["paragraph_labels"][paragraph_id] = []
        
        logger.info(f"Завершена классификация статьи: {len(flat_chunks)} чанков, {len(paragraph_labels)} параграфов")
        
        return chunks, paragraph_labels
        
    except Exception as e:
        logger.error(f"Ошибка при классификации статьи: {str(e)}")
        logger.debug(f"Трассировка: {traceback.format_exc()}")
        return chunks, {}