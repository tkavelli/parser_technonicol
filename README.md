# parser_technonicol

# Пайплайн для обработки и семантического обогащения текстовых чанков

Проект предоставляет инструментарий для обработки технических статей, их разбиения на смысловые блоки (чанки), семантического обогащения метаданными и тегами, а также подготовки для эффективного использования в RAG-системе (Retrieval-Augmented Generation).

## Назначение

Основная цель проекта — улучшить качество ответов ИИ-ассистента путем:
- Разбиения документов на смысловые чанки
- Обогащения каждого чанка информативными метками/тегами
- Построения семантических связей между чанками
- Создания многоуровневой классификации для точного извлечения
- Локальной векторизации с использованием GPU

## Возможности

- **Локальная обработка данных** на GPU (NVIDIA 4090)
- **Двухуровневое разбиение**:
  - **Чанки** — разделы статьи по заголовкам
  - **Параграфы** — отдельные абзацы внутри чанков
- **Семантическое обогащение контента**:
  - Автоматическое присвоение тематических меток/хештегов
  - Добавление структурных заголовков для улучшения контекста
  - Извлечение числовых данных с единицами измерения
  - Определение технических параметров и спецификаций
- **Мультиуровневая классификация**:
  - **Базовый уровень**: классификация по ключевым словам
  - **Продвинутый уровень**: классификация с помощью локальных BERT-моделей
- **Векторизация и индексация**:
  - Локальная генерация эмбеддингов с помощью GPU
  - Загрузка в Weaviate для быстрого семантического поиска

## Требования

### Аппаратные требования

- **GPU**: NVIDIA 4090 или аналогичный с 24+ ГБ VRAM
- **CPU**: 8+ ядер рекомендуется для предобработки данных
- **RAM**: 32+ ГБ для обработки больших наборов данных
- **Дисковое пространство**: 50+ ГБ для хранения моделей и промежуточных данных

### Программные требования

- Python 3.8+
- CUDA 11.8+ и cuDNN
- Docker (для запуска Weaviate)
- Git LFS (для скачивания моделей)

## Установка

1. **Клонирование репозитория**:
```bash
git clone https://github.com/yourusername/semantic-chunking-pipeline.git
cd semantic-chunking-pipeline
```

2. **Создание виртуального окружения**:
```bash
python -m venv venv
source venv/bin/activate  # Для Linux/Mac
venv\Scripts\activate     # Для Windows
```

3. **Установка зависимостей**:
```bash
pip install -r requirements.txt
```

4. **Установка CUDA и PyTorch с поддержкой GPU**:
```bash
# Для CUDA 11.8
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
```

5. **Запуск Weaviate в Docker**:
```bash
docker run -d -p 8080:8080 --name weaviate-server \
  --env ENABLE_MODULES="text2vec-transformers" \
  --env TRANSFORMERS_INFERENCE_API="http://localhost:8000" \
  --env PERSISTENCE_DATA_PATH="/var/lib/weaviate" \
  semitechnologies/weaviate:1.19.6
```

## Использование

### Подготовка данных

1. **Подготовка входных данных**:
   - Поместите HTML-файлы в директорию `data/html/`
   - Или создайте Excel-файл со списком URL статей в колонке `URL`

2. **Конфигурация классификации**:
   - Настройте словари ключевых слов в `classifier.py`
   - При необходимости адаптируйте BERT-модель в конфигурации

### Основной процесс обработки

```bash
# Базовая обработка
python main.py --input articles.xlsx

# С использованием GPU для обработки модели BERT и генерации эмбеддингов
python main.py --input articles.xlsx --use-gpu --workers 4
```

### Параметры командной строки

- `--input`, `-i`: Путь к файлу со списком статей/документов (Excel, CSV)
- `--workers`, `-w`: Количество параллельных процессов (по умолчанию: 4)
- `--use-gpu`: Использовать GPU для вычислений (рекомендуется для BERT и эмбеддингов)
- `--force-restart`, `-f`: Принудительно начать обработку заново для всех статей
- `--skip-vectorization`, `-s`: Пропустить этап векторизации (для отладки)
- `--log-level`, `-l`: Уровень логирования (DEBUG, INFO, WARNING, ERROR)

### Процесс семантического обогащения

Каждый чанк и параграф проходит следующие стадии обработки:

1. **Извлечение структуры**: Определение заголовков, уровней, взаимосвязей
2. **Классификация контента**: Присвоение тематических меток
3. **Обогащение метаданными**:
   - Извлечение числовых параметров с единицами измерения
   - Определение технических характеристик
   - Выявление связей с другими документами
4. **Векторизация**:
   - Генерация эмбеддингов с использованием GPU
   - Оптимизация векторных представлений

### Тестирование обогащенных данных

Для проверки классификации и обогащения:
```bash
python tools/test_classification.py
```

## Структура проекта

```
.
├── main.py            # Основной модуль управления пайплайном
├── fetcher.py         # Загрузка HTML-контента статей
├── parser.py          # Извлечение структуры документов и хлебных крошек
├── chunker.py         # Разбиение контента на чанки и параграфы
├── classifier.py      # Семантическая классификация контента
├── extractor.py       # Извлечение метаданных (тегов, числовых данных)
├── serializer.py      # Сохранение данных в различных форматах
├── db.py              # Работа с локальной базой данных
├── vectorizer.py      # Векторизация и загрузка в Weaviate
├── state_manager.py   # Управление состоянием обработки
├── utils.py           # Вспомогательные функции
├── text_utils.py      # Утилиты для обработки текста
├── data/              # Директория с исходными данными
├── output/            # Результаты обработки
└── tools/             # Вспомогательные инструменты
```

## База данных и метаданные

### Структура базы данных SQLite (`articles.db`)

- **articles**: Основная информация о документах
- **chunks**: Разделы документов с привязкой к структуре
- **paragraphs**: Отдельные параграфы с семантическими метками
- **categorized_entities**: Семантические метки и их характеристики
- **numbers**: Извлеченные числовые данные с единицами измерения
- **tech_specs**: Технические характеристики из таблиц и текста
- **links**: Внутренние связи между документами

### Семантические метки

Проект поддерживает следующие категории семантических меток:

- `installation_type`: Методы и типы монтажа/установки
- `material`: Типы материалов и их свойства
- `method`: Методологии и процессы
- `layer`: Структура слоев и покрытий
- `overlap_type`: Типы нахлестов и соединений
- `technical_characteristics`: Технические параметры
- `tools_equipment`: Инструменты и оборудование
- `safety`: Меры безопасности
- `maintenance`: Обслуживание и эксплуатация

## Настройка и оптимизация

### Оптимизация GPU-вычислений

Для максимальной производительности на NVIDIA 4090:

1. **Настройка размера батча:**
```python
# В vectorizer.py, определение размера батча в зависимости от модели
BERT_BATCH_SIZE = 32  # Для 4090 можно использовать 32-64
EMBEDDING_BATCH_SIZE = 128  # Для генерации эмбеддингов
```

2. **Настройка CUDA ядер:**
```bash
# Установка переменных окружения перед запуском
export CUDA_VISIBLE_DEVICES=0
export OMP_NUM_THREADS=8
```

### Настройка классификаторов

Для улучшения семантического обогащения:

1. **Редактирование словарей ключевых слов** в `classifier.py`
2. **Выбор оптимальной модели BERT** (рекомендуется ruBERT или multilingual BERT)
3. **Настройка пороговых значений уверенности** для меток

### Настройка Weaviate

Рекомендации по настройке Weaviate для оптимальной производительности:

```bash
# Запуск с оптимизированными параметрами
docker run -d -p 8080:8080 --name weaviate-server \
  --env QUERY_DEFAULTS_LIMIT=25 \
  --env AUTHENTICATION_ANONYMOUS_ACCESS_ENABLED=true \
  --env PERSISTENCE_DATA_PATH="/var/lib/weaviate" \
  --env DEFAULT_VECTORIZER_MODULE="text2vec-transformers" \
  --env ENABLE_MODULES="text2vec-transformers" \
  --env TRANSFORMERS_INFERENCE_API="http://localhost:8000" \
  --env DISK_USE_READONLY_PERCENTAGE=95 \
  semitechnologies/weaviate:1.19.6
```

## Устранение проблем

### Проблемы с GPU

**Проблема**: CUDA out of memory
**Решение**: Уменьшите размер батча в `vectorizer.py` или используйте опцию `--skip-vectorization` с последующей отдельной векторизацией.

**Проблема**: Модель BERT не загружается на GPU
**Решение**: Проверьте версию PyTorch и CUDA, при необходимости переустановите с поддержкой CUDA.

### Проблемы с Weaviate

**Проблема**: Не удается подключиться к Weaviate
**Решение**: Проверьте, запущен ли контейнер Docker и доступен ли порт 8080.

**Проблема**: Timeout при загрузке данных
**Решение**: Увеличьте таймаут в `vectorizer.py` и уменьшите размер батча при загрузке.

## Примеры использования

### Пример 1: Полная обработка с GPU

```bash
python main.py --input articles.xlsx --use-gpu --workers 4
```

### Пример 2: Только классификация без векторизации

```bash
python main.py --input articles.xlsx --skip-vectorization
```

### Пример 3: Тестирование классификации на определенном файле

```bash
python tools/test_classification.py --file data/html/example.html
```

## Лицензия

MIT
