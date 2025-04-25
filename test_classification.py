"""
Скрипт для тестирования двухуровневой классификации контента на примере.
Позволяет проверить работу классификаторов на тестовых данных.
"""
import os
import sys
import json
import logging
from pathlib import Path
from typing import List, Dict, Any

# Добавляем родительскую директорию в sys.path для импорта модулей
sys.path.append(str(Path(__file__).parent.parent))

# Импорт модулей пайплайна
from classifier import (
    KeywordClassifier, BertClassifier, GPTClassifier, HybridClassifier,
    ContentClassifier, get_classifier, classify_article
)
from chunker import get_chunk_text

# Настройка логирования
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger("test_classification")


# Тестовые данные
TEST_CHUNKS = [
    {
        "id": "test_chunk_1",
        "title": "Монтаж кровельного материала",
        "level": 2,
        "elements": [
            {
                "type": "paragraph",
                "text": "Для монтажа кровельного материала используйте строительный степлер, гвоздевой пистолет "
                       "или специальные саморезы. Крепление должно быть надежным и обеспечивать плотное прилегание."
            },
            {
                "type": "paragraph",
                "text": "Избегайте перетяжки креплений, чтобы не повредить материал. "
                       "Соблюдайте технологию монтажа, рекомендованную производителем."
            }
        ],
        "paragraphs": [
            {
                "id": "test_chunk_1_p_1",
                "text": "Для монтажа кровельного материала используйте строительный степлер, гвоздевой пистолет "
                       "или специальные саморезы. Крепление должно быть надежным и обеспечивать плотное прилегание.",
                "parent_chunk_id": "test_chunk_1",
                "parent_chunk_title": "Монтаж кровельного материала"
            },
            {
                "id": "test_chunk_1_p_2",
                "text": "Избегайте перетяжки креплений, чтобы не повредить материал. "
                       "Соблюдайте технологию монтажа, рекомендованную производителем.",
                "parent_chunk_id": "test_chunk_1",
                "parent_chunk_title": "Монтаж кровельного материала"
            }
        ],
        "children": [],
        "labels": [],
        "paragraph_labels": {},
        "numbers": [],
        "outgoing_links": []
    },
    {
        "id": "test_chunk_2",
        "title": "Характеристики битумной мастики",
        "level": 2,
        "elements": [
            {
                "type": "paragraph",
                "text": "Битумная мастика ТЕХНОНИКОЛЬ №21 обладает следующими характеристиками: "
                       "расход - 3,8-5,7 кг/м², время высыхания - 24 часа, эластичность при изгибе - 5 мм."
            },
            {
                "type": "paragraph",
                "text": "Температура применения от -20°C до +40°C. Материал не содержит растворителей, "
                       "что делает его безопасным для здоровья человека и окружающей среды."
            }
        ],
        "paragraphs": [
            {
                "id": "test_chunk_2_p_1",
                "text": "Битумная мастика ТЕХНОНИКОЛЬ №21 обладает следующими характеристиками: "
                       "расход - 3,8-5,7 кг/м², время высыхания - 24 часа, эластичность при изгибе - 5 мм.",
                "parent_chunk_id": "test_chunk_2",
                "parent_chunk_title": "Характеристики битумной мастики"
            },
            {
                "id": "test_chunk_2_p_2",
                "text": "Температура применения от -20°C до +40°C. Материал не содержит растворителей, "
                       "что делает его безопасным для здоровья человека и окружающей среды.",
                "parent_chunk_id": "test_chunk_2",
                "parent_chunk_title": "Характеристики битумной мастики"
            }
        ],
        "children": [],
        "labels": [],
        "paragraph_labels": {},
        "numbers": [],
        "outgoing_links": []
    },
    {
        "id": "test_chunk_3",
        "title": "Нахлест гидроизоляционных мембран",
        "level": 2,
        "elements": [
            {
                "type": "paragraph",
                "text": "При укладке гидроизоляционных мембран необходимо обеспечить нахлест "
                       "не менее 100 мм в продольном направлении и 150 мм в поперечном."
            },
            {
                "type": "paragraph",
                "text": "Швы должны быть герметично проклеены или сварены горячим воздухом, чтобы обеспечить "
                       "полную водонепроницаемость. Особое внимание уделяйте местам пересечения швов."
            }
        ],
        "paragraphs": [
            {
                "id": "test_chunk_3_p_1",
                "text": "При укладке гидроизоляционных мембран необходимо обеспечить нахлест "
                       "не менее 100 мм в продольном направлении и 150 мм в поперечном.",
                "parent_chunk_id": "test_chunk_3",
                "parent_chunk_title": "Нахлест гидроизоляционных мембран"
            },
            {
                "id": "test_chunk_3_p_2",
                "text": "Швы должны быть герметично проклеены или сварены горячим воздухом, чтобы обеспечить "
                       "полную водонепроницаемость. Особое внимание уделяйте местам пересечения швов.",
                "parent_chunk_id": "test_chunk_3",
                "parent_chunk_title": "Нахлест гидроизоляционных мембран"
            }
        ],
        "children": [],
        "labels": [],
        "paragraph_labels": {},
        "numbers": [],
        "outgoing_links": []
    }
]


def test_keyword_classifier():
    """Тестирование классификатора на основе ключевых слов."""
    print("\n=== Тестирование классификатора по ключевым словам ===")
    
    classifier = KeywordClassifier()
    
    for i, chunk in enumerate(TEST_CHUNKS):
        chunk_text = get_chunk_text(chunk)
        print(f"\nЧанк {i+1}: {chunk['title']}")
        print(f"Текст: {chunk_text[:100]}...")
        
        # Классификация чанка
        chunk_labels = classifier.classify_chunk(chunk)
        print(f"Метки чанка: {', '.join(chunk_labels) if chunk_labels else 'Нет меток'}")
        
        # Классификация каждого параграфа
        for j, paragraph in enumerate(chunk["paragraphs"]):
            paragraph_labels = classifier.classify_paragraph(paragraph, chunk)
            print(f"  Параграф {j+1}: {paragraph['text'][:50]}...")
            print(f"  Метки параграфа: {', '.join(paragraph_labels) if paragraph_labels else 'Нет меток'}")


def test_hybrid_classifier(use_bert=False, use_gpt=False):
    """
    Тестирование гибридного классификатора.
    
    Args:
        use_bert: Использовать ли BERT-классификатор
        use_gpt: Использовать ли GPT-классификатор
    """
    classifiers = ["keyword"]
    if use_bert:
        classifiers.append("BERT")
    if use_gpt:
        classifiers.append("GPT")
    
    print(f"\n=== Тестирование гибридного классификатора ({', '.join(classifiers)}) ===")
    
    # Создание гибридного классификатора
    classifier = HybridClassifier(use_bert=use_bert, use_gpt=use_gpt)
    
    for i, chunk in enumerate(TEST_CHUNKS):
        chunk_text = get_chunk_text(chunk)
        print(f"\nЧанк {i+1}: {chunk['title']}")
        print(f"Текст: {chunk_text[:100]}...")
        
        # Классификация чанка
        chunk_labels = classifier.classify_chunk(chunk)
        print(f"Метки чанка: {', '.join(chunk_labels) if chunk_labels else 'Нет меток'}")
        
        # Классификация каждого параграфа
        for j, paragraph in enumerate(chunk["paragraphs"]):
            paragraph_labels = classifier.classify_paragraph(paragraph, chunk)
            print(f"  Параграф {j+1}: {paragraph['text'][:50]}...")
            print(f"  Метки параграфа: {', '.join(paragraph_labels) if paragraph_labels else 'Нет меток'}")


def test_full_pipeline():
    """Тестирование полного пайплайна классификации."""
    print("\n=== Тестирование полного пайплайна классификации ===")
    
    # Копия тестовых данных для модификации
    chunks = json.loads(json.dumps(TEST_CHUNKS))
    
    # Классификация всех чанков и параграфов
    classifier = get_classifier()
    updated_chunks, paragraph_labels = classify_article(chunks, classifier)
    
    # Вывод результатов
    for i, chunk in enumerate(updated_chunks):
        print(f"\nЧанк {i+1}: {chunk['title']}")
        print(f"Метки чанка: {', '.join(chunk['labels']) if chunk['labels'] else 'Нет меток'}")
        
        for j, paragraph in enumerate(chunk["paragraphs"]):
            p_id = paragraph["id"]
            print(f"  Параграф {j+1}: {paragraph['text'][:50]}...")
            
            if p_id in paragraph_labels:
                print(f"  Метки параграфа: {', '.join(paragraph_labels[p_id]) if paragraph_labels[p_id] else 'Нет меток'}")
            else:
                print(f"  Метки параграфа: Нет меток (ID не найден)")


def test_bert_standalone():
    """Тестирование BERT-классификатора отдельно."""
    print("\n=== Тестирование BERT-классификатора ===")
    
    try:
        # Проверка наличия transformers
        import transformers
        
        # Создание BERT-классификатора
        classifier = BertClassifier()
        
        # Проверка, успешно ли инициализирована модель
        if not classifier.model:
            print("BERT-модель не инициализирована. Пропуск теста.")
            return
        
        for i, chunk in enumerate(TEST_CHUNKS):
            print(f"\nЧанк {i+1}: {chunk['title']}")
            
            # Классификация чанка
            chunk_labels = classifier.classify_chunk(chunk)
            print(f"BERT-метки чанка: {', '.join(chunk_labels) if chunk_labels else 'Нет меток'}")
            
            # Классификация первого параграфа
            if chunk["paragraphs"]:
                paragraph = chunk["paragraphs"][0]
                paragraph_labels = classifier.classify_paragraph(paragraph, chunk)
                print(f"BERT-метки параграфа: {', '.join(paragraph_labels) if paragraph_labels else 'Нет меток'}")
        
    except ImportError:
        print("Библиотека transformers не установлена. Тест BERT пропущен.")


def test_gpt_standalone():
    """Тестирование GPT-классификатора отдельно."""
    print("\n=== Тестирование GPT-классификатора ===")
    
    try:
        # Проверка наличия openai
        import openai
        
        # Получение API-ключа из переменной окружения
        api_key = os.environ.get("OPENAI_API_KEY")
        
        if not api_key:
            print("API-ключ OpenAI не найден в переменных окружения. Пропуск теста.")
            return
        
        # Создание GPT-классификатора
        classifier = GPTClassifier(api_key)
        
        for i, chunk in enumerate(TEST_CHUNKS):
            print(f"\nЧанк {i+1}: {chunk['title']}")
            
            # Классификация чанка
            chunk_labels = classifier.classify_chunk(chunk)
            print(f"GPT-метки чанка: {', '.join(chunk_labels) if chunk_labels else 'Нет меток'}")
            
            # Классификация первого параграфа
            if chunk["paragraphs"]:
                paragraph = chunk["paragraphs"][0]
                paragraph_labels = classifier.classify_paragraph(paragraph, chunk)
                print(f"GPT-метки параграфа: {', '.join(paragraph_labels) if paragraph_labels else 'Нет меток'}")
        
    except ImportError:
        print("Библиотека openai не установлена. Тест GPT пропущен.")


def main():
    """Основная функция для запуска тестов."""
    print("=== Тестирование двухуровневой классификации контента ===")
    
    # Тест классификатора по ключевым словам
    test_keyword_classifier()
    
    # Тест BERT-классификатора
    test_bert_standalone()
    
    # Тест GPT-классификатора
    test_gpt_standalone()
    
    # Тест гибридного классификатора (только ключевые слова)
    test_hybrid_classifier()
    
    # Тест гибридного классификатора с BERT (если доступен)
    try:
        import transformers
        test_hybrid_classifier(use_bert=True)
    except ImportError:
        print("\nТест гибридного классификатора с BERT пропущен (transformers не установлен)")
    
    # Тест гибридного классификатора с GPT (если доступен и указан ключ API)
    try:
        import openai
        if os.environ.get("OPENAI_API_KEY"):
            test_hybrid_classifier(use_gpt=True)
        else:
            print("\nТест гибридного классификатора с GPT пропущен (нет API-ключа)")
    except ImportError:
        print("\nТест гибридного классификатора с GPT пропущен (openai не установлен)")
    
    # Тест полного пайплайна классификации
    test_full_pipeline()


if __name__ == "__main__":
    main()
