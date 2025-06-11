from transformers import pipeline
import torch

# Инициализация модели с кэшированием
model_name = "bigscience/bloom-560m"
device = "cuda" if torch.cuda.is_available() else "cpu"

# Создаем пайплайн один раз при загрузке модуля
generator = pipeline(
    'text-generation',
    model=model_name,
    device=device,
    torch_dtype=torch.float16 if device == "cuda" else torch.float32
)


def generate_prompt(theme="фэнтези"):
    # Системный промпт для направления генерации
    system_prompt = (
        "Ты генератор промптов для нейросети-художника. Создай один детализированный промпт на тему: "
        f"{theme}. Включи описание сцены, стиль искусства, освещение, детали. "
        "Используй ключевые слова через запятую. Без лишних слов."
    )

    try:
        # Генерация текста с контролем длины
        result = generator(
            system_prompt,
            max_new_tokens=50,  # Ограничение длины вывода
            num_return_sequences=1,  # Только один вариант
            temperature=0.9,  # Баланс креативности
            top_k=50,  # Контроль разнообразия
            truncation=True  # Обрезать длинные входы
        )

        return result[0]['generated_text'].replace(system_prompt, '').strip()

    except Exception as e:
        print(f"Ошибка генерации: {e}")
        # Возвращаем запасной вариант
        return f"детализированная сцена в стиле фэнтези на тему {theme}, цифровое искусство, высокая детализация"