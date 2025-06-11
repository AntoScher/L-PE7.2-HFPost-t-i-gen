from transformers import AutoTokenizer, GPT2LMHeadModel
import torch
import logging

logging.basicConfig(level=logging.INFO)

# Используем более легкую и рабочую модель
model_name = "sberbank-ai/rugpt3small_based_on_gpt2"
device = "cuda" if torch.cuda.is_available() else "cpu"

# Загружаем модель и токенизатор
try:
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = GPT2LMHeadModel.from_pretrained(model_name).to(device)
    logging.info(f"Модель {model_name} успешно загружена на устройство {device}")
except Exception as e:
    logging.error(f"Ошибка загрузки модели: {e}")
    tokenizer = None
    model = None


def generate_prompt(theme="фэнтези"):
    if not model or not tokenizer:
        logging.error("Модель или токенизатор не загружены")
        return f"{theme}, цифровое искусство, высокая детализация, 4K"

    # Системный промпт
    system_prompt = (
        "Ты генератор промптов для нейросети-художника. Создай один детализированный промпт на тему: "
        f"{theme}. Формат: [Объект], [стиль], [освещение], [детали]. Ключевые слова через запятую."
    )

    try:
        # Токенизируем вход
        inputs = tokenizer(system_prompt, return_tensors="pt", max_length=512, truncation=True).to(device)

        # Генерация текста
        outputs = model.generate(
            inputs.input_ids,
            max_new_tokens=50,
            num_return_sequences=1,
            temperature=0.9,
            top_k=50,
            pad_token_id=tokenizer.eos_token_id,
            eos_token_id=tokenizer.eos_token_id,
            do_sample=True,
            no_repeat_ngram_size=2  # Предотвращаем повторения
        )

        # Декодируем результат
        generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)

        # Удаляем исходный промпт из результата
        result = generated_text.replace(system_prompt, "").strip()

        # Удаляем возможные повторения в конце
        if result.endswith(theme):
            result = result.replace(theme, "").strip()

        logging.info(f"Сгенерирован промпт: {result}")
        return result

    except Exception as e:
        logging.error(f"Ошибка генерации промпта: {e}")
        return f"{theme}, цифровое искусство, высокая детализация, 4K"