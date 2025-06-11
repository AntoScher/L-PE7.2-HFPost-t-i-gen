from diffusers import StableDiffusionPipeline
import torch
import os
from datetime import datetime
import logging

logging.basicConfig(level=logging.INFO)

# Конфигурация
MODEL_NAME = "runwayml/stable-diffusion-v1-5"
OUTPUT_DIR = "generated_images"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Проверка доступности GPU
device = "cuda" if torch.cuda.is_available() else "cpu"
dtype = torch.float16 if device == "cuda" else torch.float32
logging.info(f"Используемое устройство: {device}, тип данных: {dtype}")

# Инициализация pipeline с обработкой ошибок
try:
    pipe = StableDiffusionPipeline.from_pretrained(
        MODEL_NAME,
        torch_dtype=dtype,
        safety_checker=None  # Отключаем встроенный NSFW-фильтр
    )
    pipe = pipe.to(device)
    logging.info(f"Модель {MODEL_NAME} успешно загружена")
except Exception as e:
    logging.error(f"Ошибка загрузки модели: {e}")
    pipe = None


def generate_image(prompt):
    if not pipe:
        logging.error("Pipeline не инициализирован")
        return None

    try:
        # Генерация изображения
        logging.info(f"Генерация изображения для промпта: {prompt}")
        result = pipe(
            prompt,
            num_inference_steps=30,
            guidance_scale=7.5,
            width=512,
            height=512
        )

        # Проверка результата
        if not result.images or len(result.images) == 0:
            logging.error("Не удалось сгенерировать изображение")
            return None

        image = result.images[0]

        # Сохранение с уникальным именем
        timestamp = datetime.now().strftime("%Y%m%d%H%M%S")
        filename = f"image_{timestamp}.png"
        image_path = os.path.join(OUTPUT_DIR, filename)
        image.save(image_path)
        logging.info(f"Изображение сохранено: {image_path}")

        return image_path

    except Exception as e:
        logging.error(f"Ошибка генерации изображения: {e}")
        return None