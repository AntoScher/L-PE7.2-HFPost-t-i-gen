from diffusers import StableDiffusionPipeline
import torch
import os
from datetime import datetime

# Конфигурация
MODEL_NAME = "runwayml/stable-diffusion-v1-5"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
OUTPUT_DIR = "generated_images"

# Создаем папку для результатов
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Загружаем модель один раз
pipe = StableDiffusionPipeline.from_pretrained(
    MODEL_NAME,
    torch_dtype=torch.float16 if DEVICE == "cuda" else torch.float32
)
pipe = pipe.to(DEVICE)


def generate_image(prompt):
    try:
        # Генерация изображения
        image = pipe(
            prompt,
            num_inference_steps=30,  # Оптимальное качество/скорость
            guidance_scale=7.5,  # Баланс креативности
            width=512,
            height=512
        ).images[0]

        # Сохранение с уникальным именем
        timestamp = datetime.now().strftime("%Y%m%d%H%M%S")
        filename = f"image_{timestamp}.png"
        image_path = os.path.join(OUTPUT_DIR, filename)
        image.save(image_path)

        return image_path

    except Exception as e:
        print(f"Ошибка генерации изображения: {e}")
        return None