import requests
from flask import Flask, render_template, request, jsonify
import utils  # Импорт генератора промптов
import math
import os
import random
from concurrent.futures import ThreadPoolExecutor
from PIL import Image
import io
from dotenv import load_dotenv

load_dotenv()

app = Flask(__name__)

# Конфигурация
HUGGING_FACE_API_KEY = os.getenv("HUGGING_FACE_API_KEY")
API_URL = "https://api-inference.huggingface.co/models/"  # Добавлен базовый URL

MODELS = [
    "CompVis/stable-diffusion-v1-4"
    #"stabilityai/stable-diffusion-3.5-medium",
    # "runwayml/stable-diffusion-v1-5",
    # "CompVis/stable-diffusion-v1-4"
]
ASPECT_RATIOS = {
    "1/1": "Square",
    "16/9": "Landscape",
    "9/16": "Portrait"
}
EXAMPLE_PROMPTS = [
    "A magic forest with glowing plants and fairy homes among giant mushrooms",
    "An old steampunk airship floating through golden clouds at sunset",
    "A future Mars colony with glass domes and gardens against red mountains",
]


def get_image_dimensions(aspect_ratio, base_size=512):
    """Рассчитывает размеры изображения с учетом соотношения сторон"""
    width_ratio, height_ratio = map(int, aspect_ratio.split('/'))
    scale_factor = base_size / math.sqrt(width_ratio * height_ratio)

    width = int(width_ratio * scale_factor)
    height = int(height_ratio * scale_factor)

    # Приведение к кратности 16
    width = (width // 16) * 16
    height = (height // 16) * 16

    return width, height


def generate_single_image(prompt, model, aspect_ratio, index):
    """Генерирует одно изображение через Hugging Face API"""
    width, height = get_image_dimensions(aspect_ratio)
    url = API_URL + model  # Формируем полный URL

    try:
        response = requests.post(
            url,
            headers={
                "Authorization": f"Bearer {HUGGING_FACE_API_KEY}",
                "Content-Type": "application/json",
                "x-use-cache": "false"
            },
            json={
                "inputs": prompt,
                "parameters": {
                    "width": width,
                    "height": height
                }
            }
        )

        if response.status_code != 200:
            raise Exception(f"API Error: {response.status_code} - {response.text}")

        return index, response.content, None

    except Exception as e:
        return index, None, str(e)


def save_image(image_data, filename):
    """Сохраняет изображение на диск"""
    try:
        with open(filename, 'wb') as f:
            f.write(image_data)
        return True
    except Exception as e:
        print(f"Error saving image: {str(e)}")
        return False


# Маршруты Flask
@app.route('/')
def index():
    return render_template('index.html')


@app.route('/generate_prompt', methods=['POST'])
def generate_prompt():
    theme = request.json.get('theme', '')
    prompt = utils.generate_image_prompt(theme)
    return jsonify({"prompt": prompt})


# Добавляем новый эндпоинт для генерации
@app.route('/generate_images', methods=['POST'])
def generate_images():
    data = request.json
    prompt = data.get('prompt', 'a beautiful landscape with mountains and lake')
    model = data.get('model', MODELS[0])
    count = int(data.get('count', 4))
    ratio = data.get('ratio', "16/9")

    # Генерация изображений
    results = []
    with ThreadPoolExecutor() as executor:
        futures = []
        for i in range(count):
            futures.append(executor.submit(
                generate_single_image,
                prompt,
                model,
                ratio,
                i
            ))

        for future in futures:
            index, img_data, error = future.result()
            if img_data:
                filename = f"image_{index}.png"
                if save_image(img_data, filename):
                    results.append(filename)
                else:
                    results.append(f"Error saving {filename}")
            else:
                results.append(f"Error generating image {index}: {error}")

    return jsonify({"results": results})


if __name__ == '__main__':
    app.run(debug=True)