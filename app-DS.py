from flask import Flask, render_template, request, jsonify, send_from_directory
from prompt_generator import generate_prompt
from image_generator import generate_image
import os
import logging

logging.basicConfig(level=logging.INFO)
app = Flask(__name__)


@app.route('/')
def index():
    return render_template('index-DS.html')


@app.route('/generate', methods=['POST'])
def generate():
    data = request.json
    theme = data.get('theme', '')
    logging.info(f"Получен запрос на генерацию для темы: {theme}")

    # Генерация промпта
    prompt = generate_prompt(theme)

    # Генерация изображения
    image_path = generate_image(prompt)

    if not image_path:
        logging.error("Не удалось сгенерировать изображение")
        return jsonify({
            'prompt': prompt,
            'error': 'Не удалось сгенерировать изображение'
        }), 500

    return jsonify({
        'prompt': prompt,
        'image_url': f'/images/{os.path.basename(image_path)}'
    })


@app.route('/images/<filename>')
def serve_image(filename):
    try:
        return send_from_directory('generated_images', filename)
    except FileNotFoundError:
        return "Изображение не найдено", 404


if __name__ == '__main__':
    app.run(debug=True, use_reloader=False)  # use_reloader=False для избежания двойной инициализации моделей