from flask import Flask, render_template, request, jsonify, send_from_directory
from prompt_generator import generate_prompt
from image_generator import generate_image
import os

app = Flask(__name__)


@app.route('/')
def index():
    return render_template('index-DS.html')


@app.route('/generate', methods=['POST'])
def generate():
    data = request.json
    theme = data.get('theme', '')

    # Генерация промпта
    prompt = generate_prompt(theme)

    # Генерация изображения
    image_path = generate_image(prompt)

    return jsonify({
        'prompt': prompt,
        'image_url': f'/images/{os.path.basename(image_path)}'
    })


@app.route('/images/<filename>')
def serve_image(filename):
    return send_from_directory('generated_images', filename)


if __name__ == '__main__':
    app.run(debug=True)