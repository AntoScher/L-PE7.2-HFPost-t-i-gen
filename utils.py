import random

# Примеры промптов и параметров
example_prompts = [
    "Магический лес с светящимися растениями и домиками фей среди гигантских грибов",
    "Старинный стимпанк-дирижабль, плывущий по золотым облакам на закате"
]

art_styles = ["реализм", "аниме", "цифровое искусство", "кинематографичный", "неоновый панк"]
lighting_options = ["мягкий свет", "драматичное освещение", "неоновые огни", "солнечные лучи"]


def generate_image_prompt(theme="фэнтези"):
    base_prompt = random.choice(example_prompts)
    style = random.choice(art_styles)
    lighting = random.choice(lighting_options)
    negative_prompt = "размытость, деформированные конечности, watermark"  # Из :cite[6]

    return (
        f"{base_prompt}, {theme}. Стиль: {style}. Освещение: {lighting}. "
        f"Высокая детализация, 4K. Нежелательно: {negative_prompt}"
    )
