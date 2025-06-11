import requests

url = "http://localhost:5000/generate_images"
data = {
    "prompt": "Закат над океаном",
    "model": "stabilityai/stable-diffusion-2-1",
    "count": 2,
    "ratio": "16/9"
}

response = requests.post(url, json=data)
print("Результат:", response.json())