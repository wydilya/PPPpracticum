import base64
from io import BytesIO
import numpy as np
from fastapi import FastAPI
from PIL import Image
from pydantic import BaseModel

import base64
from io import BytesIO
import numpy as np
from fastapi import FastAPI
from PIL import Image
from pydantic import BaseModel
import requests

app = FastAPI()


class ImageRequest(BaseModel):
    image_base64: str


def base64_to_image(image_base64: str) -> Image.Image:
    image_data = base64.b64decode(image_base64)
    return Image.open(BytesIO(image_data))


def image_to_base64(image: Image.Image) -> str:
    buffer = BytesIO()
    image.save(buffer, format="PNG")
    return base64.b64encode(buffer.getvalue()).decode("utf-8")


def otsu_threshold(image: Image.Image) -> Image.Image:
    gray_image = np.array(image.convert("L"))
    hist = np.histogram(gray_image, bins=256, range=(0, 256))[0].astype(float)
    total_pixels = hist.sum()

    if total_pixels == 0:
        return image

    probabilities = hist / total_pixels
    cumulative_sum = np.cumsum(probabilities)
    cumulative_mean = np.cumsum(probabilities * np.arange(256))

    max_variance = -1
    optimal_threshold = 0

    for threshold in range(1, 256):
        w0 = cumulative_sum[threshold]
        w1 = 1 - w0

        if w0 == 0 or w1 == 0:
            continue

        mu0 = cumulative_mean[threshold] / w0
        mu1 = (cumulative_mean[-1] - cumulative_mean[threshold]) / w1
        variance = w0 * w1 * (mu0 - mu1) ** 2

        if variance > max_variance:
            max_variance = variance
            optimal_threshold = threshold

    binary_image = np.where(gray_image > optimal_threshold, 255, 0).astype(np.uint8)
    return Image.fromarray(binary_image)


@app.post("/binarize")
async def binarize_image(request: ImageRequest):
    try:
        image = base64_to_image(request.image_base64)
        processed_image = otsu_threshold(image)
        return {"result_image": image_to_base64(processed_image)}
    except Exception as e:
        return {"error": str(e)}


# Пример использования прямо в коде
if __name__ == "__main__":
    import uvicorn
    from threading import Thread
    import time

    # Запуск сервера в отдельном потоке
    server = Thread(target=uvicorn.run, kwargs={"app": app, "port": 8000})
    server.daemon = True
    server.start()

    # Даем серверу время на запуск
    time.sleep(2)

    try:
        # Чтение тестового изображения
        with open("foto.png", "rb") as f:
            test_image_base64 = base64.b64encode(f.read()).decode("utf-8")

        # Отправка запроса
        response = requests.post(
            "http://localhost:8000/binarize",
            json={"image_base64": test_image_base64}
        )

        # Обработка результата
        if response.status_code == 200:
            result = response.json()
            with open("result_image.png", "wb") as f:
                f.write(base64.b64decode(result["result_image"]))
            print("✅ Успех! Результат сохранен в result_image.png")
        else:
            print("❌ Ошибка:", response.json())

    except Exception as e:
        print("🔥 Ошибка при выполнении теста:", str(e))
    finally:
        # Остановка сервера
        server.join(0.1)