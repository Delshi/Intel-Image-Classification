import matplotlib.pyplot as plt
import numpy as np
import io
from PIL import Image


def create_3d_histogram(data, title, epoch):
    """Создает 3D-гистограмму с перспективой"""
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection="3d")

    # Создаем данные для 3D гистограммы
    hist, bins = np.histogram(data, bins=50)
    xs = (bins[:-1] + bins[1:]) / 2

    # Создаем 3D бар-плот
    ax.bar(xs, hist, zs=epoch, zdir="y", alpha=0.8, width=(bins[1] - bins[0]) * 0.8)

    ax.set_xlabel("Value")
    ax.set_ylabel("Epoch")
    ax.set_zlabel("Frequency")
    ax.set_title(f"{title} - Epoch {epoch}")

    # Угол обзора
    ax.view_init(elev=20, azim=45)

    # Конвертируем в изображение
    buf = io.BytesIO()
    plt.savefig(buf, format="png", dpi=150, bbox_inches="tight")
    buf.seek(0)
    image = Image.open(buf)
    image_array = np.array(image)
    plt.close(fig)

    return image_array
