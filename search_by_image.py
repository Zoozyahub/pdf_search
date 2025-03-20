import torch
from transformers import CLIPProcessor, CLIPModel
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import json

# Инициализация модели CLIP
model_path = "C:/Users/Ярослав/.cache/huggingface/hub/models--openai--clip-vit-base-patch32/snapshots/3d74acf9a28c67741b2f4f2ea7635f0aaf6f0268"
device = "cuda" if torch.cuda.is_available() else "cpu"
model = CLIPModel.from_pretrained(model_path).to(device)
processor = CLIPProcessor.from_pretrained(model_path)

def encode_image(img_path):
    image = Image.open(img_path).convert("RGB")
    inputs = processor(images=image, return_tensors="pt").to(device)

    with torch.no_grad():
        img_embedding = model.get_image_features(**inputs).cpu().numpy().flatten()
    
    return img_embedding

def search_by_image_sklearn(image_path, df, top_n=3):
    # Векторизация загруженной картинки
    image_embedding = encode_image(image_path).reshape(1, -1)
    
    # Загрузка текстовых эмбеддингов из DataFrame
    df["Код_эмбеддинг_CLIP"] = df["Код_эмбеддинг_CLIP"].apply(lambda x: json.loads(x) if isinstance(x, str) else x)
    text_embeddings = np.array(df["Код_эмбеддинг_CLIP"].tolist())
    
    # Вычисление косинусного сходства
    similarities = cosine_similarity(image_embedding, text_embeddings)[0]
    
    # Получение индексов топ-N статей
    top_indexes = similarities.argsort()[-top_n:][::-1]
    
    # Формирование результатов
    results = []
    for idx in top_indexes:
        article_title = df.iloc[idx]["Статья"]
        article_text = df.iloc[idx]["Текст статьи"]
        article_image_path = df.iloc[idx]["Путь к подходящей картинке"]
        results.append((article_title, article_text, article_image_path, similarities[idx]))
    
    return results


def display_results(results):
    print("\nРезультаты поиска по картинке (sklearn):")
    for title, text, image_path, similarity in results:
        print(f"\n{title} (сходство: {similarity:.2f}):\n{text}")
        if image_path:
            try:
                img = mpimg.imread(image_path)
                plt.imshow(img)
                plt.axis("off")  
                plt.show()
            except FileNotFoundError:
                print(f"Ошибка: Картинка по пути '{image_path}' не найдена.")
            except Exception as e:
                print(f"Ошибка при загрузке картинки: {e}")
        else:
            print("Картинка не найдена")
