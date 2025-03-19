import spacy
import string
import re

nlp = spacy.load("ru_core_news_sm")

def clean_and_lemmatize(text):
    text = text.lower().strip().replace("\n", " ")  # Убираем лишние пробелы и переносы строк
    text = re.sub(r"\s+", " ", text)  # Заменяем множественные пробелы одним
    text = re.sub(r"\d+", "", text)  # Удаляем цифры
    text = text.translate(str.maketrans("", "", string.punctuation))  # Убираем знаки препинания
    doc = nlp(text)
    return [token.lemma_ for token in doc if not token.is_stop and token.lemma_.strip()]  # Убираем пустые строки

# Создаём новый словарь с лемматизированными статьями
def create_lemma_dict(article_dict):
    lemmatized_article_dict = {
        key: [clean_and_lemmatize(p) for p in points if p.strip()]  # Убираем пустые пункты
        for key, points in article_dict.items()
    }
    return lemmatized_article_dict