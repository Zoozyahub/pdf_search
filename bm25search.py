from rank_bm25 import BM25Okapi

def init_bm25(lemmatized_article_dict):
    # Создаём корпус документов для BM25
    corpus = [" ".join([" ".join(p) for p in points]) for points in lemmatized_article_dict.values()]
    tokenized_corpus = [doc.split() for doc in corpus]  # Разделяем по словам
    bm25 = BM25Okapi(tokenized_corpus)
    return bm25

def search_bm25(query, tokenizer, article_dict, lemmatized_article_dict, bm25, df, top_n=3):
    query = query.strip()
    
    # Если запрос выглядит как "Статья X", возвращаем её текст и картинку
    if query.lower().startswith("статья"):
        if query in article_dict:
            article_index = int(query[7:])  # Извлекаем номер статьи (например, "Статья 15" -> 15)
            return [(query, " ".join(article_dict[query]), df.iloc[article_index]['Путь к подходящей картинке'])]
        else:
            return [("Ошибка", f"Статья '{query}' не найдена", None)]
    
    # Поиск с BM25
    query_tokens = tokenizer(query) 
    scores = bm25.get_scores(query_tokens)  # Получаем оценки релевантности
    top_indexes = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)[:top_n]  # Топ-n результатов
    top_articles = list(lemmatized_article_dict.keys())  # Список названий статей
    
    # Формирование результатов
    results = []
    for i in top_indexes:
        article_title = top_articles[i]
        original_text = "\n".join(article_dict[article_title])  # Восстанавливаем исходный текст
        image_path = df.iloc[i]['Путь к подходящей картинке']  # Путь к картинке
        results.append((article_title, original_text, image_path))
    
    return results