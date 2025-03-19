from rank_bm25 import BM25Okapi

def init_bm25(lemmatized_article_dict):
    # создаём корпус документов для BM25
    corpus = [" ".join([" ".join(p) for p in points]) for points in lemmatized_article_dict.values()]
    tokenized_corpus = [doc.split() for doc in corpus]  # разделяем по словам
    bm25 = BM25Okapi(tokenized_corpus)
    return bm25

# Функция поиска
def search_bm25(query, tokenizer, article_dict, lemmatized_article_dict, bm25,top_n=3):
    query = query.strip()
    
    # Если запрос выглядит как "Статья X" возвращаем её текст
    if query.lower().startswith("статья"):
        if query in article_dict:
            return [(query, "\n".join(article_dict[query]))]
        else:
            return [("Ошибка", f"Статья '{query}' не найдена")]
    
    # поиск с BM25
    query_tokens = tokenizer(query) 
    scores = bm25.get_scores(query_tokens)  # получаем оценки релевантности
    top_indexes = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)[:top_n]  # Топ-n результатов
    top_articles = list(lemmatized_article_dict.keys())  # список названий статей
    
    results = []
    for i in top_indexes:
        article_title = top_articles[i]
        original_text = "\n".join(article_dict[article_title])  # восстанавливаем исходный текст
        results.append((article_title, original_text))
    
    return results