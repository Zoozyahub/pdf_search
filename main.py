from pdf_reader import read_pdf
from lemmatization import create_lemma_dict, clean_and_lemmatize
from bm25search import init_bm25, search_bm25
import os

if __name__ == '__main__':
    print('Инициализация...')
    BASE_DIR = os.path.dirname(os.path.abspath(__file__))  # папка, где main.py
    pdf_path = os.path.join(BASE_DIR, "data/constitution_rf.pdf") 
    article_dict = dict(read_pdf(pdf_path))
    lemmatized_article_dict = dict(create_lemma_dict(article_dict))
    bm25 = init_bm25(lemmatized_article_dict)
    print('Для завершения введите стоп:')
    while True:
        print('Введите запрос:')
        query = input()
        if query == 'стоп':
            break
        result = search_bm25(query, clean_and_lemmatize, article_dict, lemmatized_article_dict, bm25,top_n=3)   
        for title, text in result:
            print(f"\n{title}:\n{text}")