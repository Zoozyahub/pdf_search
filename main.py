from pdf_reader import read_pdf
from lemmatization import create_lemma_dict, clean_and_lemmatize
from bm25search import init_bm25, search_bm25
from search_by_image import search_by_image_sklearn, display_results
import os
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.image as mpimg

if __name__ == '__main__':
    print('Инициализация...')
    BASE_DIR = os.path.dirname(os.path.abspath(__file__))  # Папка, где main.py
    pdf_path = os.path.join(BASE_DIR, "data/constitution_rf.pdf") 
    article_dict = dict(read_pdf(pdf_path))
    lemmatized_article_dict = dict(create_lemma_dict(article_dict))
    bm25 = init_bm25(lemmatized_article_dict)
    
    # Загрузка DataFrame с картинками
    df = pd.read_csv(os.path.join(BASE_DIR, "articles_with_images.csv"), encoding="utf-8")
    
    print('Для завершения введите "стоп":')
    while True:
        print('Выберите тип поиска:')
        print('1. Поиск по тексту')
        print('2. Поиск по картинке')
        choice = input("Введите номер (1 или 2): ")
        
        if choice == 'стоп':
            break
        
        if choice == '1':
            # Поиск по тексту
            print('Введите текстовый запрос:')
            query = input()
            if query == 'стоп':
                break
            
            # Поиск статей
            results = search_bm25(query, clean_and_lemmatize, article_dict, lemmatized_article_dict, bm25, df, top_n=3)
            
            # Вывод результатов
            for title, text, image_path in results:
                print(f"\n{title}:\n{text}")
                if image_path:
                    print(f"Путь к картинке: {image_path}")
                    try:
                        img = mpimg.imread(image_path)
                        plt.imshow(img)
                        plt.axis("off")  # Скрыть оси
                        plt.show()
                    except FileNotFoundError:
                        print(f"Ошибка: Картинка по пути '{image_path}' не найдена.")
                    except Exception as e:
                        print(f"Ошибка при загрузке картинки: {e}")
                else:
                    print("Картинка не найдена")
                
        elif choice == '2':
            # Поиск по картинке
            print('Введите путь к картинке:')
            image_path = input()
            if image_path == 'стоп':
                break
            
            # Поиск статей по картинке
            results = search_by_image_sklearn(image_path, df, top_n=3)
            
            # Вывод результатов
            display_results(results)
        
        else:
            print('Неверный выбор. Введите 1 или 2.')