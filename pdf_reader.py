import fitz
import re

# Загрузка PDF
def read_pdf(pdf_path):
    doc = fitz.open(pdf_path)
    # Читаем весь текст
    full_text = "\n".join([page.get_text() for page in doc])

    # Удаляем символы <*> и лишние пробелы
    full_text = re.sub(r"<.*?>", "", full_text)
    full_text = re.sub(r"[ \t]+", " ", full_text)  
    full_text = full_text.strip().replace('\n', ' ')
    # Регулярное выражение для поиска заголовков статей
    article_pattern = re.compile(r"(Статья \d+)", re.MULTILINE)

    # Разбиваем текст по заголовкам статей
    articles = article_pattern.split(full_text)

    # Создаем словарь {Статья: [пункты]}
    article_dict = {}

    for i in range(1, len(articles), 2):  # Четные индексы - заголовки, нечетные - текст статьи
        article_title = articles[i].strip()
        article_body = articles[i + 1].strip()
        
        # Разбиваем текст статьи на пункты по номерам (1., 2., 3.)
        points = re.split(r"\d+\.\s", article_body)
        
        # Восстанавливаем номера пунктов
        points = [f"{idx-1}. {p.strip()}" if idx > 1 else f"{p.strip()}" for idx, p in enumerate(points, start=1) if p.strip()]
        
        # Сохраняем в словарь
        article_dict[article_title] = points
    return article_dict
